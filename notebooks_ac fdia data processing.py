import pandas as pd
import pandapower as pp
import numpy as np
from os import path

# --- Define the Attack Region and Boundaries ---

# 1. Define Attacking Buses (1-based ID)
# ATTACK_REGION_IDS = list(range(100, 113))
ATTACK_REGION_IDS = list(range(10, 13))

# 2. Load the Network
# You must ensure pandapower is installed and run this locally.
# net = pp.networks.case118()
net = pp.networks.case14()

# Convert IDs to 0-based indices used by pandapower internally
attack_region_indices = [i - 1 for i in ATTACK_REGION_IDS]
boundary_bus_indices = set()



# Iterate over all lines in the network
for index, line in net.line.iterrows():
    from_bus_idx = line.from_bus
    to_bus_idx = line.to_bus
    
    # Check if a line connects the attack region to the non-attack region
    
    from_in_SA = from_bus_idx in attack_region_indices
    to_in_SA = to_bus_idx in attack_region_indices

    #print(f"Line {index}: from_bus {from_bus_idx} (in_SA={from_in_SA}) to_bus {to_bus_idx} (in_SA={to_in_SA})")
    
    # A boundary condition is met if one end is IN the attack region (in_SA)
    # and the other end is OUTSIDE the attack region (not in_SA).
    
    if from_in_SA and not to_in_SA:
        # 'from' bus is in SA, 'to' bus is outside (Boundary Bus found at 'from')
        boundary_bus_indices.add(from_bus_idx)
        
    elif to_in_SA and not from_in_SA:
        # 'to' bus is in SA, 'from' bus is outside (Boundary Bus found at 'to')
        boundary_bus_indices.add(to_bus_idx)



# Convert the 0-based indices back to 1-based IDs for reporting
boundary_bus_ids = [i + 1 for i in sorted(list(boundary_bus_indices))]

print(f"Attack Region (S_A): Buses {ATTACK_REGION_IDS}")
print(f"Non-Attack Region (S_N): All other buses (1-99 and 113-118)")
print(f"Boundary Buses connecting S_A to S_N: {boundary_bus_ids}")



tie_lines_info = {}

for index, line in net.line.iterrows(): 
    from_bus_id = line.from_bus 
    to_bus_id = line.to_bus 

    from_bus_in_attack_region = from_bus_id in attack_region_indices
    to_bus_in_attack_region = to_bus_id in attack_region_indices

    # Check for a tie line (one end in SA, one end in SN)
    if from_bus_in_attack_region != to_bus_in_attack_region: 
        
        if from_bus_in_attack_region:
            # Line is (Boundary Bus) -> (External Bus)
            boundary_bus = from_bus_id
            external_bus = to_bus_id
            # Flow to subtract is p_from_mw
            flow_key = 'p_from_mw'
        else:
            # Line is (External Bus) -> (Boundary Bus)
            boundary_bus = to_bus_id
            external_bus = from_bus_id
            # Flow to subtract is p_to_mw (flow at the 'to' end)
            flow_key = 'p_to_mw'
            
        # Store all necessary info for Equating/Restoring
        tie_lines_info[index] = {
            'boundary_bus_idx': boundary_bus,
            'external_bus_idx': external_bus,
            'flow_key': flow_key # Tells us where to read the flow from the results
        }
        
print(tie_lines_info)

print(f"Attack Region (S_A): Buses {attack_region_indices}")
print(f"Boundary Buses connecting S_A to S_N: {boundary_bus_indices}")
print("Tie Lines information: \n",tie_lines_info)

internal_lines_info = {}

for index, line in net.line.iterrows():
    # print("net.line.iterrows().index is")
    # print(index)
    # print("net.line.iterrows().line is")
    # print(line)
    from_bus_id = line.from_bus
    to_bus_id = line.to_bus

    # Check if both buses are in attack region
    if from_bus_id in attack_region_indices and to_bus_id in attack_region_indices:
        # Store info: both buses, line index, and which side to use for flow
        # We'll use 'from_bus' side as standard
        print("from bus index is ")
        print(from_bus_id)
        print("to bus index is ")
        print(to_bus_id)
        internal_lines_info[index] = {
            'from_bus_idx': from_bus_id,
            'to_bus_idx': to_bus_id,
            'p_key': 'p_from_mw',   # flow to read from res_line
            'q_key': 'q_from_mvar'
        }

# Example output
print(internal_lines_info)


# Phase 2: Loading dataset and calculating the total external flow 

# data_file = r"D:\FDI\OneDrive_1_2-16-2025\IEEE118Normal_Corrected.csv"
data_file = r"D:\FDI\_FromMarzia\_FromMarzia\FDIG\FDIG\ieee14_nonzero_pg_dataset.csv"

# Load the raw dataset

try:
    df_raw = pd.read_csv(data_file)
    print(f"Data loaded successfully. Rows: {len(df_raw)}, Columns: {len(df_raw.columns)}")
except Exception as e:
    print(f"Error loading file: {e}")


print(df_raw.columns)

selected_columns = [col for col in df_raw.columns if col.endswith(('PG', 'PL','QG','QL'))]

print(selected_columns)
df_z=df_raw[selected_columns].copy()

net_injection_df = pd.DataFrame()

# for bus_id in range(0,118):
for bus_id in range(0,14):
    # Define column names
    pg_col = f"GenBus{bus_id+1}_PG"
    pl_col = f"Bus{bus_id+1}_PL"
    qg_col = f"GenBus{bus_id+1}_QG"
    ql_col = f"Bus{bus_id+1}_QL"

    # Define new result column names
    net_p_col = f"P{bus_id+1}"
    net_q_col = f"Q{bus_id+1}"

    # Compute safely: if the column doesn't exist, treat as 0
    PG = df_z[pg_col] if pg_col in df_z.columns else 0
    PL = df_z[pl_col] if pl_col in df_z.columns else 0
    QG = df_z[qg_col] if qg_col in df_z.columns else 0
    QL = df_z[ql_col] if ql_col in df_z.columns else 0

    # Store results in the new DataFrame
    net_injection_df[net_p_col] = PG - PL
    net_injection_df[net_q_col] = QG - QL



import re


def sort_by_power_type_and_number(column_name):
    """
    Creates a two-part key for sorting: (Type, Bus ID).
    1. Type: 0 for 'P' (Active Power), 1 for 'Q' (Reactive Power).
    2. Bus ID: The extracted integer (1, 2, 3...).
    """
    
    # 1. Extract Type Key (P=0, Q=1)
    power_type = column_name[0]
    type_key = 0 if power_type == 'P' else 1
    
    # 2. Extract Numerical Bus ID
    # Pattern: [P, Q] followed by one or more digits (\d+)
    match = re.search(r'[PQ](\d+)', column_name)
    bus_number = int(match.group(1)) if match else float('inf')

    # The resulting tuple (0, 1), (0, 2), ..., (1, 1), (1, 2), ... guarantees the correct order.
    return (type_key, bus_number)


# --- Applying the Sort to Your DataFrame ---

# 1. Get the current column order
current_columns = net_injection_df.columns.tolist()

# 2. Sort the column list using the custom key
# This will result in: [P1, P2, P3, ..., Q1, Q2, Q3, ...]
sorted_columns = sorted(current_columns, key=sort_by_power_type_and_number)

# 3. Apply the new column order to your DataFrame
df = net_injection_df.reindex(columns=sorted_columns)

print("Columns sorted successfully (P first, then Q, both numerically).")

df.head()

import pandapower as pp
import pandapower.networks as pn
import pandas as pd
import copy
# net_pn = pn.case118()
net_pn = pn.case14()

def calculate_tie_line_flow(row):
    # Use a copy of the network
    net = copy.deepcopy(net_pn)
    net.load['p_mw'] = 0
    net.load['q_mvar'] = 0
    net.gen['p_mw'] = 0
    net.gen['vm_pu'] = 1.0

    for bus in net.bus.index:
        P = row.get(f'P{bus+1}', 0)   # safe get
        Q = row.get(f'Q{bus+1}', 0)
        pp.create_load(net, bus=bus, p_mw=max(P,0), q_mvar=max(Q,0))
        pp.create_sgen(net, bus=bus, p_mw=-min(P,0), q_mvar=-min(Q,0))

    pp.runpp(net, numba=False)  # disable numba

    line_flows = net.line[['from_bus', 'to_bus']].join(
        net.res_line[['p_from_mw', 'q_from_mvar', 'p_to_mw', 'q_to_mvar']]
    )
    line_flows = line_flows.reset_index()  # make sure line indices are 0..n-1

    tie_power = {}
    for line_idx, info in tie_lines_info.items():
        # print("line_idx is ")
        # print(line_idx)
        # print("info is ")
        # print(info)
        # print("info ends")
        if line_idx >= len(line_flows):
            continue  # skip if line index not present
        row_line = line_flows.loc[line_idx]
        bus = info['boundary_bus_idx']

        if info['flow_key'] == 'p_to_mw':
            p_flow = row_line['p_to_mw']
            q_flow = row_line['q_to_mvar']
        else:
            p_flow = row_line['p_from_mw']
            q_flow = row_line['q_from_mvar']

        if bus not in tie_power:
            tie_power[bus] = {'P_tie': 0.0, 'Q_tie': 0.0}
        tie_power[bus]['P_tie'] += p_flow
        tie_power[bus]['Q_tie'] += q_flow

    # build dataframe with Pbus / Qbus columns
    data = {}
    for bus, vals in tie_power.items():
        data[f'P{bus}'] = [vals['P_tie']]
        data[f'Q{bus}'] = [vals['Q_tie']]
    tie_df = pd.DataFrame(data)

    print("internal_df is")
    print(internal_lines_info)
    # --- Internal line flows ---
    internal_data = {}
    for line_idx, info in internal_lines_info.items():
        if line_idx >= len(line_flows):
            continue
        row_line = line_flows.loc[line_idx]
        # use 'from_bus' side
        internal_data[f"P_flow_{info['from_bus_idx']}_{info['to_bus_idx']}"] = [row_line[info['p_key']]]
        internal_data[f"Q_flow_{info['from_bus_idx']}_{info['to_bus_idx']}"] = [row_line[info['q_key']]]

    internal_df = pd.DataFrame(internal_data)

    # print("tie_df is")
    # print(tie_df)
    # print("internal_df is")
    # print(internal_df)
    return tie_df, internal_df


type(df)

df.shape[0]

count=0
all_tie_flows = []
all_internal_flows = []
for row in  range(0, df.shape[0]):
    # print(df.iloc[row])
    tie_df,internal_df = calculate_tie_line_flow(df.iloc[row])
    all_tie_flows.append(tie_df)
    all_internal_flows.append(internal_df)
    print(internal_df)
    # print(count)
    count+=1

print("all_tie_flows:", len(all_tie_flows))
print("all_internal_flows:", len(all_internal_flows))

tie_flows_df = pd.concat(all_tie_flows, ignore_index=True)
print(tie_flows_df)

internal_flows_df = pd.concat(all_internal_flows, ignore_index=True)
internal_flows_df.head

internal_flows_df.head()

internal_flows_df.columns

df.head()

df.to_csv("P_Q_15k.csv", index=False)
internal_flows_df.to_csv("Attack_Region_Internal_Line_Flow_15k.csv", index=False)
tie_flows_df.to_csv("Tie_Line_Flows_15k.csv", index=False)





selected_columns_attack_region = [
    col for col in df.columns
    if col.startswith(('P', 'Q'))
    and int(re.findall(r'\d+', col)[0]) in ATTACK_REGION_IDS
]


print(selected_columns_attack_region)

attack_region_pq= df[selected_columns_attack_region].copy()

attack_region_pq.head()

attack_region_pq.shape

tie_flows_df.head()

def shift_column_names(df, shift=1):
    new_cols = {}
    for col in df.columns:
        prefix = col[0]  # 'P' or 'Q'
        num = int(col[1:])
        new_cols[col] = f"{prefix}{num + shift}"
    return df.rename(columns=new_cols)

tie_flows_df = shift_column_names(tie_flows_df, shift=1)
tie_flows_df.head()

common_cols = tie_flows_df.columns.intersection(attack_region_pq.columns)
print(common_cols)

attack_region_pq[common_cols] = attack_region_pq[common_cols] - tie_flows_df[common_cols]

attack_region_pq.head()

internal_flows_df.head()

z= pd.concat([attack_region_pq, internal_flows_df], axis=1)
z.head()

z.to_csv("Final_Training_Data_15k.csv", index=False)

z.columns

