import pandapower as pp
import pandapower.networks as pn
import numpy as np
import pandas as pd
import copy

# -----------------------------
# Settings
# -----------------------------
N_SAMPLES = 5000
LOAD_VARIATION = 0.02
GEN_VARIATION = 0.01

dataset = []
total_P_mismatch = []
total_Q_mismatch = []

# -----------------------------
# Base Network
# -----------------------------
# net = pn.case14()
net = pn.case118()

net.bus["min_vm_pu"] = 0.90
net.bus["max_vm_pu"] = 1.10

net.gen["vm_pu"] = 1.0
net.ext_grid["vm_pu"] = 1.0

# -----------------------------
# Sampling Loop
# -----------------------------
for sample_id in range(N_SAMPLES):

    net_temp = copy.deepcopy(net)

    # -----------------------------
    # Load perturbation
    # -----------------------------
    load_noise = 1 + LOAD_VARIATION * (2*np.random.rand(len(net_temp.load)) - 1)
    net_temp.load["p_mw"] *= load_noise
    net_temp.load["q_mvar"] *= load_noise

    # -----------------------------
    # Generator perturbation
    # -----------------------------
    gen_noise = 1 + GEN_VARIATION * (2*np.random.rand(len(net_temp.gen)) - 1)
    net_temp.gen["p_mw"] *= gen_noise

    # -----------------------------
    # Run Power Flow
    # -----------------------------
    try:
        pp.runpp(net_temp, calculate_voltage_angles=True)
    except:
        continue

    res_bus = net_temp.res_bus
    res_gen = net_temp.res_gen
    res_ext = net_temp.res_ext_grid

    gen_buses = net_temp.gen["bus"]
    ext_buses = net_temp.ext_grid["bus"]

    row = {}

    total_P_err = 0.0
    total_Q_err = 0.0

    # -----------------------------
    # Loop over buses
    # -----------------------------
    for i in net_temp.bus.index:

        # -----------------------------
        # Active Power (from elements)
        # -----------------------------
        gen_mask = net_temp.gen["bus"] == i
        ext_mask = net_temp.ext_grid["bus"] == i

        P_gen = (
                res_gen.loc[gen_mask, "p_mw"].sum() +
                res_ext.loc[ext_mask, "p_mw"].sum()
        )

        Q_gen = (
                res_gen.loc[gen_mask, "q_mvar"].sum() +
                res_ext.loc[ext_mask, "q_mvar"].sum()
        )

        mask = net_temp.load["bus"] == i
        P_load = net_temp.load.loc[mask, "p_mw"].sum()
        Q_load = net_temp.load.loc[mask, "q_mvar"].sum()

        P_bus = res_bus.loc[i, "p_mw"]
        P_net = -res_bus.loc[i, "p_mw"]
        # P_net = P_gen - P_load
        P_mismatch = P_net + P_bus
        total_P_err += P_mismatch

        # -----------------------------
        # Reactive Power (from bus physics)
        # -----------------------------
        Q_bus = res_bus.loc[i, "q_mvar"]

        # Correct net injection
        # Q_net = Q_gen - Q_load
        Q_net = -res_bus.loc[i, "q_mvar"]
        Q_mismatch = Q_net + Q_bus
        total_Q_err += Q_mismatch

        # -----------------------------
        # Store features
        # -----------------------------
        idx = i + 1

        row[f"P{idx}"] = P_net
        row[f"Q{idx}"] = Q_net
        row[f"V{idx}"] = res_bus.loc[i, "vm_pu"]
        row[f"theta{idx}"] = res_bus.loc[i, "va_degree"]
        row[f"PG{idx}"] = P_gen
        row[f"PL{idx}"] = P_load
        row[f"QG{idx}"] = Q_gen
        row[f"QL{idx}"] = Q_load

        # 找到该 bus 的 shunt
        # q_shunt = 0
        # mask = net_temp.shunt["bus"] == i
        # if mask.any():
        #     q_shunt = -net_temp.res_shunt.loc[mask, "q_mvar"].sum()
        #
        # row[f"Qsh{idx}"] = q_shunt

    # -----------------------------
    # Store totals
    # -----------------------------
    # row["Total_P_balance_error"] = total_P_err
    # row["Total_Q_balance_error"] = total_Q_err

    total_P_mismatch.append(total_P_err)
    total_Q_mismatch.append(total_Q_err)

    dataset.append(row)

    if (sample_id + 1) % 500 == 0:
        print(f"{sample_id+1} samples generated...")

# -----------------------------
# Final Results
# -----------------------------
df = pd.DataFrame(dataset)

print("\n======================================")
print(f"Avg P mismatch: {np.mean(total_P_mismatch):.10e}")
print(f"Max P mismatch: {np.max(np.abs(total_P_mismatch)):.10e}")
print("--------------------------------------")
print(f"Avg Q mismatch: {np.mean(total_Q_mismatch):.10e}")
print(f"Max Q mismatch: {np.max(np.abs(total_Q_mismatch)):.10e}")
print("======================================")


df.to_csv(f"pf_dataset_FINAL_correct_PQ_IEEE118small.csv", index=False)
print(f"Dataset saved as: pf_dataset_FINAL_correct_PQ.csv")