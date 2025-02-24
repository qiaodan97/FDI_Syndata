import pandas as pd

# data = pd.read_csv(r"E:\FDI\OneDrive_1_1-15-2025\118\Datasets\IEEE118NormalWithPd_Qd.csv")
# print(f"Number of rows: {data.shape[0]}")
# print(f"Number of columns: {data.shape[1]}")
# print(f"Dataset size: {data.shape}")

# data_VGM = pd.read_csv(r"E:\FDI\OneDrive_1_1-15-2025\118\Datasets\VGM_class.csv")
# print(f"data_VGM size: {data_VGM.shape}")
# data_QL = pd.read_csv(r"E:\FDI\OneDrive_1_1-15-2025\118\Datasets\QL_class.csv")
# print(f"data_QL size: {data_QL.shape}")
selected_columns = [f'PL{i}' for i in range(1, 65)]
# real_data = pd.read_csv(r"E:\FDI\OneDrive_1_1-15-2025\118\Datasets\IEEE118NormalWithPd_Qd.csv").drop(columns=columns_to_drop,
#                                                                                                 errors='ignore')
#
# syn_data = pd.readd.read_csv(r"E:\FDI\OneDrive_1_1-15-2025\118\Datasets\IEEE118NormalWithPd_Qd.csv").drop(columns=columns_to_drop,
#                                                                                                 errors='ignore')
#
# syn_data = pd.read_csv(r"E:\FDI\OneDrive_1_1-15-2025\118\Datasets\PL_class.csv")

real_data = pd.read_csv(r"E:\FDI\OneDrive_1_2-16-2025\IEEE118Normal_Corrected.csv")

columns_to_drop = [col for col in real_data.columns if 'QG' in col]
columns_to_drop.append("Label")
real_data = real_data.drop(columns=columns_to_drop, errors='ignore').dropna()

syn_data = pd.read_csv(r'PL_Class_FDI_AllScaled.csv').drop(columns=columns_to_drop, errors='ignore').dropna()

print(f"data_PL: {syn_data.head}")
# data_Combine = pd.read_csv(r"E:\FDI\OneDrive_1_1-15-2025\118\Datasets\Combine_class.csv")
# print(f"data_Combine size: {data_Combine.shape}")
print(f"data_real: {real_data.head}")
#
# columns_VGM = set(data_VGM.columns)
# columns_QL = set(data_QL.columns)
# columns_PL = set(data_PL.columns)
# columns_Combine = set(data_Combine.columns)
# columns_real = set(data_real.columns)
#
# not_in_VGM = columns_real - columns_VGM
# not_in_QL = columns_real - columns_QL
# not_in_PL = columns_real - columns_PL
# not_in_Combine = columns_real - columns_Combine
#
# # Print the results
# print("Columns not in VGM:", not_in_VGM)
# print("Columns not in QL:", not_in_QL)
# print("Columns not in PL:", not_in_PL)
# print("Columns not in Combine:", not_in_Combine)