import pandas as pd

# 假设你的 DataFrame 叫 df
# real_data_path = r"E:\FDI\OneDrive_1_2-16-2025\IEEE118Normal_Corrected.csv"
real_data_path = r"pf_dataset_FINAL_correct_PQ_IEEE118small.csv"
# real_data_path = r"E:\FDI\_FromMarzia\_FromMarzia\FDIG\FDIG\ieee14_nonzero_pg_dataset.csv"
df = pd.read_csv(real_data_path)
# columns = [col for col in df.columns if 'PL' in col]
# columns = [col for col in df.columns if 'VLM' in col]
# columns = [col for col in df.columns if 'VLA' in col]
# columns = [col for col in df.columns if 'VGA' in col]

# columns = [c for c in df.columns if c.startswith("GenBus") and c.endswith("_PG")]
# columns = [c for c in df.columns if c.startswith("GenBus") and c.endswith("_QG")]
# columns = [f"PL{i}" for i in range(1, 15)]
columns = [f"PL{i}" for i in range(1, 119)]
# columns = [c for c in df.columns if c.startswith("Bus") and c.endswith("_QL")]
# columns = [c for c in df.columns if c.startswith("Bus") and c.endswith("_V")]
# columns = [c for c in df.columns if c.startswith("Bus") and c.endswith("_angle")]

# 计算统计信息
stats_df = df[columns].agg(['min', 'max', 'mean', 'std']).T  # 转置方便按列排列
stats_df.columns = ['Min', 'Max', 'Mean', 'Std']  # 重命名列

# 保存为 CSV 文件
stats_df.to_csv('pl_column_stats_118.csv')
