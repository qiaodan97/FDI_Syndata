import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

file_path = r"E:\FDI\OneDrive_1_1-15-2025\118\Datasets\IEEE118NormalWithPd_Qd.csv"  # Replace with the actual file path
data = pd.read_csv(file_path)
syn_file_path = r"E:\FDI\OneDrive_1_1-15-2025\118\Datasets\PL_class.csv"  # Replace with the actual file path
syn_data = pd.read_csv(syn_file_path)

# # Independ - from Krishna's paper
# vgm_columns = [ col for col in data . columns if "VGM" in col ]
# pg_columns = [ col for col in data . columns if "PG" in col ]
# pl_columns = [ col for col in data . columns if "PL" in col ]
# ql_columns = [ col for col in data . columns if "QL" in col ]
# # Depend
# vlm_columns = [ col for col in data . columns if "VLM" in col ]
# vla_columns = [ col for col in data . columns if "VLA" in col ]
# vga_columns = [ col for col in data . columns if "VGA" in col ]
#
# #
# independent_columns = vgm_columns + pg_columns + pl_columns + ql_columns
# dependent_columns = vlm_columns + vla_columns + vga_columns


data['Dataset'] = 'Original'
syn_data['Dataset'] = 'FDI'

combined_data = pd.concat([data, syn_data], ignore_index=True)
print(combined_data.shape)
# selected_columns = ['PL5', 'PL6', 'PL7', 'PL8', 'PL9', 'Dataset']
# sns.pairplot(combined_data[selected_columns], hue='Dataset', diag_kind='kde', palette='husl').savefig('pairplot_pl_5~9.png')
#
# selected_columns = ['PL5', 'PL6', 'PL7', 'PL8', 'PL9', 'Dataset']
# sns.pairplot(combined_data[selected_columns], hue='Dataset', diag_kind='kde', palette='husl').savefig('pairplot_pl_5~9.png')

pl_columns = [f'PL{i}' for i in range(1, 65)]
batch_size = 5
for i in range(0, len(pl_columns), batch_size):
    selected_columns = pl_columns[i:i + batch_size] + ['Dataset']
    pairplot = sns.pairplot(combined_data[selected_columns], hue='Dataset', diag_kind='kde', palette='husl')
    filename = f'pairplot_pl_{i+1}~{i+batch_size}.png'
    pairplot.savefig(filename)
    plt.close()

print("Pair plots saved successfully.")