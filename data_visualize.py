import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import numpy as np
from sklearn.preprocessing import MinMaxScaler

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

# Independent variables
vgm_columns = [col for col in real_data.columns if 'VGM' in col]
pg_columns = [col for col in real_data.columns if 'PG' in col]
pl_columns = [col for col in real_data.columns if 'PL' in col]
ql_columns = [col for col in real_data.columns if 'QL' in col]
# Dependent variables
vlm_columns = [col for col in real_data.columns if 'VLM' in col]
vla_columns = [col for col in real_data.columns if 'VLA' in col]
vga_columns = [col for col in real_data.columns if 'VGA' in col]

X = real_data[vgm_columns + pg_columns + pl_columns + ql_columns]
y = real_data[vlm_columns + vla_columns + vga_columns]

scaler = MinMaxScaler()  # Is MinMax the best option?
y_scaled = scaler.fit_transform(y)

real_data = pd.concat([X, pd.DataFrame(y_scaled, columns=y.columns)], axis=1)


syn_data = pd.read_csv(r'PL_Class_FDI_NotScaled.csv').drop(columns=columns_to_drop, errors='ignore').dropna()

# real_data = real_data[selected_columns]
# syn_data = syn_data[selected_columns]

X = pd.concat([real_data, syn_data], axis=0).reset_index(drop=True)
y = np.array([0] * len(real_data) + [1] * len(syn_data))

print("Start PCA")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

print("Save PCA")
plt.figure(figsize=(14, 12))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette=["blue", "red"])
plt.title("PCA Visualization")
plt.savefig(r"result\pca_visualization.png", dpi=300)

print("Start TSNE")
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

print("Save TSNE")
plt.figure(figsize=(14, 12))
sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y, palette=["blue", "red"])
plt.title("t-SNE Visualization")
plt.savefig(r"result\tsne_visualization.png", dpi=300)
