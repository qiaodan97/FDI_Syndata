import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# =========================
# Load datasets
# =========================

# real_data = pd.read_csv(r"IEEE118_normal_50k.csv")
# sagan_data = pd.read_csv(r"ieee118_sagan.csv")
# xgb_data = pd.read_csv(r"IEEE118_attack_XGBoost.csv")
# lgbm_data = pd.read_csv(r"IEEE118_attack_LightGBM.csv")

real_data = pd.read_csv(r"IEEE14_normal_50k.csv")
sagan_data = pd.read_csv(r"ieee14_sagan.csv")
xgb_data = pd.read_csv(r"IEEE14_attack_XGBoost.csv")
lgbm_data = pd.read_csv(r"IEEE14_attack_LightGBM.csv")

# Keep only common columns
common_cols = sorted(list(set(real_data.columns) & set(sagan_data.columns) & set(xgb_data.columns) & set(lgbm_data.columns)))

real_data = real_data[common_cols]
sagan_data = sagan_data[common_cols]
xgb_data = xgb_data[common_cols]
lgbm_data = lgbm_data[common_cols]
# =========================
# Preprocessing function
# =========================

def preprocess(real_df, syn_df, sample_size=5000):

    # Remove label columns
    columns_to_drop = [col for col in real_df.columns if "Label" in col]

    real_df = real_df.drop(columns=columns_to_drop, errors="ignore")
    syn_df = syn_df.drop(columns=columns_to_drop, errors="ignore")



    # Convert to numeric
    real_df = real_df.apply(pd.to_numeric, errors="coerce")
    syn_df = syn_df.apply(pd.to_numeric, errors="coerce")

    # Drop NaN
    real_df = real_df.dropna()
    syn_df = syn_df.dropna()

    # Sample
    real_df = real_df.sample(sample_size, random_state=42)
    syn_df = syn_df.sample(sample_size, random_state=42)

    # Combine
    X = pd.concat([real_df, syn_df], axis=0).reset_index(drop=True)

    y = np.array(
        ["Normal"] * len(real_df) +
        ["Attack"] * len(syn_df)
    )
    print(title)
    print("Real:", len(real_df))
    print("Syn :", len(syn_df))
    print("y unique:", np.unique(y))
    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

# =========================
# Create subplot figure
# =========================

fig, axes = plt.subplots(1, 3, figsize=(24, 8))

datasets = [
    ("SAGAN Model", sagan_data),
    ("XGBoost Model", xgb_data),
    ("LightGBM Model", lgbm_data),
]

# =========================
# Generate each subplot
# =========================

for ax, (title, syn_data) in zip(axes, datasets):

    print(f"Processing {title}")

    X_scaled, y = preprocess(real_data, syn_data)

    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=30,
        learning_rate='auto',
        init='pca'
    )

    X_tsne = tsne.fit_transform(X_scaled)
    # 先把结果放进 DataFrame
    plot_df = pd.DataFrame({
        "Dim1": X_tsne[:, 0],
        "Dim2": X_tsne[:, 1],
        "Label": y
    })
    sns.scatterplot(
        data=plot_df,
        x="Dim1",
        y="Dim2",
        hue="Label",
        style="Label",
        palette={"Normal": "blue", "Attack": "red"},
        markers={"Normal": "o", "Attack": "X"},
        s=18,
        alpha=0.6,
        ax=ax
    )
    ax.set_title(f"Real Data vs {title} FDI Data", fontsize=14)
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")

# =========================
# Improve layout
# =========================

handles, labels = axes[0].get_legend_handles_labels()

for ax in axes:
    ax.legend_.remove()

fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=2,
    fontsize=12
)

plt.tight_layout(rect=[0, 0, 1, 0.95])

# =========================
# Save figure
# =========================

plt.savefig(
    "tsne_comparison_all_models_ieee14.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()