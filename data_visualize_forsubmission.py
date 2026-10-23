from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


# -----------------------------------------------------------------------------
# Input and output settings
# -----------------------------------------------------------------------------
real_data_name = r"IEEE118_normal_50k.csv"

# Adjust a filename here if the corresponding CSV uses a different name.
# model_files = {
#     "SAGAN": r"ieee14_sagan.csv",
#     "XGBoost": r"IEEE14_attack_XGBoost_scaled0.005(random+).csv",
#     "LightGBM": r"IEEE14_attack_LightGBM_scaled0.005(random+).csv",
# }
model_files = {
    "SAGAN": r"ieee118_sagan.csv",
    "XGBoost": r"IEEE118_attack_XGBoost_scaled0.1(random+).csv",
    "LightGBM": r"IEEE118_attack_LightGBM_scaled0.1(random+).csv",
}
figure_title = "IEEE 118-Bus System: Normal vs. Model-Generated FDI Samples"
output_name = "IEEE118_model_comparison_pca_tsne.png"
samples_per_class = 5000
random_seed = 42

# Metadata must not contribute to PCA or t-SNE.
excluded_columns = {"label", "base_id"}


def sample_rows(data, sample_size, seed):
    """Return a reproducible sample, or all rows when the file is smaller."""
    return data.sample(n=min(sample_size, len(data)), random_state=seed)


def prepare_pair(real_data, attack_data):
    """Align, standardize, and label one real-versus-attack data pair."""
    common_columns = [
        column
        for column in real_data.columns
        if column in attack_data.columns
        and column.lower() not in excluded_columns
        and pd.api.types.is_numeric_dtype(real_data[column])
        and pd.api.types.is_numeric_dtype(attack_data[column])
    ]

    if not common_columns:
        raise ValueError("The normal and attack datasets have no common numeric features.")

    real_values = real_data[common_columns]
    attack_values = attack_data[common_columns]
    combined = pd.concat([real_values, attack_values], ignore_index=True)
    scaled = StandardScaler().fit_transform(combined)
    labels = np.array(
        ["Normal data"] * len(real_values) + ["Attack data"] * len(attack_values)
    )
    return scaled, labels, len(common_columns)


def calculate_embeddings(features):
    """Calculate PCA and t-SNE projections for one model comparison."""
    pca = PCA(n_components=2)
    pca_embedding = pca.fit_transform(features)

    tsne = TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        random_state=random_seed,
    )
    tsne_embedding = tsne.fit_transform(features)

    return pca_embedding, tsne_embedding, pca.explained_variance_ratio_


def draw_embedding(ax, embedding, labels, colors):
    """Draw one consistently styled projection without a local legend."""
    # Draw attack points first so both classes remain visible in dense regions.
    for class_name in ("Attack data", "Normal data"):
        mask = labels == class_name
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=7,
            c=colors[class_name],
            alpha=0.48,
            linewidths=0,
            rasterized=True,
        )

    ax.grid(True, color="#D8D8D8", linewidth=0.55, alpha=0.65)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8)
    sns.despine(ax=ax)


# -----------------------------------------------------------------------------
# Load data and calculate one pair of embeddings per generator
# -----------------------------------------------------------------------------
real_data = pd.read_csv(real_data_name).dropna()
real_data = sample_rows(real_data, samples_per_class, random_seed)

results = {}
for model_index, (model_name, attack_file) in enumerate(model_files.items()):
    print(f"Processing {model_name}: {attack_file}")
    attack_data = pd.read_csv(attack_file).dropna()
    attack_data = sample_rows(
        attack_data, samples_per_class, random_seed + model_index + 1
    )

    features, labels, feature_count = prepare_pair(real_data, attack_data)
    pca_data, tsne_data, variance_ratio = calculate_embeddings(features)
    results[model_name] = {
        "labels": labels,
        "pca": pca_data,
        "tsne": tsne_data,
        "variance_ratio": variance_ratio,
    }
    print(
        f"  normal={len(real_data)}, attack={len(attack_data)}, "
        f"common features={feature_count}"
    )


# -----------------------------------------------------------------------------
# Publication-style 2 x 3 comparison figure
# -----------------------------------------------------------------------------
sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)

colors = {
    "Normal data": "#2878B5",
    "Attack data": "#D9534F",
}

fig, axes = plt.subplots(2, 3, figsize=(14.8, 8.4))
row_order_letters = (("a", "b", "c"), ("d", "e", "f"))

for column, (model_name, model_result) in enumerate(results.items()):
    tsne_ax = axes[0, column]
    pca_ax = axes[1, column]

    draw_embedding(
        tsne_ax, model_result["tsne"], model_result["labels"], colors
    )
    tsne_ax.set_title(
        f"({row_order_letters[0][column]}) {model_name}",
        fontsize=12,
        fontweight="semibold",
        pad=8,
    )
    tsne_ax.set_xlabel("t-SNE dimension 1", fontsize=9)
    tsne_ax.set_ylabel("t-SNE dimension 2", fontsize=9)

    draw_embedding(pca_ax, model_result["pca"], model_result["labels"], colors)
    pca_ax.set_title(
        f"({row_order_letters[1][column]}) {model_name}",
        fontsize=12,
        fontweight="semibold",
        pad=8,
    )
    variance = model_result["variance_ratio"]
    pca_ax.set_xlabel(f"Principal component 1 ({variance[0]:.1%})", fontsize=9)
    pca_ax.set_ylabel(f"Principal component 2 ({variance[1]:.1%})", fontsize=9)

legend_handles = [
    Line2D(
        [0],
        [0],
        marker="o",
        linestyle="none",
        label=class_name,
        markerfacecolor=colors[class_name],
        markeredgecolor="none",
        markersize=7,
    )
    for class_name in ("Normal data", "Attack data")
]

fig.suptitle(figure_title, fontsize=16, fontweight="bold", y=0.985)
fig.legend(
    handles=legend_handles,
    loc="upper center",
    ncol=2,
    frameon=False,
    bbox_to_anchor=(0.5, 0.945),
)

# Row labels make the six-panel structure immediately clear.
fig.text(
    0.018,
    0.665,
    "t-SNE projection",
    rotation=90,
    ha="center",
    va="center",
    fontsize=12,
    fontweight="semibold",
)
fig.text(
    0.018,
    0.265,
    "PCA projection",
    rotation=90,
    ha="center",
    va="center",
    fontsize=12,
    fontweight="semibold",
)

fig.subplots_adjust(
    left=0.075,
    right=0.985,
    bottom=0.075,
    top=0.875,
    wspace=0.25,
    hspace=0.34,
)
fig.savefig(output_name, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)

print(f"Saved figure: {Path(output_name).resolve()}")
