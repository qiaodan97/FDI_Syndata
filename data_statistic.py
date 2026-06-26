import pandas as pd
import numpy as np

from scipy.stats import wasserstein_distance, ks_2samp, energy_distance
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score


# =========================
# Load datasets
# =========================
number = "IEEE14"
# real_data = pd.read_csv(r"IEEE118_normal_50k.csv")
#
# model_files = {
#     "SAGAN": r"ieee118_sagan.csv",
#     "XGBoost": r"IEEE118_attack_XGBoost.csv",
#     "LightGBM": r"IEEE118_attack_LightGBM.csv",
# }
#
# real_data = pd.read_csv(r"IEEE118_normal_50k.csv")
# sagan_data = pd.read_csv(r"ieee118_sagan.csv")
# xgb_data = pd.read_csv(r"IEEE118_attack_XGBoost.csv")
# lgbm_data = pd.read_csv(r"IEEE118_attack_LightGBM.csv")

real_data = pd.read_csv(r"IEEE14_normal_50k.csv")

model_files = {
    "SAGAN": r"ieee14_sagan.csv",
    "XGBoost": r"IEEE14_attack_XGBoost.csv",
    "LightGBM": r"IEEE14_attack_LightGBM.csv",
}

real_data = pd.read_csv(r"IEEE14_normal_50k.csv")
sagan_data = pd.read_csv(r"ieee14_sagan.csv")
xgb_data = pd.read_csv(r"IEEE14_attack_XGBoost.csv")
lgbm_data = pd.read_csv(r"IEEE14_attack_LightGBM.csv")

# Keep only common columns
common_cols = sorted(list(set(real_data.columns) & set(sagan_data.columns) & set(xgb_data.columns) & set(lgbm_data.columns)))


# =========================
# Preprocessing
# =========================

def align_and_clean(real_df, syn_df, sample_size=5000, random_state=42):
    columns_to_drop = [
        col for col in set(real_df.columns) | set(syn_df.columns)
        if "Label" in col
    ]

    real_df = real_df.drop(columns=columns_to_drop, errors="ignore")
    syn_df = syn_df.drop(columns=columns_to_drop, errors="ignore")

    real_df = real_df[common_cols]
    syn_df = syn_df[common_cols]

    real_df = real_df.apply(pd.to_numeric, errors="coerce")
    syn_df = syn_df.apply(pd.to_numeric, errors="coerce")

    real_df = real_df.dropna()
    syn_df = syn_df.dropna()

    n = min(sample_size, len(real_df), len(syn_df))

    real_df = real_df.sample(n=n, random_state=random_state)
    syn_df = syn_df.sample(n=n, random_state=random_state)

    return real_df.reset_index(drop=True), syn_df.reset_index(drop=True), common_cols


# =========================
# Feature-wise statistics
# =========================

def featurewise_stats(real_df, syn_df):
    rows = []

    for col in real_df.columns:
        r = real_df[col].values
        s = syn_df[col].values

        real_mean = np.mean(r)
        syn_mean = np.mean(s)

        real_std = np.std(r)
        syn_std = np.std(s)

        mean_abs_diff = abs(real_mean - syn_mean)
        std_abs_diff = abs(real_std - syn_std)

        # Avoid division by zero
        mean_rel_diff = mean_abs_diff / (abs(real_mean) + 1e-8)
        std_rel_diff = std_abs_diff / (abs(real_std) + 1e-8)

        wd = wasserstein_distance(r, s)
        ks_stat, ks_pvalue = ks_2samp(r, s)

        rows.append({
            "feature": col,
            "real_mean": real_mean,
            "syn_mean": syn_mean,
            "mean_abs_diff": mean_abs_diff,
            "mean_rel_diff": mean_rel_diff,
            "real_std": real_std,
            "syn_std": syn_std,
            "std_abs_diff": std_abs_diff,
            "std_rel_diff": std_rel_diff,
            "wasserstein_distance": wd,
            "ks_stat": ks_stat,
            "ks_pvalue": ks_pvalue,
        })

    return pd.DataFrame(rows)


# =========================
# Dataset-level statistics
# =========================

def dataset_level_stats(real_df, syn_df):
    scaler = StandardScaler()
    X_all = pd.concat([real_df, syn_df], axis=0)

    X_scaled = scaler.fit_transform(X_all)

    real_scaled = X_scaled[:len(real_df)]
    syn_scaled = X_scaled[len(real_df):]

    mean_vector_l2 = np.linalg.norm(
        real_scaled.mean(axis=0) - syn_scaled.mean(axis=0)
    )

    cov_real = np.cov(real_scaled, rowvar=False)
    cov_syn = np.cov(syn_scaled, rowvar=False)

    covariance_frobenius = np.linalg.norm(cov_real - cov_syn, ord="fro")

    avg_wasserstein = np.mean([
        wasserstein_distance(real_scaled[:, i], syn_scaled[:, i])
        for i in range(real_scaled.shape[1])
    ])

    avg_energy_distance = np.mean([
        energy_distance(real_scaled[:, i], syn_scaled[:, i])
        for i in range(real_scaled.shape[1])
    ])

    return {
        "mean_vector_l2": mean_vector_l2,
        "covariance_frobenius": covariance_frobenius,
        "avg_wasserstein": avg_wasserstein,
        "avg_energy_distance": avg_energy_distance,
    }


# =========================
# PCA distribution distance
# =========================

def pca_stats(real_df, syn_df, n_components=10):
    scaler = StandardScaler()
    X_all = pd.concat([real_df, syn_df], axis=0)

    X_scaled = scaler.fit_transform(X_all)

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    real_pca = X_pca[:len(real_df)]
    syn_pca = X_pca[len(real_df):]

    pc_wasserstein = []
    pc_ks = []

    for i in range(n_components):
        pc_wasserstein.append(
            wasserstein_distance(real_pca[:, i], syn_pca[:, i])
        )
        pc_ks.append(
            ks_2samp(real_pca[:, i], syn_pca[:, i]).statistic
        )

    return {
        "pca_explained_variance_sum": pca.explained_variance_ratio_.sum(),
        "avg_pc_wasserstein": np.mean(pc_wasserstein),
        "avg_pc_ks_stat": np.mean(pc_ks),
    }


# =========================
# Classifier two-sample test
# =========================

def classifier_test(real_df, syn_df, random_state=42):
    X = pd.concat([real_df, syn_df], axis=0).reset_index(drop=True)
    y = np.array([0] * len(real_df) + [1] * len(syn_df))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.3,
        random_state=random_state,
        stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=random_state,
        n_jobs=-1
    )

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    return {
        "clf_accuracy": accuracy_score(y_test, y_pred),
        "clf_auc": roc_auc_score(y_test, y_prob),
        "clf_f1": f1_score(y_test, y_pred),
    }


# =========================
# Run comparison
# =========================

summary_rows = []
feature_stats_all = {}

for model_name, file_path in model_files.items():
    print(f"Processing {model_name}")

    syn_data = pd.read_csv(file_path)

    real_df, syn_df, common_cols = align_and_clean(
        real_data,
        syn_data,
        sample_size=5000,
        random_state=42
    )

    fw_stats = featurewise_stats(real_df, syn_df)
    feature_stats_all[model_name] = fw_stats

    ds_stats = dataset_level_stats(real_df, syn_df)
    pca_result = pca_stats(real_df, syn_df, n_components=10)
    clf_result = classifier_test(real_df, syn_df)

    row = {
        "model": model_name,
        "n_samples_each": len(real_df),
        "n_features": len(common_cols),

        "avg_feature_mean_abs_diff": fw_stats["mean_abs_diff"].mean(),
        "avg_feature_mean_rel_diff": fw_stats["mean_rel_diff"].mean(),
        "avg_feature_std_abs_diff": fw_stats["std_abs_diff"].mean(),
        "avg_feature_std_rel_diff": fw_stats["std_rel_diff"].mean(),
        "avg_feature_wasserstein": fw_stats["wasserstein_distance"].mean(),
        "avg_feature_ks_stat": fw_stats["ks_stat"].mean(),

        **ds_stats,
        **pca_result,
        **clf_result,
    }

    summary_rows.append(row)


summary_df = pd.DataFrame(summary_rows)

print("\n===== Summary Comparison =====")
print(summary_df)


# =========================
# Save outputs
# =========================

summary_df.to_csv(f"statistical_comparison_summary_{number}.csv", index=False)

for model_name, df in feature_stats_all.items():
    df.to_csv(f"featurewise_stats_{model_name}_{number}.csv", index=False)


# =========================
# Rank models
# Lower is better except classifier metrics:
# clf_auc closer to 0.5 is better
# =========================

ranking_df = summary_df.copy()
ranking_df["clf_auc_distance_from_0.5"] = abs(ranking_df["clf_auc"] - 0.5)

rank_cols = [
    "avg_feature_wasserstein",
    "avg_feature_ks_stat",
    "mean_vector_l2",
    "covariance_frobenius",
    "avg_pc_wasserstein",
    "avg_pc_ks_stat",
    "clf_auc_distance_from_0.5",
]

for col in rank_cols:
    ranking_df[f"rank_{col}"] = ranking_df[col].rank(method="min")

ranking_df["average_rank"] = ranking_df[
    [f"rank_{col}" for col in rank_cols]
].mean(axis=1)

ranking_df = ranking_df.sort_values("average_rank")

print("\n===== Model Ranking =====")
print(ranking_df[["model", "average_rank"] + rank_cols])

ranking_df.to_csv(f"statistical_comparison_ranking_{number}.csv", index=False)