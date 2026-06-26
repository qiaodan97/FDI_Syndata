import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# =========================
# 1. Load data
# =========================
#
syn_data_name = r"IEEE118_attack_LightGBM.csv"
# syn_data_name = r"IEEE118_attack_XGBoost.csv"
# syn_data_name = r"ieee118_sagan.csv"
real_data_name = r"IEEE118_normal_50k.csv"

# syn_data_name = r"IEEE14_attack_LightGBM.csv"
# syn_data_name = r"IEEE14_attack_XGBoost.csv"
# syn_data_name = r"ieee14_sagan.csv"
# real_data_name = r"IEEE14_normal_50k.csv"
normal_df = pd.read_csv(real_data_name)
attack_df = pd.read_csv(syn_data_name)

normal_df["Label"] = 0
attack_df["Label"] = 1

normal_df["base_id"] = range(len(normal_df))
attack_df["base_id"] = range(len(attack_df))

merged_df = pd.concat([normal_df, attack_df], ignore_index=True)


# =========================
# 2. Paired split by base_id
# teacher_train_ids 0.2
# student_train_ids 0.6
# test_ids 0.1
# val_ids 0.1
# =========================

base_ids = normal_df["base_id"].values

teacher_train_ids, temp_ids = train_test_split(
    base_ids,
    test_size=0.80,
    random_state=42
)

student_train_ids, temp_ids = train_test_split(
    temp_ids,
    test_size=0.25,
    random_state=42
)

val_ids, test_ids = train_test_split(
    temp_ids,
    test_size=0.50,
    random_state=42
)

teacher_train_df = merged_df[merged_df["base_id"].isin(teacher_train_ids)]
student_train_df = merged_df[merged_df["base_id"].isin(student_train_ids)]
val_df = merged_df[merged_df["base_id"].isin(val_ids)]
test_df = merged_df[merged_df["base_id"].isin(test_ids)]


# =========================
# 3. Train AE only on normal samples
# =========================

train_normal_df = teacher_train_df[teacher_train_df["Label"] == 0]
val_normal_df = val_df[val_df["Label"] == 0]

student_X_train = student_train_df.drop(columns=["Label", "base_id"])

X_train = train_normal_df.drop(columns=["Label", "base_id"])
X_val_normal = val_normal_df.drop(columns=["Label", "base_id"])

X_test = test_df.drop(columns=["Label", "base_id"])

# X_train = train_normal_df.drop(columns=["Label", "base_id", "PL116", "PL90", "PL59"])
# X_val_normal = val_normal_df.drop(columns=["Label", "base_id", "PL116", "PL90", "PL59"])
#
# X_test = test_df.drop(columns=["Label", "base_id", "PL116", "PL90", "PL59"])

# X_train = train_normal_df.drop(columns=["Label", "base_id", "PL3"])
# X_val_normal = val_normal_df.drop(columns=["Label", "base_id", "PL3"])
#
# X_test = test_df.drop(columns=["Label", "base_id", "PL3"])
y_test = test_df["Label"].values
student_y_train = student_train_df["Label"].values


# Optional: only use V and theta
# top_features = [
#     col for col in X_train.columns
#     if col.startswith("V") or col.startswith("theta")
# ]
# X_train = X_train[top_features]
# X_val_normal = X_val_normal[top_features]
# X_test = X_test[top_features]


# =========================
# 4. Scale
# fit scaler only on normal training data
# =========================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
student_X_train_scaled = scaler.transform(student_X_train)
X_val_normal_scaled = scaler.transform(X_val_normal)
X_test_scaled = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
student_X_train_tensor = torch.tensor(student_X_train_scaled, dtype=torch.float32)
X_val_normal_tensor = torch.tensor(X_val_normal_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)


# =========================
# 5. DataLoader
# =========================

batch_size = 256

train_loader = DataLoader(
    TensorDataset(X_train_tensor),
    batch_size=batch_size,
    shuffle=True
)


# =========================
# 6. AutoEncoder
# =========================

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim=32):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_dim = X_train_tensor.shape[1]
model = AutoEncoder(input_dim=input_dim, latent_dim=32).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# =========================
# 7. Train
# no y is used here
# =========================

epochs = 100
best_val_loss = float("inf")
best_model_state = None

for epoch in range(epochs):
    model.train()
    train_loss = 0

    for (xb,) in train_loader:
        xb = xb.to(device)

        x_hat = model(xb)
        loss = criterion(x_hat, xb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    model.eval()
    with torch.no_grad():
        X_val_device = X_val_normal_tensor.to(device)
        X_val_hat = model(X_val_device)
        val_loss = criterion(X_val_hat, X_val_device).item()

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = model.state_dict()

    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch [{epoch+1}/{epochs}], "
            f"Train Loss: {train_loss:.4f}, "
            f"Val Normal Loss: {val_loss:.6f}"
        )


# =========================
# 8. Reconstruction error threshold
# use validation normal errors only
# =========================

model.load_state_dict(best_model_state)
model.eval()

with torch.no_grad():
    X_val_device = X_val_normal_tensor.to(device)
    X_val_hat = model(X_val_device)

    val_errors = torch.mean(
        (X_val_hat - X_val_device) ** 2,
        dim=1
    ).cpu().numpy()

threshold = np.percentile(val_errors, 95)

print("\nThreshold:", threshold)


# =========================
# 9. Test
# reconstruction error > threshold => anomaly / synthetic / attack
# =========================

with torch.no_grad():
    X_test_device = X_test_tensor.to(device)
    X_test_hat = model(X_test_device)

    test_errors = torch.mean(
        (X_test_hat - X_test_device) ** 2,
        dim=1
    ).cpu().numpy()

test_pred = (test_errors > threshold).astype(int)

print("Teacher model on test set:")
print("\n" + "=" * 60)
print("Unsupervised AutoEncoder Anomaly Detector")
print("Best Validation Normal Loss:", best_val_loss)
print("Test Accuracy:", accuracy_score(y_test, test_pred))
print("Test Confusion Matrix:")
print(confusion_matrix(y_test, test_pred))
print("Test Classification Report:")
print(classification_report(y_test, test_pred))


# =========================
# 10. student model
# =========================

with torch.no_grad():
    student_X_train_device = student_X_train_tensor.to(device)
    student_X_train_hat = model(student_X_train_device)

    student_train_errors = torch.mean(
        (student_X_train_hat - student_X_train_device) ** 2,
        dim=1
    ).cpu().numpy()

student_y_train_from_teacher = (student_train_errors > threshold).astype(int)

print("Teacher model on student train set:")
print("\n" + "=" * 60)
print("Unsupervised AutoEncoder Anomaly Detector")
print("Best Validation Normal Loss:", best_val_loss)
print("Test Accuracy:", accuracy_score(student_y_train, student_y_train_from_teacher))
print("Test Confusion Matrix:")
print(confusion_matrix(student_y_train, student_y_train_from_teacher))
print("Test Classification Report:")
print(classification_report(student_y_train, student_y_train_from_teacher))

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

# models = {
#     # "Logistic Regression": Pipeline([
#     #     ("scaler", StandardScaler()),
#     #     ("clf", LogisticRegression(max_iter=5000, class_weight="balanced"))
#     # ]),
#
#     "Random Forest": RandomForestClassifier(
#         n_estimators=300,
#         random_state=42,
#         class_weight="balanced",
#         n_jobs=-1
#     )
# }
# model = RandomForestClassifier(
#         n_estimators=300,
#         random_state=42,
#         class_weight="balanced",
#         n_jobs=-1
#     )
# model = RandomForestClassifier(
#     n_estimators=300,
#     max_depth=15,
#     min_samples_leaf=3,
#     max_features="sqrt",
#     random_state=42,
#     class_weight="balanced",
#     n_jobs=-1
# )
model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000, class_weight="balanced"))
    ])
model.fit(student_X_train, student_y_train_from_teacher)

test_pred = model.predict(X_test)

print("Student model on test set:")
print("Test Accuracy:", accuracy_score(y_test, test_pred))
print("Test Confusion Matrix:")
print(confusion_matrix(y_test, test_pred))
print("Test Classification Report:")
print(classification_report(y_test, test_pred))
# for name, model in models.items():
#     print("\n" + "="*60)
#     print(name)
#
#     model.fit(student_X_train, student_y_train_from_teacher)
#
#     test_pred = model.predict(X_test)
#
#     print("Student model on test set:")
#     print("Test Accuracy:", accuracy_score(y_test, test_pred))
#     print("Test Confusion Matrix:")
#     print(confusion_matrix(y_test, test_pred))
#     print("Test Classification Report:")
#     print(classification_report(y_test, test_pred))

shuffled_teacher_labels = np.random.permutation(student_y_train_from_teacher)

# rf = RandomForestClassifier(
#     n_estimators=300,
#     random_state=42,
#     class_weight="balanced",
#     n_jobs=-1
# )
#
# rf.fit(student_X_train, shuffled_teacher_labels)
# shuffled_pred = rf.predict(X_test)

lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000, class_weight="balanced"))
    ])
lr.fit(student_X_train, shuffled_teacher_labels)

shuffled_pred = lr.predict(X_test)


print("Model with shuffled teacher labels:")
print("Accuracy:", accuracy_score(y_test, shuffled_pred))
print(confusion_matrix(y_test, shuffled_pred))
print(classification_report(y_test, shuffled_pred))


# teacher_test_pred = (test_errors > threshold).astype(int)
# student_test_pred = model.predict(X_test)
#
# teacher_wrong = teacher_test_pred != y_test
# student_wrong = student_test_pred != y_test
#
# print("Teacher wrong:", teacher_wrong.sum())
# print("Student wrong:", student_wrong.sum())
#
# print("Teacher wrong but Student right:",
#       np.sum(teacher_wrong & ~student_wrong))
#
# print("Teacher right but Student wrong:",
#       np.sum(~teacher_wrong & student_wrong))
#
# print("Both wrong:",
#       np.sum(teacher_wrong & student_wrong))
#
#
# teacher_wrong_idx = teacher_test_pred != y_test
#
# wrong_df = X_test.copy()
# wrong_df["true_label"] = y_test
# wrong_df["teacher_pred"] = teacher_test_pred
# wrong_df["student_pred"] = student_test_pred
# wrong_df["teacher_wrong"] = teacher_wrong_idx
#
# diffs = []
#
# for col in X_test.columns:
#     mean_correct = wrong_df.loc[~wrong_df["teacher_wrong"], col].mean()
#     mean_wrong = wrong_df.loc[wrong_df["teacher_wrong"], col].mean()
#     diffs.append((col, abs(mean_correct - mean_wrong)))
#
# diffs = sorted(diffs, key=lambda x: x[1], reverse=True)
#
# print(diffs[:20])
#
# train_pred = model.predict(student_X_train)
#
# print("Student on pseudo-label train:")
# print(accuracy_score(student_y_train_from_teacher, train_pred))
# print(classification_report(student_y_train_from_teacher, train_pred))
#
# print("Student on true train label:")
# print(accuracy_score(student_y_train, train_pred))
# print(classification_report(student_y_train, train_pred))
#
# import matplotlib.pyplot as plt
#
# plt.hist(test_errors[y_test == 0], bins=50, alpha=0.5, label="Normal")
# plt.hist(test_errors[y_test == 1], bins=50, alpha=0.5, label="Attack/SAGAN")
# plt.axvline(threshold, linestyle="--", label="Threshold")
# plt.legend()
# plt.title("Reconstruction Error Distribution")
# plt.show()
