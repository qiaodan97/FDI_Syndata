import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
seed = 42

np.random.seed(seed)

torch.manual_seed(seed)

torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# =========================
# 1. Load data
# =========================
#
# syn_data_name = r"IEEE118_attack_LightGBM.csv"
# syn_data_name = r"IEEE118_attack_XGBoost.csv"
# syn_data_name = r"ieee118_sagan.csv"
# real_data_name = r"IEEE118_normal_50k.csv"

# syn_data_name = r"IEEE14_attack_xgboost_scaled0002.csv".005(random+-6040)
syn_data_names = [

                  # r"IEEE14_attack_xgboost_scaled0.005(fixed-).csv",
                  r"IEEE14_attack_lightgbm_scaled0.csv",
                  ]

# syn_data_names = [r"IEEE14_attack_xgboost_scaled0002.csv"]
# syn_data_name = r"IEEE14_attack_XGBoost.csv"
# syn_data_name = r"ieee14_sagan.csv"
for syn_data_name in syn_data_names:
    print(syn_data_name)
    real_data_name = r"IEEE14_normal_50k.csv"
    normal_df = pd.read_csv(real_data_name)
    attack_df = pd.read_csv(syn_data_name)

    normal_df["Label"] = 0
    attack_df["Label"] = 1

    normal_df["base_id"] = range(len(normal_df))
    attack_df["base_id"] = range(len(attack_df))

    merged_df = pd.concat([normal_df, attack_df], ignore_index=True)
    print(normal_df.columns.tolist())
    print(attack_df.columns.tolist())
    print(normal_df.columns.equals(attack_df.columns))

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

    # =========================
    # Teacher model on test set:
    # =========================
    print("Teacher model on test set:")
    # print("Best Validation Normal Loss:", best_val_loss)
    print("Test Accuracy:", accuracy_score(y_test, test_pred))
    # print("Test Confusion Matrix:")
    # print(confusion_matrix(y_test, test_pred))
    # print("Test Classification Report:")
    # print(classification_report(y_test, test_pred))

    print(f1_score(y_test, test_pred, average="macro"))


    # # =========================
    # # 10. student model
    # # =========================
    #
    # with torch.no_grad():
    #     student_X_train_device = student_X_train_tensor.to(device)
    #     student_X_train_hat = model(student_X_train_device)
    #
    #     student_train_errors = torch.mean(
    #         (student_X_train_hat - student_X_train_device) ** 2,
    #         dim=1
    #     ).cpu().numpy()
    #
    # student_y_train_from_teacher = (student_train_errors > threshold).astype(int)
    #
    # # =========================
    # # Teacher model on student train set:
    # # =========================
    # # print("Teacher model on student train set:")
    # # print("\n" + "=" * 60)
    # # print("Unsupervised AutoEncoder Anomaly Detector")
    # # print("Best Validation Normal Loss:", best_val_loss)
    # # print("Test Accuracy:", accuracy_score(student_y_train, student_y_train_from_teacher))
    # # print("Test Confusion Matrix:")
    # # print(confusion_matrix(student_y_train, student_y_train_from_teacher))
    # # print("Test Classification Report:")
    # # print(classification_report(student_y_train, student_y_train_from_teacher))
    #
    # from sklearn.pipeline import Pipeline
    # from sklearn.preprocessing import StandardScaler
    # from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    #
    # from sklearn.linear_model import LogisticRegression
    # from sklearn.svm import LinearSVC, SVC
    # from sklearn.neighbors import KNeighborsClassifier
    # from sklearn.neural_network import MLPClassifier
    # from sklearn.ensemble import RandomForestClassifier
    #
    # # models = {
    # #     # "Logistic Regression": Pipeline([
    # #     #     ("scaler", StandardScaler()),
    # #     #     ("clf", LogisticRegression(max_iter=5000, class_weight="balanced"))
    # #     # ]),
    # #
    # #     "Random Forest": RandomForestClassifier(
    # #         n_estimators=300,
    # #         random_state=42,
    # #         class_weight="balanced",
    # #         n_jobs=-1
    # #     )
    # # }
    # # model = RandomForestClassifier(
    # #         n_estimators=300,
    # #         random_state=42,
    # #         class_weight="balanced",
    # #         n_jobs=-1
    # #     )
    # model = RandomForestClassifier(
    #     n_estimators=300,
    #     random_state=42,
    #     class_weight="balanced",
    #     n_jobs=-1
    # )
    # # model = Pipeline([
    # #         ("scaler", StandardScaler()),
    # #         ("clf", LogisticRegression(max_iter=5000, class_weight="balanced"))
    # #     ])
    # model.fit(student_X_train, student_y_train_from_teacher)
    #
    # test_pred = model.predict(X_test)
    #
    # print("Student model on test set:")
    # print("Test Accuracy:", accuracy_score(y_test, test_pred))
    # print("Test Confusion Matrix:")
    # print(confusion_matrix(y_test, test_pred))
    # print("Test Classification Report:")
    # print(classification_report(y_test, test_pred))

    # =========================
    # 10. Student model
    # Improvements:
    # 1) add AE reconstruction error as a feature
    # 2) filter uncertain pseudo-labels near the threshold
    # 3) compare RF and HistGradientBoosting
    # =========================

    from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    with torch.no_grad():
        student_X_train_device = student_X_train_tensor.to(device)
        student_X_train_hat = model(student_X_train_device)

        student_train_errors = torch.mean(
            (student_X_train_hat - student_X_train_device) ** 2,
            dim=1
        ).cpu().numpy()

    student_y_train_from_teacher = (student_train_errors > threshold).astype(int)

    # Add AE reconstruction error as an extra feature
    student_features = student_X_train.copy()
    student_features["ae_error"] = student_train_errors

    test_features = X_test.copy()
    test_features["ae_error"] = test_errors

    # Filter uncertain pseudo-labels close to the AE threshold
    margin = np.abs(student_train_errors - threshold)

    keep_ratio = 0.70
    keep_threshold = np.percentile(margin, 100 * (1 - keep_ratio))
    keep_mask = margin >= keep_threshold

    print("\nPseudo-label filtering:")
    print("Original student samples:", len(student_features))
    print("Kept student samples:", keep_mask.sum())
    print("Teacher pseudo-label distribution before filtering:")
    print(pd.Series(student_y_train_from_teacher).value_counts())
    print("Teacher pseudo-label distribution after filtering:")
    print(pd.Series(student_y_train_from_teacher[keep_mask]).value_counts())

    student_features_filtered = student_features.loc[keep_mask]
    student_y_filtered = student_y_train_from_teacher[keep_mask]

    student_models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=2,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        ),

        # "HistGradientBoosting": HistGradientBoostingClassifier(
        #     learning_rate=0.05,
        #     max_iter=300,
        #     max_leaf_nodes=31,
        #     l2_regularization=0.01,
        #     random_state=42
        # )
    }

    for student_name, student_model in student_models.items():
        print("\n" + "=" * 60)
        print(f"Student model: {student_name}")

        # student_model.fit(student_features_filtered, student_y_filtered)
        student_model.fit(student_features_filtered, student_y_train[keep_mask])

        student_pred = student_model.predict(test_features)

        print("Student model on test set:")
        print("Test Accuracy:", accuracy_score(y_test, student_pred))
        print("Test Confusion Matrix:")
        print(confusion_matrix(y_test, student_pred))
        print("Test Classification Report:")
        print(classification_report(y_test, student_pred))

        print(f1_score(y_test, student_pred, average="macro"))
        # print(f1_score(y_test, student_pred, pos_label=1))