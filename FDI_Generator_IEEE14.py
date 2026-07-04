import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
import random
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

def independent_var(data, cols, changing_rate):
    """
    Multiply selected columns by random factors in [1, 1+changing_rate] row-wise.
    """
    for col in cols:
        if col not in data.columns:
            raise KeyError(f"Column not found in X: {col}")
        random_factors = np.random.uniform(1, 1 + changing_rate, size=len(data))
        data[col] *= random_factors
    return data

def dependent_var(real_X, real_y, syn_X, model="LightGBM", scale_y=True):
    X_train, X_test, y_train, y_test = train_test_split(
        real_X, real_y, test_size=0.2, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_test, y_test, test_size=0.5, random_state=42
    )

    x_scaler = StandardScaler()
    X_train_s = x_scaler.fit_transform(X_train)
    X_val_s = x_scaler.transform(X_val)
    X_test_s = x_scaler.transform(X_test)
    syn_X_s = x_scaler.transform(syn_X)

    if scale_y:
        y_scaler = StandardScaler()
        y_train_s = y_scaler.fit_transform(y_train)
        y_val_s = y_scaler.transform(y_val)
    else:
        y_scaler = None
        y_train_s = y_train
        y_val_s = y_val

    if model == "XGBoost":
        # xgb_regressor = XGBRegressor(
        #     n_estimators=1000,
        #     learning_rate=0.1,
        #     max_depth=6,
        #     subsample=0.8,
        #     colsample_bytree=0.8,
        #     early_stopping_rounds=50,
        #     objective="reg:squarederror",
        # )
        regressor = XGBRegressor(
            n_estimators=1500,
            learning_rate=0.05,
            max_depth=6,
            min_child_weight=3,
            gamma=0.0,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.0,
            reg_lambda=2.0,
            objective="reg:squarederror"
        )

    elif model == "LightGBM":
        regressor = MultiOutputRegressor(
            LGBMRegressor(
                n_estimators=600,
                learning_rate=0.1,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
            )
        )
    else:
        raise ValueError("Unsupported model type.")

    regressor.fit(X_train_s, y_train_s)

    y_pred_s = regressor.predict(X_test_s)
    syn_y_s = regressor.predict(syn_X_s)

    if scale_y:
        y_pred = y_scaler.inverse_transform(y_pred_s)
        syn_y = y_scaler.inverse_transform(syn_y_s)
    else:
        y_pred = y_pred_s
        syn_y = syn_y_s

    print(f"Real Data MSE for {model}:", mean_squared_error(y_test, y_pred))
    print(f"Real Data R2 for {model}:", r2_score(y_test, y_pred, multioutput="uniform_average"))

    return syn_y


def main():
    for model in ["XGBoost", "LightGBM"]:
    # for model in ["LightGBM"]:
        # ======== CHANGE THIS PATH TO YOUR IEEE-14 NORMAL CSV ========
        data_real = pd.read_csv(r"IEEE14_normal_50k.csv")

        # -------------------------
        # IEEE-14 column groups
        # -------------------------
        # Inputs (independent variables): generator outputs + loads
        gen_pg_cols = [f"PG{i}" for i in range(1, 15)]
        gen_qg_cols = [f"QG{i}" for i in range(1, 15)]
        pl_cols = [f"PL{i}" for i in range(1, 15)]
        ql_cols = [f"QL{i}" for i in range(1, 15)]
        qsh_cols = [f"Qsh{i}" for i in range(1, 15)]
        q_cols = [f"Q{i}" for i in range(1, 15)]
        p_cols = [f"P{i}" for i in range(1, 15)]

        # Targets (dependent variables): bus voltage magnitude + angle
        v_cols = [f"V{i}" for i in range(1, 15)]
        angle_cols = [f"theta{i}" for i in range(1, 15)]

        X_cols = gen_pg_cols + gen_qg_cols + pl_cols + ql_cols #+ qsh_cols
        y_cols = v_cols + angle_cols

        missing_X = [c for c in X_cols if c not in data_real.columns]
        missing_y = [c for c in y_cols if c not in data_real.columns]
        if missing_X or missing_y:
            raise KeyError(f"Missing columns. X missing: {missing_X}, y missing: {missing_y}")

        X = data_real[X_cols].copy()
        y = data_real[y_cols].copy()

        print("Generating independent data (IEEE-14)")
        # Option A (recommended): perturb a sampled subset of PL columns
        independent_values = independent_var(X.copy(), ["PL11", "PL4", "PL3"], changing_rate=0.001)

        # Option B: perturb ALL PL columns (uncomment if desired)
        # independent_values = independent_var(X.copy(), pl_cols, changing_rate=0.1)

        print("Generating dependent data")
        dependent_values = dependent_var(
            X,
            y,
            independent_values,
            model=model,
            scale_y=True
        )

        result_df = pd.concat([independent_values, pd.DataFrame(dependent_values, columns=y.columns)], axis=1)

        print("Saving results")
        result_df.to_csv(f'IEEE14_attack_{model}_scaled0001_pl11pl4pl3.csv', index=False)


if __name__ == "__main__":
    main()
