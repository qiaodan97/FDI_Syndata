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


def dependent_var(real_X, real_y, syn_X, model="XGBoost", scale=False):
    """
    Train on real (X->y), predict y for syn_X, and report test MSE on held-out real split.
    """
    if scale:
        x_scaler = MinMaxScaler()
        y_scaler = MinMaxScaler()
        real_X_scaled = x_scaler.fit_transform(real_X)
        syn_X_scaled = x_scaler.transform(syn_X)
        real_y_scaled = y_scaler.fit_transform(real_y)

        X_train, X_test, y_train, y_test = train_test_split(
            real_X_scaled, real_y_scaled, test_size=0.2, random_state=42
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            real_X, real_y, test_size=0.2, random_state=42
        )
        syn_X_scaled = syn_X  # keep naming consistent

    X_val, X_test, y_val, y_test = train_test_split(
        X_test, y_test, test_size=0.5, random_state=42
    )

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
        xgb_regressor = XGBRegressor(
            n_estimators=1500,
            learning_rate=0.05,
            max_depth=6,
            min_child_weight=3,
            gamma=0.0,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.0,
            reg_lambda=2.0,
            objective="reg:squarederror",
            early_stopping_rounds=50,
        )
        xgb_regressor.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)
        y_pred = xgb_regressor.predict(X_test)
        syn_y = xgb_regressor.predict(syn_X_scaled)

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
        regressor.fit(X_train, y_train)
        y_pred = regressor.predict(X_test)
        syn_y = regressor.predict(syn_X_scaled)

    elif model == "MultiOutputRegressor":
        regr = MultiOutputRegressor(Ridge(random_state=100))
        regr.fit(X_train, y_train)
        y_pred = regr.predict(X_test)
        syn_y = regr.predict(syn_X_scaled)

    elif model == "RandomForest":
        rf = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=100))
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        syn_y = rf.predict(syn_X_scaled)

    else:
        raise ValueError("Unsupported model type.")

    mse = mean_squared_error(y_test, y_pred)
    print(f"Real Data Mean Squared Error for {model}:", mse)

    # If y was scaled, you may want to invert back to original units for saving.
    if scale:
        syn_y = y_scaler.inverse_transform(syn_y)

    return syn_y


def main():
    for percentage in [0.75]:
        # ======== CHANGE THIS PATH TO YOUR IEEE-14 NORMAL CSV ========
        data_real = pd.read_csv(r"pf_dataset_FINAL_correct_PQ_IEEE14small.csv")

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
        independent_values = independent_var(X.copy(), ["PL3"], changing_rate=0.3)

        # Option B: perturb ALL PL columns (uncomment if desired)
        # independent_values = independent_var(X.copy(), pl_cols, changing_rate=0.1)

        print("Generating dependent data (IEEE-14)")
        dependent_values = dependent_var(X, y, independent_values, model="XGBoost", scale=False)

        result_df = pd.concat(
            [independent_values, pd.DataFrame(dependent_values, columns=y.columns)],
            axis=1
        )

        print("Saving results")
        result_df.to_csv(
            f"IEEE14_PL_Class_FDI_NotScaled_range+1.3_nestimator600_XGBoost_model2.csv",
            index=False
        )


if __name__ == "__main__":
    main()
