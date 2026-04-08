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
    for col in cols:
        # data[col] *= np.random.uniform(1-changing_rate, 1+changing_rate) #higher value
        # data[col] *= np.random.uniform(1-changing_rate, 1+changing_rate) #higher value 0.3
        random_factors = np.random.uniform(1, 1 + changing_rate, size=len(data))
        data[col] *= random_factors
# randomlize each of the row
    return data

def dependent_var(real_X, real_y, syn_X, model="XGBoost", nestimator=600, scale=False):
    # Try scale==TRUE
    if scale:
        scaler = MinMaxScaler() # Is MinMax the best option?
        real_X = scaler.fit_transform(real_X)
        syn_X = scaler.transform(syn_X)
        # normalize all data
        real_y = scaler.fit_transform(real_y)
    X_train, X_test, y_train, y_test = train_test_split(real_X, real_y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)
    # random forest?
    # multiregressor output
    if model == "XGBoost":
        xgb_regressor = XGBRegressor(
            n_estimators=2500,
            learning_rate=0.03,
            max_depth=10,
            min_child_weight=5,
            gamma=0.1,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=1,
            reg_lambda=5,
            objective='reg:squarederror',
            early_stopping_rounds=100
        )
        # xgb_regressor = XGBRegressor(n_estimators=nestimator,
        #                              learning_rate=0.05,
        #                              max_depth=10,
        #                              subsample=0.8,
        #                              colsample_bytree=0.8,
        #                              early_stopping_rounds=50,
        #                              objective='reg:squarederror')
        # xgb_regressor = XGBRegressor(n_estimators=nestimator,
        #                              learning_rate=0.1,
        #                              max_depth=6,
        #                              subsample=0.8,
        #                              colsample_bytree=0.8,
        #                              early_stopping_rounds=50,
        #                              objective='reg:squarederror')
        xgb_regressor.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)
        y_pred = xgb_regressor.predict(X_test)
        syn_y = xgb_regressor.predict(syn_X)

    elif model == "LightGBM":
        regressor = MultiOutputRegressor(
            LGBMRegressor(
                n_estimators=nestimator,
                learning_rate=0.1,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8
            )
        )
        regressor.fit(X_train, y_train)
        y_pred = regressor.predict(X_test)
        syn_y = regressor.predict(syn_X)

    elif model == "MultiOutputRegressor":
        regr = MultiOutputRegressor(Ridge(random_state=100))
        regr.fit(X_train, y_train)
        y_pred = regr.predict(X_test)
        syn_y = regr.predict(syn_X)

    elif model == "RandomForest":
        rf = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=100))
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        syn_y = rf.predict(syn_X)

    else:
        raise ValueError("Unsupported model type.")

    mse = mean_squared_error(y_test, y_pred)
    print(f"Real Data Mean Squared Error for {model}:", mse)
    return syn_y

def main():
    # for nestimator in [600, 800, 1000]:
    #     for model in ["XGBoost", "LightGBM"]:
    for nestimator in [2000]:
        for model in ["XGBoost"]:
            data_real = pd.read_csv(r"pf_dataset_FINAL_correct_PQ_IEEE118small.csv")
            # Independent variables
            gen_pg_cols = [f"PG{i}" for i in range(1, 119)]
            gen_qg_cols = [f"QG{i}" for i in range(1, 119)]
            pl_cols = [f"PL{i}" for i in range(1, 119)]
            ql_cols = [f"QL{i}" for i in range(1, 119)]
            qsh_cols = [f"Qsh{i}" for i in range(1, 119)]
            q_cols = [f"Q{i}" for i in range(1, 119)]
            p_cols = [f"P{i}" for i in range(1, 119)]

            # Targets (dependent variables): bus voltage magnitude + angle
            v_cols = [f"V{i}" for i in range(1, 119)]
            angle_cols = [f"theta{i}" for i in range(1, 119)]

            X_cols = gen_pg_cols + gen_qg_cols + pl_cols + ql_cols  # + qsh_cols
            y_cols = v_cols + angle_cols

            X = data_real[X_cols].copy()
            y = data_real[y_cols].copy()
            print("Generating independent data")
            independent_values = independent_var(X.copy(), ["PL59", "PL116", "PL90"], 0.1)






        # ["PL60", "PL117", "PL81", "PL57", "PL43", "PL16", "PL50", "PL78", "PL41", "PL20", "PL35", "PL82","PL7","PL45","PL2","PL108","PL13","PL33","PL101","PL95"], 0.3)
            # ["PL37", "PL30", "PL5", "PL9", "PL38", "PL11", "PL81", "PL67", "PL68", "PL63", "PL64", "PL71", "PL93", "PL88"] pandapower
            # ["PL23", "PL30", "PL38", "PL64", "PL68", "PL102"] statistical analysis
            print("Generating dependent data")
            dependent_values = dependent_var(X, y, independent_values, model, nestimator, False)

            result_df = pd.concat([independent_values, pd.DataFrame(dependent_values, columns=y.columns)], axis=1)

            print("Saving results")
            result_df.to_csv(f'IEEE118_PL_Class_FDI_NotScaled_range+1.3_nestimator{nestimator}_{model}_model4.csv', index=False)
            # result_df.to_csv('Real_AllScaled.csv', index=False)
            # result_df.to_csv('Real_XScaled.csv', index=False)

if __name__ == "__main__":
    main()
