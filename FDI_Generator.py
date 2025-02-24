import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor


def independent_var(data, cols, changing_rate):
    for col in cols:
        data[col] *= np.random.uniform(1-changing_rate, 1+changing_rate)
    return data

def dependent_var(real_X, real_y, syn_X, model="XGBoost", scale=False):
    if model=="XGBoost":
        if scale:
            scaler = MinMaxScaler() # Is MinMax the best option?
            real_X = scaler.fit_transform(real_X)
            syn_X = scaler.transform(syn_X)
            # normalize all data
            real_y = scaler.fit_transform(real_y)
        X_train, X_test, y_train, y_test = train_test_split(real_X, real_y, test_size=0.2, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

        xgb_regressor = XGBRegressor(n_estimators=1500,
                                     learning_rate=0.07,
                                     max_depth=6,
                                     subsample=0.8,
                                     colsample_bytree=0.8,
                                     early_stopping_rounds=50,
                                     objective='reg:squarederror')
        xgb_regressor.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)

        y_pred_xgb = xgb_regressor.predict(X_test)
        mse_xgb = mean_squared_error(y_test, y_pred_xgb)
        print("Real Data Mean Squared Error for XGBoost:", mse_xgb)

        syn_y = xgb_regressor.predict(syn_X)
    return syn_y

def main():
    data_real = pd.read_csv(r"E:\FDI\OneDrive_1_2-16-2025\IEEE118Normal_Corrected.csv")
    # Independent variables
    vgm_columns = [col for col in data_real.columns if 'VGM' in col]
    pg_columns = [col for col in data_real.columns if 'PG' in col]
    pl_columns = [col for col in data_real.columns if 'PL' in col]
    ql_columns = [col for col in data_real.columns if 'QL' in col]

    # Dependent variables
    vlm_columns = [col for col in data_real.columns if 'VLM' in col]
    vla_columns = [col for col in data_real.columns if 'VLA' in col]
    vga_columns = [col for col in data_real.columns if 'VGA' in col]

    X = data_real[vgm_columns + pg_columns + pl_columns + ql_columns]
    y = data_real[vlm_columns + vla_columns + vga_columns]

    print("Generating independent data")
    independent_values = independent_var(X.copy(), ["PL30"], 0.3)
    print("Generating dependent data")
    dependent_values = dependent_var(X, y, independent_values, "XGBoost", False)

    result_df = pd.concat([independent_values, pd.DataFrame(dependent_values, columns=y.columns)], axis=1)

    print("Saving results")
    result_df.to_csv('PL_Class_FDI_NotScaled.csv', index=False)
    # result_df.to_csv('Real_AllScaled.csv', index=False)
    # result_df.to_csv('Real_XScaled.csv', index=False)

if __name__ == "__main__":
    main()
