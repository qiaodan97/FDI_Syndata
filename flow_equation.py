import pandapower as pp
import pandapower.networks as pn
import numpy as np
import pandas as pd


def compute_power(V, theta, Ybus, baseMVA):
    G = Ybus.real
    B = Ybus.imag
    N = V.shape[1]

    P = np.zeros_like(V)
    Q = np.zeros_like(V)

    for i in range(N):
        for j in range(N):
            theta_ij = theta[:, i] - theta[:, j]

            P[:, i] += V[:, i] * V[:, j] * (
                G[i, j] * np.cos(theta_ij) +
                B[i, j] * np.sin(theta_ij)
            )

            Q[:, i] += V[:, i] * V[:, j] * (
                G[i, j] * np.sin(theta_ij) -
                B[i, j] * np.cos(theta_ij)
            )

    return P, Q

def build_values_14(df):
    net = pn.case14()  # case14()
    pp.runpp(net, calculate_voltage_angles=True)
    Ybus = net._ppc['internal']['Ybus']
    baseMVA = net._ppc["baseMVA"]

    n_samples = len(df)
    V = np.zeros((n_samples, 14))
    theta = np.zeros((n_samples, 14))
    P = np.zeros((n_samples, 14))
    Q = np.zeros((n_samples, 14))
    PL = np.zeros((n_samples, 14))
    QL = np.zeros((n_samples, 14))
    PG = np.zeros((n_samples, 14))
    QG = np.zeros((n_samples, 14))


    for i in range(14):
        V[:, i] = df[f"Bus{i}_V"].values
        theta[:, i] = df[f"Bus{i}_angle"].values
    theta = np.deg2rad(theta)

    gen_buses = [1, 2, 5, 7]
    load_buses = [1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13]

    # for b in gen_buses:
    #     P[:, b] += df[f"GenBus{b}_PG"].values
    #     Q[:, b] += df[f"GenBus{b}_QG"].values
    #
    # for b in load_buses:
    #     P[:, b] -= df[f"Bus{b}_PL"].values
    #     Q[:, b] -= df[f"Bus{b}_QL"].values

    for b in gen_buses:
        PG[:, b] = df[f"GenBus{b}_PG"] / baseMVA
        QG[:, b] = df[f"GenBus{b}_QG"] / baseMVA

    for b in load_buses:
        PL[:, b] = df[f"Bus{b}_PL"] / baseMVA
        QL[:, b] = df[f"Bus{b}_QL"] / baseMVA
    P = PG - PL
    Q = QG - QL
    return V, theta, Ybus, P, Q, baseMVA

def build_values_118(df):
    net = pn.case118()
    pp.runpp(net, calculate_voltage_angles=True)
    Ybus = net._ppc['internal']['Ybus']
    baseMVA = net._ppc["baseMVA"]

    n_samples = len(df)
    V = np.zeros((n_samples, 118))
    theta = np.zeros((n_samples, 118))
    P = np.zeros((n_samples, 118))
    Q = np.zeros((n_samples, 118))
    PG = np.zeros((n_samples, 118))
    QG = np.zeros((n_samples, 118))
    PL = np.zeros((n_samples, 118))
    QL = np.zeros((n_samples, 118))

    gen_buses = [
        1, 4, 6, 8, 10, 12, 15, 18, 19, 24, 25, 26, 27, 31, 32, 34,
        36, 40, 42, 46, 49, 54, 55, 56, 59, 61, 62, 65, 66, 69, 70,
        72, 73, 74, 76, 77, 80, 85, 87, 89, 90, 91, 92, 99, 100, 103,
        104, 105, 107, 110, 111, 112, 113, 116
    ]

    load_buses = [
        2, 3, 5, 7, 9, 11, 13, 14, 16, 17, 20, 21, 22, 23, 28, 29, 30,
        33, 35, 37, 38, 39, 41, 43, 44, 45, 47, 48, 50, 51, 52, 53, 57,
        58, 60, 63, 64, 67, 68, 71, 75, 78, 79, 81, 82, 83, 84, 86, 88,
        93, 94, 95, 96, 97, 98, 101, 102, 106, 108, 109, 114, 115, 117, 118
    ]

    for b in gen_buses:
        idx = b -1
        V[:, idx] = df[f"VGM{b}"].values
        theta[:, idx] = df[f"VGA{b}"].values
        PG[:, idx] = df[f"PG{b}"] / baseMVA
        QG[:, idx] = df[f"QG{b}"] / baseMVA

    for b in load_buses:
        idx = b -1
        V[:, idx] = df[f"VLM{b}"].values
        theta[:, idx] = df[f"VLA{b}"].values
        PL[:, idx] = df[f"PL{b}"] / baseMVA
        QL[:, idx] = df[f"QL{b}"] / baseMVA
    P = PG - PL
    Q = QG - QL

    theta = np.deg2rad(theta)
    return V, theta, Ybus, P, Q, baseMVA

if __name__ == "__main__":
    # df_names = ["IEEE14_PL_Class_FDI_NotScaled_range+1.3_nestimator600_LightGBM.csv",
    #             "IEEE14_PL_Class_FDI_NotScaled_range+1.3_nestimator800_LightGBM.csv",
    #             "IEEE14_PL_Class_FDI_NotScaled_range+1.3_nestimator1000_LightGBM.csv",
    #             "IEEE14_PL_Class_FDI_NotScaled_range+1.3_nestimator600_XGBoost.csv",
    #             "IEEE14_PL_Class_FDI_NotScaled_range+1.3_nestimator800_XGBoost.csv",
    #             "IEEE14_PL_Class_FDI_NotScaled_range+1.3_nestimator1000_XGBoost.csv",]
    # df_names = ["D:\FDI\_FromMarzia\_FromMarzia\FDIG\FDIG\ieee14_nonzero_pg_dataset.csv"]

    df_names = ["IEEE118_PL_Class_FDI_NotScaled_range+1.3_nestimator600_LightGBM.csv",
                "IEEE118_PL_Class_FDI_NotScaled_range+1.3_nestimator800_LightGBM.csv",
                "IEEE118_PL_Class_FDI_NotScaled_range+1.3_nestimator1000_LightGBM.csv",
                "IEEE118_PL_Class_FDI_NotScaled_range+1.3_nestimator600_XGBoost.csv",
                "IEEE118_PL_Class_FDI_NotScaled_range+1.3_nestimator800_XGBoost.csv",
                "IEEE118_PL_Class_FDI_NotScaled_range+1.3_nestimator1000_XGBoost.csv",]
    # df_names = ["D:\FDI\OneDrive_1_2-16-2025\IEEE118Normal_Corrected.csv"]

    for df_name in df_names:
        df = pd.read_csv(df_name)

        N = len(df)
        V, theta, Ybus, P_true, Q_true, baseMVA  = build_values_118(df)
        # V, theta, Ybus, P_true, Q_true, baseMVA = build_values_14(df)

        P_calc, Q_calc = compute_power(V, theta, Ybus, baseMVA)
        # result_df.append({
        #     "P_calc": P_calc,
        #     "Q_calc": Q_calc,
        #     "P_calc_percentage": P_calc_per,
        #     "Q_calc_percentage": Q_calc_per,
        # })
        error_P = np.abs(P_calc - P_true)
        error_Q = np.abs(Q_calc - Q_true)

        # eps = 1e-2

        eps = 0.5

        error_P_per = np.full_like(P_true, np.nan, dtype=float)
        error_Q_per = np.full_like(Q_true, np.nan, dtype=float)

        mask_P = np.abs(P_true) > eps
        mask_Q = np.abs(Q_true) > eps

        error_P_per[mask_P] = error_P[mask_P] / np.abs(P_true[mask_P])*100
        error_Q_per[mask_Q] = error_Q[mask_Q] / np.abs(Q_true[mask_Q])*100

        violation_P = error_P.mean(axis=1)
        violation_Q = error_Q.mean(axis=1)
        violation_P_per = np.nanmean(error_P_per, axis=1)
        violation_Q_per = np.nanmean(error_Q_per, axis=1)

        df_result = pd.DataFrame({
            "P_mean_dif": violation_P,
            "Q_mean_dif": violation_Q,
            "P_mean_dif_per": violation_P_per,
            "Q_mean_dif_per": violation_Q_per,
        })

        # gen_buses = [1, 4, 6, 8, 10, 12, 15, 18, 19, 24, 25, 26, 27, 31, 32, 34, 36, 40, 42, 46, 49, 54, 55, 56, 59, 61,
        #              62,
        #              65, 66, 69, 70, 72, 73, 74, 76, 77, 80, 85, 87, 89, 90, 91, 92, 99, 100, 103, 104, 105, 107, 110,
        #              111,
        #              112, 113, 116]
        # load_buses = [2, 3, 5, 7, 9, 11, 13, 14, 16, 17, 20, 21, 22, 23, 28, 29, 30, 33, 35, 37, 38, 39, 41, 43, 44, 45,
        #               47,
        #               48, 50, 51, 52, 53, 57, 58, 60, 63, 64, 67, 68, 71, 75, 78, 79, 81, 82, 83, 84, 86, 88, 93, 94,
        #               95,
        #               96, 97, 98, 101, 102, 106, 108, 109, 114, 115, 117, 118]
        # gen_idx = np.array(gen_buses) - 1
        # load_idx = np.array(load_buses) - 1
        #
        # print("Mean abs P error on gen buses:", np.mean(np.abs(P_calc[:, gen_idx] - P_true[:, gen_idx])))
        # print("Mean abs P error on load buses:", np.mean(np.abs(P_calc[:, load_idx] - P_true[:, load_idx])))
        #
        # print("Mean abs Q error on gen buses:", np.mean(np.abs(Q_calc[:, gen_idx] - Q_true[:, gen_idx])))
        # print("Mean abs Q error on load buses:", np.mean(np.abs(Q_calc[:, load_idx] - Q_true[:, load_idx])))

        # df_result.to_csv(f"IEEE14_ac_pf_check_results_realdata.csv", index=False)
        # df_result.to_csv(f"IEEE14_ac_pf_check_results_{df_name}.csv", index=False)
        # df_result.to_csv(f"IEEE118_ac_pf_check_results_realdata.csv", index=False)
        df_result.to_csv(f"IEEE118_ac_pf_check_results_{df_name}.csv", index=False)
