import pandapower as pp
import pandapower.networks as pn
import numpy as np
import pandas as pd
import scipy


def compute_power(V, theta, Ybus):
    G = Ybus.real
    B = Ybus.imag
    N = V.shape[1]
    P = np.zeros(V.shape, dtype=float)
    Q = np.zeros(V.shape, dtype=float)

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

def build_values(df, dataset_num):
    if dataset_num == 14:
        net = pn.case14()  # case14()
    else:
        net = pn.case118()
    pp.runpp(net, calculate_voltage_angles=True)
    Ybus = net._ppc['internal']['Ybus']
    baseMVA = net._ppc["baseMVA"]

    n_samples = len(df)
    V = np.zeros((n_samples, dataset_num))
    theta = np.zeros((n_samples, dataset_num))
    P = np.zeros((n_samples, dataset_num))
    Q = np.zeros((n_samples, dataset_num))
    Qsh = np.zeros((n_samples, dataset_num))
    PL = np.zeros((n_samples, dataset_num))
    QL = np.zeros((n_samples, dataset_num))
    PG = np.zeros((n_samples, dataset_num))
    QG = np.zeros((n_samples, dataset_num))
    #
    #
    # for i in range(14):
    #     V[:, i] = df[f"Bus{i}_V"].values
    #     theta[:, i] = df[f"Bus{i}_angle"].values
    # theta = np.deg2rad(theta)

    # gen_buses = [1, 2, 5, 7]
    # load_buses = [1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13]
    #
    # # for b in gen_buses:
    # #     P[:, b] += df[f"GenBus{b}_PG"].values
    # #     Q[:, b] += df[f"GenBus{b}_QG"].values
    # #
    # # for b in load_buses:
    # #     P[:, b] -= df[f"Bus{b}_PL"].values
    # #     Q[:, b] -= df[f"Bus{b}_QL"].values
    #
    print(net.shunt)
    print(net.res_shunt)
    for i in range(dataset_num):
        b = i + 1
        V[:, i] = df[f"V{b}"].values
        theta[:, i] = df[f"theta{b}"].values
        # P[:, i] = df[f"P{b}"].values
        # Q[:, i] = df[f"Q{b}"].values
        # Qsh[:, i] = df[f"Qsh{b}"].values
        PL[:, i] = df[f"PL{b}"]
        QL[:, i] = df[f"QL{b}"]
        PG[:, i] = df[f"PG{b}"]
        QG[:, i] = df[f"QG{b}"]
    # P = PG * baseMVA - PL  * baseMVA
    # Q = QG * baseMVA  - QL * baseMVA
    P = PG - PL
    Q = QG - QL
    # Q = QG * baseMVA  - QL * baseMVA + Qsh
    # Q = Q - Qsh
    # for b in range(14):
    #     PL[:, b] = df[f"Bus{b}_PL"] / baseMVA
    #     QL[:, b] = df[f"Bus{b}_QL"] / baseMVA
    # P = PG - PL
    # Q = QG - QL
    return V, theta, P, Q, Ybus, baseMVA

if __name__ == "__main__":
    # df_names = ["IEEE14_PL_Class_FDI_NotScaled_range+1.3_nestimator600_LightGBM.csv",
    #             "IEEE14_PL_Class_FDI_NotScaled_range+1.3_nestimator800_LightGBM.csv",
    #             "IEEE14_PL_Class_FDI_NotScaled_range+1.3_nestimator1000_LightGBM.csv",
    #             "IEEE14_PL_Class_FDI_NotScaled_range+1.3_nestimator600_XGBoost.csv",
    #             "IEEE14_PL_Class_FDI_NotScaled_range+1.3_nestimator800_XGBoost.csv",
    #             "IEEE14_PL_Class_FDI_NotScaled_range+1.3_nestimator1000_XGBoost.csv",]
    # df_names = ["pf_dataset_FINAL_correct_PQ_IEEE14.csv"]
    # df_names = ["pf_dataset_FINAL_correct_PQ_IEEE14small.csv"]
    df_names = [r"IEEE14_PL_Class_FDI_NotScaled_range+1.3_nestimator600_XGBoost_model2.csv"]

    # df_names = ["IEEE118_PL_Class_FDI_NotScaled_range+1.3_nestimator600_LightGBM.csv",
    #             "IEEE118_PL_Class_FDI_NotScaled_range+1.3_nestimator800_LightGBM.csv",
    #             "IEEE118_PL_Class_FDI_NotScaled_range+1.3_nestimator1000_LightGBM.csv",
    #             "IEEE118_PL_Class_FDI_NotScaled_range+1.3_nestimator600_XGBoost.csv",
    #             "IEEE118_PL_Class_FDI_NotScaled_range+1.3_nestimator800_XGBoost.csv",
    #             "IEEE118_PL_Class_FDI_NotScaled_range+1.3_nestimator1000_XGBoost.csv",]
    # df_names = ["pf_dataset_FINAL_correct_PQ_IEEE118small.csv"]
    # df_names = ["IEEE118_PL_Class_FDI_NotScaled_range+1.3_nestimator2000_XGBoost_model4.csv"]
    # dataset_num = 118
    dataset_num = 14
    for df_name in df_names:
        df = pd.read_csv(df_name)

        N = len(df)
        V, theta, P_true, Q_true, Ybus, baseMVA  = build_values(df, dataset_num)

        theta = np.deg2rad(theta)
        P_calc, Q_calc = compute_power(V, theta, Ybus)
        P_calc = P_calc * baseMVA
        Q_calc = Q_calc * baseMVA
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

        error_P_per[mask_P] = error_P[mask_P] / np.abs(P_true[mask_P])
        error_Q_per[mask_Q] = error_Q[mask_Q] / np.abs(Q_true[mask_Q])

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

        # df_result.to_csv(f"IEEE14_ac_pf_check_results_realdata.csv", index=False)
        # df_result.to_csv(f"IEEE14_ac_pf_check_results_{df_name}.csv", index=False)
        # df_result.to_csv(f"IEEE118_ac_pf_check_results_realdata.csv", index=False)
        df_result.to_csv(f"IEEE114_ac_pf_check_results_{df_name}.csv", index=False)
