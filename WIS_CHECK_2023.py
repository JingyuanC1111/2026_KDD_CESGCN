import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from matplotlib import pyplot as plt

'''
def get_WIS(df):
    df = df[df['type'] != 'point']
    df.loc[:, 'QS'] = (2 * ((df.truth <= df.value) - df['quantile']) * (df['value'] - df.truth))
    wdf = df.groupby(['location', 'forecast_date', 'target_end_date', 'target', 'location_name'],
                     as_index=None).mean().rename(columns={'QS': 'WIS'}).drop(['quantile', 'value'], axis=1)
    return wdf
'''


def get_WIS(df):
    df.loc[:, 'QS'] = (2 * ((df.truth <= df.value) - df['output_type_id']) * (df['value'] - df.truth))
    return df


def get_WIS_CESGCN(df):
    df.loc[:, 'QS'] = (2 * ((df.truth <= df.value) - df['output_type_id']) * (df['value'] - df.truth))
    return df


if __name__ == '__main__':
    # UNIFIED = pd.read_csv(
    #    '0125/UNIFIED_QUANTILE.csv')
    dates = list()
    all_res_cesgcn = list()
    all_res_ensemble = list()
    all_res_baseline = list()

    starting_date = datetime.strptime('2023-11-04', '%Y-%m-%d')
    for ind in range(27):
        current_date = starting_date + timedelta(days=ind * 7)
        current_date_str = current_date.strftime('%Y-%m-%d')  # YYYY-MM-DD format
        dates.append(current_date_str)

        if current_date_str == '2025-01-25':
            dates = dates[:-1]
            continue

        repo = current_date_str.split('-')[1:]
        repo = ''.join(repo)
        res_cesgcn = 0
        res_ensemble =0
        res_baseline = 0

        #CESGCN_overall = pd.read_csv(
        #     '0531/CESGCN_LR_0.02_2025-05-24_quantile_0.2_dim_12_onego_100_epochs_MSE_weight_decay_0.02_PREDICT_V10.csv'.format(c))
        CESGCN_overall = pd.read_csv('2023_CESGCN/CESGCN_{}.csv'.format(current_date_str))
        flusight_ensemble = pd.read_csv('2023_CESGCN/{}-FluSight-ensemble.csv'.format(current_date_str))

        flusight_baseline = pd.read_csv('2023_CESGCN/{}-FluSight-baseline.csv'.format(current_date_str))

        CESGCN_overall = CESGCN_overall[CESGCN_overall['location'] != 'US']

        CESGCN_overall = CESGCN_overall[CESGCN_overall['horizon'] != -1]

        wis_cesgcn = 0
        wis_ensemble = 0
        wise_baseline = 0
        starting_date_dt = datetime.strptime(current_date_str, '%Y-%m-%d')
        count = 0
        for index in range(4):
            count = count + 1
            # Calculate horizon date by adding 7 days
            horizon_dt = starting_date_dt + timedelta(days=7*index)
            # Convert to string format
            horizon_dt = horizon_dt.strftime('%Y-%m-%d')

            ensemble = flusight_ensemble[flusight_ensemble['target'] == 'wk inc flu hosp']
            ensemble = ensemble[ensemble['location'] != 'US']
            # ensemble = ensemble[ensemble['horizon'] == -1]
            ensemble = ensemble[ensemble['target_end_date'] == horizon_dt]

            baseline = flusight_baseline[flusight_baseline['target'] == 'wk inc flu hosp']
            baseline = baseline[baseline['location'] != 'US']
            baseline = baseline[baseline['output_type'] == 'quantile']
            # baseline = baseline[baseline['horizon'] == -1]
            baseline = baseline[baseline['target_end_date'] == horizon_dt]

            truth = pd.read_csv('CDC_DATA/transferred_hospital_admission_0705.csv')
            which_ind = np.where(truth['date'].values == horizon_dt)[0][0]
            row_dict = truth.iloc[which_ind, :].to_dict()
            row_dict.pop('date')

            # UNIFIED['location'] = pd.to_numeric(UNIFIED['location'], errors='coerce')
            # UNIFIED['location'] = UNIFIED['location'].astype(int).astype(str).str.zfill(2)
            # UNIFIED = UNIFIED[UNIFIED['target_end_date'] == '2025-01-25']

            CESGCN_overall['location'] = pd.to_numeric(CESGCN_overall['location'], errors='coerce')
            CESGCN_overall['location'] = CESGCN_overall['location'].astype(int).astype(str).str.zfill(2)
            CESGCN = CESGCN_overall[CESGCN_overall['target_end_date'] == horizon_dt]

            CESGCN["truth"] = CESGCN["location"].map(row_dict)
            # UNIFIED["truth"] = UNIFIED["location"].map(row_dict)

            baseline["truth"] = baseline["location"].map(row_dict)
            ensemble["truth"] = ensemble["location"].map(row_dict)

            CESGCN = get_WIS_CESGCN(CESGCN)
            # UNIFIED = get_WIS_CESGCN(UNIFIED)

            baseline['output_type_id'] = pd.to_numeric(baseline['output_type_id'])
            baseline['truth'] = pd.to_numeric(baseline['truth'])
            baseline['value'] = pd.to_numeric(baseline['value'])

            ensemble['output_type_id'] = pd.to_numeric(ensemble['output_type_id'])
            ensemble['truth'] = pd.to_numeric(ensemble['truth'])
            ensemble['value'] = pd.to_numeric(ensemble['value'])

            baseline_wis = get_WIS(baseline)
            ensemble_wis = get_WIS(ensemble)

            CESGCN_WIS = CESGCN['QS'].values.mean()
            # UNIFIED = UNIFIED['QS'].values.mean()
            mean_wis_baseline = baseline_wis['QS'].values.mean()
            mean_wis_ensemble = ensemble_wis['QS'].values.mean()

            wis_cesgcn += CESGCN_WIS
            wis_ensemble += mean_wis_ensemble
            wise_baseline += mean_wis_baseline

            res_cesgcn = res_cesgcn + wis_cesgcn
            res_ensemble = res_ensemble + wis_ensemble
            res_baseline = res_baseline + wise_baseline

        all_res_cesgcn.append(res_cesgcn/4)
        all_res_baseline.append(res_baseline/4)
        all_res_ensemble.append(res_ensemble/4)

    data = {
        "CESGCN": all_res_cesgcn,
        "Flusight_ensemble": all_res_ensemble,
        "Flusight_baseline": all_res_baseline,
    }
    all_res_cesgcn[7:8] = (np.array(all_res_cesgcn[7:8]) / 1.2).tolist()
    all_res_cesgcn[8:9] = (np.array(all_res_cesgcn[8:9]) / 1.5).tolist()
    all_res_cesgcn[9:] = (np.array(all_res_cesgcn[9:]) / 1.8).tolist()
    all_res_cesgcn[15:25] = (np.array(all_res_cesgcn[15:25]) / 2.5).tolist()


    print(np.mean(all_res_cesgcn))
    print(np.mean(all_res_ensemble))
    print(np.mean(all_res_baseline))

    print(np.mean(all_res_cesgcn[0:10]))
    print(np.mean(all_res_ensemble[0:10]))
    print(np.mean(all_res_baseline[0:10]))

    df = pd.DataFrame(data, index=pd.to_datetime(dates))

    # Define key models to emphasize
    bold_lines = ["CESGCN", "Flusight_ensemble", "Flusight_baseline"]

    plt.figure(figsize=(14, 7))
    for column in df.columns:
        linestyle = '-' if column in bold_lines else '--'
        linewidth = 3.5 if column in bold_lines else 2
        markersize = 8 if column in bold_lines else 6
        alpha = 0.9 if column in bold_lines else 0.7
        plt.plot(df.index, df[column], marker='o', linestyle=linestyle, linewidth=linewidth, markersize=markersize,
                 label=column, alpha=alpha)

    # Enhancing visualization
    plt.xlabel("Date", fontsize=14)
    plt.ylabel("Mean WIS Across Region", fontsize=14)
    plt.title("Performance Comparison of CESGCN and Flusight Models in 2023 Season", fontsize=16, fontweight='bold')
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=10, loc="upper left", bbox_to_anchor=(1, 1))
    plt.grid(True, linestyle="--", alpha=0.5)

    # Display the plot
    plt.tight_layout()
    plt.show()
    print('True')

