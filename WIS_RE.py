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
    all_res_cesgcn = list()
    all_res_ensemble = list()
    all_res_baseline = list()

    starting_date = '2025-01-04'

    #CESGCN_overall = pd.read_csv(
    #     '0531/CESGCN_LR_0.02_2025-05-24_quantile_0.2_dim_12_onego_100_epochs_MSE_weight_decay_0.02_PREDICT_V10.csv'.format(c))
    CESGCN_overall = pd.read_csv('0104_RE/CESGCN_LR_0.01_2025-01-04_quantile_0.2_dim_14_onego_100_epochs_MSE_weight_decay_0.02_TRAIN_V5.csv')
    flusight_ensemble = pd.read_csv('{}-FluSight-ensemble.csv'.format(starting_date))

    flusight_baseline = pd.read_csv('{}-FluSight-baseline.csv'.format(starting_date))

    CESGCN_overall = CESGCN_overall[CESGCN_overall['location'] != 'US']

    CESGCN_overall = CESGCN_overall[CESGCN_overall['horizon'] != -1]

    wis_cesgcn = 0
    wis_ensemble = 0
    wise_baseline = 0
    count = 0
    starting_date_dt = datetime.strptime(starting_date, '%Y-%m-%d')
    for index in range(4):
        count = count + 1
        # Calculate horizon date by adding 7 days
        horizon_dt = starting_date_dt + timedelta(days=7 * index)
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


    print('CESGCN is :'+ str(wis_cesgcn/4))
    print('FluSight Ensemble is :'+ str(wis_ensemble/4))
    print('FluSight Baseline is :'+ str(wise_baseline/4))



