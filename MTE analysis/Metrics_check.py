import pandas as pd
import numpy as np
from datetime import datetime, timedelta
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

    starting_date = '2024-11-23'
    repo = starting_date.split('-')[1:]
    repo = ''.join(repo)
    res_cesgcn = 0
    res_ensemble =0
    res_baseline = 0
    c = 0
    rmse_cesgcn_all = 0
    mae_cesgcn_all = 0
    mape_cesgcn_all = 0
    for index_version in range(10):
        c = 'V'+str(index_version+1)
        CESGCN_overall = pd.read_csv(
             '1123/CESGCN_LR_0.02_2024-11-23_0.2_dim_20_onego_100_MSE_weight_decay_0.02_TRAIN_V10_selected.csv'.format(c))
        #CESGCN_overall = pd.read_csv(
        #    '1130/CESGCN_LR_0.02_2024-11-30_quantile_0.2_dim_20_onego_100_epochs_MSE_weight_decay_0.02_TRAIN_V10_selected.csv')
        flusight_ensemble = pd.read_csv('{}-FluSight-ensemble.csv'.format(starting_date))
        flusight_baseline = pd.read_csv('{}-FluSight-baseline.csv'.format(starting_date))
        CESGCN_overall = CESGCN_overall[CESGCN_overall['location'] != 'US']
        # UNIFIED = UNIFIED[UNIFIED['horizon'] != -1]
        # CESGCN_overall = CESGCN_overall[CESGCN_overall['horizon'] != -1]

        '''
        rmse = np.sqrt(((new_df - ground_truth_df.values) ** 2).mean())
        mae = (np.abs(new_df - ground_truth_df.values)).mean()
        mape = (np.abs((new_df - ground_truth_df.values) / ground_truth_df.values)).mean() * 100
        '''

        starting_date_dt = datetime.strptime(starting_date, '%Y-%m-%d')
        count = 0
        for index in range(1):
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

            truth = pd.read_csv('CDC_DATA/transferred_hospital_admission_0201.csv')
            which_ind = np.where(truth['date'].values == horizon_dt)[0][0]
            row_dict = truth.iloc[which_ind, :].to_dict()
            row_dict.pop('date')

            # UNIFIED['location'] = pd.to_numeric(UNIFIED['location'], errors='coerce')
            # UNIFIED['location'] = UNIFIED['location'].astype(int).astype(str).str.zfill(2)
            # UNIFIED = UNIFIED[UNIFIED['target_end_date'] == '2025-01-25']

            CESGCN_overall['location'] = pd.to_numeric(CESGCN_overall['location'], errors='coerce')
            CESGCN_overall['location'] = CESGCN_overall['location'].astype(int).astype(str).str.zfill(2)
            CESGCN = CESGCN_overall[CESGCN_overall['fct_date'] == horizon_dt]

            CESGCN["truth"] = CESGCN["location"].map(row_dict)
            # UNIFIED["truth"] = UNIFIED["location"].map(row_dict)

            baseline["truth"] = baseline["location"].map(row_dict)
            ensemble["truth"] = ensemble["location"].map(row_dict)

            rmse_CESGCN = np.sqrt(((CESGCN['value'].values - CESGCN['truth'].values) ** 2).mean())
            mae_CESGCN = (np.abs((CESGCN['value'].values - CESGCN['truth'].values))).mean()

            mask = CESGCN['truth'].values != 0
            # Compute MAPE excluding zero values
            mape_CESGCN = (np.abs((CESGCN['value'].values[mask] - CESGCN['truth'].values[mask])
                                  / CESGCN['truth'].values[mask])).mean() * 100

            rmse_cesgcn_all = rmse_cesgcn_all + rmse_CESGCN
            mae_cesgcn_all = mae_cesgcn_all + mae_CESGCN
            mape_cesgcn_all = mape_cesgcn_all + mape_CESGCN


        '''
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
        '''


    print('The Mean RMSE for CESGCN is '+str(rmse_cesgcn_all/10))
    print('The Mean MAE for CESGCN is '+str(mae_cesgcn_all/10))
    print('The Mean MAPE for CESGCN is '+str(mape_cesgcn_all/10))
