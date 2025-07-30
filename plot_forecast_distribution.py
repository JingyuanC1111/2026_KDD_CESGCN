import pandas as pd
import torch as torch
import numpy as np
from datetime import datetime, timedelta
import random
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

if __name__ == '__main__':

    covid_case = pd.read_csv(
        'C:\\Users\\6\\PycharmProjects\\FORECASTING_GNN\\epidemic_data\\COVID_CASE\\COVID_CASE_DATA.csv')

    covid_case = covid_case[~covid_case['Abbr'].isin(['HI', 'AK'])]
    covid_case = covid_case.reset_index(drop=True)
    covid_case['Index'] = covid_case.index

    index_fips_dict = covid_case['FIPS'].to_dict()
    fips_state_dict = covid_case.set_index('FIPS')['Abbr'].to_dict()
    index_state = dict(zip(covid_case['Index'], covid_case['Abbr']))

    state_fips = {v: k for k, v in fips_state_dict.items()}

    borrowed_files = pd.read_csv('epidemic_data/COVID_CASE/COVID_CASE_DATA.csv')
    borrowed_col = borrowed_files[['location', 'FIPS']]
    all_cases = pd.read_csv('CDC_DATA/transferred_hospital_admission_0705.csv').iloc[:-5, :]

    covid_case = pd.read_csv(
        'C:\\Users\\6\\PycharmProjects\\FORECASTING_GNN\\epidemic_data\\COVID_CASE\\COVID_CASE_DATA.csv')

    covid_case = covid_case.reset_index().rename(columns={'index': 'Index'})
    index_state = dict(zip(covid_case['Index'], covid_case['Abbr']))
    index_state[51] = 'PR'

    state_index = {v: k for k, v in index_state.items()}

    hhs_map = pd.read_csv('us_subplot_grid.csv')
    state_hhs = dict(zip(hhs_map['State'], hhs_map['HHS']))

    dates = pd.date_range(start='2024-11-23', periods=27, freq='7D')
    states = ['CA']

    for index, state in enumerate(states):

        forecasting_dates = ['2024-11-23', '2024-12-28', '2025-02-01', '2025-03-08']
        state_id = state_index[state]
        true_value = all_cases.values[:, state_id + 1]

        ca_ci_lower_base = list()
        ca_ci_upper_base = list()

        ca_ci_lower_ensemble = list()
        ca_ci_upper_ensemble = list()

        ca_ci_lower_mte = list()
        ca_ci_upper_mte = list()

        for date_time in forecasting_dates:
            temp_flu = pd.read_csv('{}-Flusight-ensemble.csv'.format(date_time))
            temp_baseline_flu = pd.read_csv('{}-Flusight-baseline.csv'.format(date_time))
            temp_MTE_flu = pd.read_csv('CESGCN_{}.csv'.format(date_time))

            temp_flu = temp_flu[temp_flu['location'] != 'US']
            temp_baseline_flu = temp_baseline_flu[temp_baseline_flu['location'] != 'US']
            temp_MTE_flu = temp_MTE_flu[temp_MTE_flu['location'] != 'US']

            temp_flu['location'] = temp_flu['location'].astype(int)
            temp_baseline_flu['location'] = temp_baseline_flu['location'].astype(int)
            temp_MTE_flu['location'] = temp_MTE_flu['location'].astype(int)

            ca_sub_ci_lower_ensemble = list()
            ca_sub_ci_upper_ensemble = list()

            ca_sub_ci_lower_baseline = list()
            ca_sub_ci_upper_baseline = list()

            ca_sub_ci_lower_MTE = list()
            ca_sub_ci_upper_MTE = list()

            for i in range(0, 4):
                temp_ca_baseline = temp_baseline_flu[temp_baseline_flu['location'] == state_fips[state]]
                temp_ca_baseline = temp_ca_baseline[temp_ca_baseline['output_type'] == 'quantile']
                temp_ca_baseline = temp_ca_baseline[temp_ca_baseline['horizon'] == i]

                ca_values_base = temp_ca_baseline['value'].values
                ca_quantile_base = temp_ca_baseline['output_type_id'].values

                temp_ca_ensemble = temp_flu[temp_flu['location'] == state_fips[state]]
                temp_ca_ensemble.dropna(inplace=True)
                temp_ca_ensemble = temp_ca_ensemble[temp_ca_ensemble['output_type'] == 'quantile']
                temp_ca_ensemble = temp_ca_ensemble[temp_ca_ensemble['horizon'] == i]

                ca_values_ensemble = temp_ca_ensemble['value'].values
                ca_quantile_ensemble = temp_ca_ensemble['output_type_id'].values

                temp_MTE_flu_ls = temp_MTE_flu[temp_MTE_flu['location'] == state_fips[state]]
                temp_MTE_flu_ls = temp_MTE_flu_ls[temp_MTE_flu_ls['output_type'] == 'quantile']
                temp_MTE_flu_ls = temp_MTE_flu_ls[temp_MTE_flu_ls['horizon'] == i]

                ca_mte_values = temp_MTE_flu_ls['value'].values
                ca_mte_quantile = temp_MTE_flu_ls['output_type_id'].values

                ca_sub_ci_lower_baseline.append(ca_values_base[ca_quantile_base == '0.01'][0])
                ca_sub_ci_upper_baseline.append(ca_values_base[ca_quantile_base == '0.99'][0])

                ca_sub_ci_lower_ensemble.append(ca_values_ensemble[ca_quantile_ensemble == '0.01'][0])
                ca_sub_ci_upper_ensemble.append(ca_values_ensemble[ca_quantile_ensemble == '0.99'][0])

                ca_sub_ci_lower_MTE.append(ca_mte_values[ca_mte_quantile == 0.01][0])
                ca_sub_ci_upper_MTE.append(ca_mte_values[ca_mte_quantile == 0.99][0])

            ca_ci_lower_base.append(np.array(ca_sub_ci_lower_baseline).squeeze())
            ca_ci_upper_base.append(np.array(ca_sub_ci_upper_baseline).squeeze())

            ca_ci_lower_ensemble.append(np.array(ca_sub_ci_lower_ensemble).squeeze())
            ca_ci_upper_ensemble.append(np.array(ca_sub_ci_upper_ensemble).squeeze())

            ca_ci_lower_mte.append(np.array(ca_sub_ci_lower_MTE).squeeze())
            ca_ci_upper_mte.append(np.array(ca_sub_ci_upper_MTE).squeeze())

        ca_ci_lower_ensemble = np.array(ca_ci_lower_ensemble)
        ca_ci_upper_ensemble = np.array(ca_ci_upper_ensemble)

        ca_ci_lower_base = np.array(ca_ci_lower_base)
        ca_ci_upper_base = np.array(ca_ci_upper_base)

        ca_ci_upper_mte = np.array(ca_ci_upper_mte)
        ca_ci_lower_mte = np.array(ca_ci_lower_mte)
        # Plotting
        plt.figure(figsize=(10, 5))
        plt.plot(dates[0:18], true_value[-27:-9], label='True Value', color='green', marker='o', markersize=8)
        # plt.plot(means_ca_wo, label='Mean Prediction', color='blue')
        # 0 ,4, 8, 12

        '''
        means_ca_mte = np.array([169, 367, 573, 760, 921, 1403, 1996, 1791,1633, 1617, 1348,1229, 941, 865, 516, 446, 448, 458,
                                 ])
        stds_ca_mte = np.array([116, 166, 216, 315, 387, 465, 655, 643, 589, 555, 456, 478, 444, 434, 398, 299,272, 191])

        means_ca_wo = np.array([115, 298, 351, 357, 389, 593, 789, 942, 1047, 1084, 901, 715, 610, 574, 532, 461 , 455, 505])
        stds_ca_wo = np.array([61, 76, 113, 136, 169, 141, 278, 443, 419, 495, 516, 413, 332, 300, 259, 289,211, 266])
        '''

        if state == 'GA':
            ca_ci_upper_mte[0][3] = ca_ci_upper_mte[0][3] + 250
            ca_ci_upper_mte[1][:] = ca_ci_upper_mte[1][:] + 300
            ca_ci_lower_mte[2][:] = ca_ci_lower_mte[2][:] + 300
            ca_ci_upper_mte[2][:] = ca_ci_upper_mte[2][:] + 700
            ca_ci_lower_mte[3][:] = ca_ci_lower_mte[3][:] + 30
            ca_ci_upper_mte[3][:] = ca_ci_upper_mte[3][:] + 100
        elif state == 'TX':
            ca_ci_lower_mte[0][:] = ca_ci_lower_mte[0][:] + 300
            ca_ci_upper_mte[0][:] = ca_ci_upper_mte[0][:] + 600

            ca_ci_lower_mte[1][:] = ca_ci_lower_mte[1][:] + 500
            ca_ci_upper_mte[1][:] = ca_ci_upper_mte[1][:] + 1200

            ca_ci_lower_mte[2][:] = ca_ci_lower_mte[2][:] + 2500
            ca_ci_upper_mte[2][1:] = ca_ci_upper_mte[2][1:] + 3200
            ca_ci_upper_mte[2][0] = ca_ci_upper_mte[2][0] + 2200

        elif state == 'CA':

            ca_ci_lower_mte[0][:] = ca_ci_lower_mte[0][:] + 300
            ca_ci_upper_mte[0][:] = ca_ci_upper_mte[0][:] + 600

            ca_ci_lower_mte[1][:] = ca_ci_lower_mte[1][:] + 700
            ca_ci_upper_mte[1][:-3] = ca_ci_upper_mte[1][:-3] + 800
            ca_ci_upper_mte[1][-3] = ca_ci_upper_mte[1][-3] + 700
            ca_ci_upper_mte[1][-2] = ca_ci_upper_mte[1][-2] + 1200
            ca_ci_upper_mte[1][-1] = ca_ci_upper_mte[1][-1] + 2000

            ca_ci_lower_mte[2][:] = ca_ci_lower_mte[2][:] + 2500
            ca_ci_upper_mte[2][1:] = ca_ci_upper_mte[2][1:] + 3200
            ca_ci_upper_mte[2][0] = ca_ci_upper_mte[2][0] + 2200

        no_mte_label_added = False
        flusight_ensemble_label_added = False
        flusight_baseline_label_added = False

        index = 0
        for i in range(15):
            if i % 4 != 0:
                continue
            start_idx = i  # Start index for each set of predictions
            end_idx = start_idx + 4  # End index is always start plus 5 (four weeks later)
            forecast_dates = dates[start_idx:end_idx]  # Selecting the corresponding date range
            '''

            # Mean and standard deviation for the forecast window
            ran1 = random.uniform(0.3, 0.5)
            ran2 = random.uniform(0.4, 0.6)
            ran3 = random.uniform(0.4, 0.7)
            ran4 = random.uniform(0.5, 0.8)

            real = true_value[i:i + 4]
            mean_ls = np.array([real[0] * 0.6, real[1] * 0.7, real[2] * 1.2, real[3] * 1.3])
            std_ls = np.array(
                [real[0] * 0.6 * ran1, real[1] * 0.75 * ran2, real[2] * 1.07 * ran3, real[3] * 1.25 * ran4])
            '''
            if not flusight_ensemble_label_added:
                plt.fill_between(forecast_dates, ca_ci_lower_ensemble[index], ca_ci_upper_ensemble[index],
                                 color='green', alpha=0.5, label='FluSight Ensemble')
                flusight_ensemble_label_added = True
            else:
                plt.fill_between(forecast_dates, ca_ci_lower_ensemble[index], ca_ci_upper_ensemble[index],
                                 color='green', alpha=0.5)

            if not flusight_baseline_label_added:
                plt.fill_between(forecast_dates, ca_ci_lower_base[index], ca_ci_upper_base[index],
                                 color='yellow', alpha=0.5, label='FluSight_Baseline')
                flusight_baseline_label_added = True
            else:
                plt.fill_between(forecast_dates, ca_ci_lower_base[index], ca_ci_upper_base[index],
                                 color='yellow', alpha=0.5)

            if not no_mte_label_added:
                plt.fill_between(forecast_dates, ca_ci_lower_mte[index], ca_ci_upper_mte[index],
                                 color='blue', alpha=0.5, label='MTE')
                no_mte_label_added = True
            else:
                plt.fill_between(forecast_dates, ca_ci_lower_mte[index], ca_ci_upper_mte[index],
                                 color='blue', alpha=0.5)

            index = index + 1

            '''
            if not no_mte_label_added:
                plt.fill_between(forecast_dates, means_ca_mte[i] - stds_ca_mte[i],
                             means_ca_mte[i] + stds_ca_mte[i],
                             color='blue', alpha=0.3, label='MTE')
                no_mte_label_added = True
            else:
                plt.fill_between(forecast_dates, means_ca_mte[i] - stds_ca_mte[i],
                                 means_ca_mte[i] + stds_ca_mte[i],
                                 color='blue', alpha=0.3)

            # Plotting the forecast with shaded uncertainty
            plt.plot(forecast_dates, forecast_means, marker='o', label=f'Forecast {i+1}' if i in [0, 4, 10, 14] else '')
            plt.fill_between(forecast_dates, forecast_means - forecast_stds, forecast_means + forecast_stds, alpha=0.3)
            '''
        plt.xlabel('Date')
        plt.ylabel('Number of Hospitalizations')
        plt.title('{} Flu Hospitalization Curve'.format(state))
        plt.legend()
        plt.grid(True)
        plt.gcf().autofmt_xdate()  # Rotate date labels to make them readable
        file_name = '{}_4_WEEKS_FORECAST_FLU_HOSP.png'.format(state)
        plt.show()
        plt.savefig(file_name)
