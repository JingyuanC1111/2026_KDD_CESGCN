import pickle
import pandas as pd
import numpy as np
import torch

all_matrix = pd.read_csv('contact_matrix-state-wdeg.csv')
all_matrix['contact_duration_hr'] = np.log(all_matrix['contact_duration_hr'] + 1)
indices = pd.read_csv('state_hhs_map.csv',header=None)
index_dict = indices.set_index(0)[1].to_dict()
numbers = pd.read_parquet('exp1-inc-E-state-weekly.parquet')
numbers['fips'] = numbers['fips'].astype(int)


if __name__ == '__main__':
    res_list = list()
    MTE_static_matrices = torch.zeros((28, 28, 49, 49))
    time = 28
    with open('CDC_DATA/MTE_repeat_1_to_10_from_0_to_28', 'rb') as file:
        results = pickle.load(file)
        for target in results.keys():
            selected_sources = results[target]['selected_vars_sources']
            selected_sources_te = results[target]['selected_sources_te']
            selected_target_past = results[target]['selected_vars_target']
            if len(selected_sources) != 0:
                for idx, source in enumerate(selected_sources):
                    source_process = source[0]
                    source_process_lag = source[1]
                    source_te = selected_sources_te[idx]
                    # if source_process_lag == 0:
                    #    MTE_static_matrices[
                    #        2, source_process, target] = source_te
                    # else:
                    current_index = time - 1 # 23
                    index_ls = list(range(1, source_process_lag + 1))  # lags start from 1
                    selected_index = [current_index - ind for ind in index_ls] # 22
                    MTE_static_matrices[
                        selected_index, current_index, source_process, target] = source_te  # fix this one, should be all the stuff
            if len(selected_target_past) != 0:
                for idx2, target_past in enumerate(selected_target_past):
                    target_process = target_past[0]
                    target_process_lag = target_past[1]
                    # if target_process_lag == 0:
                    #    MTE_static_matrices[
                    #        2, target_process, target] = 1
                    # else:
                    current_index = time - 1
                    index_ls = list(range(1, target_process_lag + 1))  # lags start from 1
                    selected_index = [current_index - ind for ind in index_ls]
                    MTE_static_matrices[selected_index, current_index, target_process, target_process] = 1
                    # MTE_static_matrices[selected_index, target_process, target] = 1

    MTE_static_matrices = MTE_static_matrices.numpy()
    MTE_static_matrices = MTE_static_matrices[-5:,-1,:,:]
    print('True')
    np.save('Simulation_repeat_from_1_to_10_all_data.npy', MTE_static_matrices)


import networkx as nx
import pandas as pd
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import wilcoxon, spearmanr
from collections import defaultdict, Counter

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

degree_check = np.load('Simulation_repeat_from_1_to_10_all_data.npy')


covid_case = pd.read_csv(
    'C:\\Users\\6\\PycharmProjects\\FORECASTING_GNN\\epidemic_data\\COVID_CASE\\COVID_CASE_DATA.csv')

covid_case = covid_case[~covid_case['Abbr'].isin(['HI','AK'])]
covid_case = covid_case.reset_index(drop=True)

index_fips_dict = covid_case['FIPS'].to_dict()
fips_state_dict = covid_case.set_index('FIPS')['Abbr'].to_dict()
index_state = dict(zip(covid_case['index'], covid_case['Abbr']))


count_source_target = list()
temp_source_target = list()
cur_MTE = degree_check
cur_MTE_agg = np.sum(cur_MTE, axis=0)
for i in range(len(cur_MTE_agg)):
    for j in range(len(cur_MTE_agg)):
        if i == j:
            continue
        else:
            if cur_MTE_agg[i][j]:
                temp_source_target.append([i, j])
count_source_target.extend(temp_source_target)

df = pd.DataFrame(count_source_target)
frequency = df.value_counts().reset_index(name='count')
frequency.columns = ['source','target','count']
frequency['target'] = frequency['target'].map(index_fips_dict)
frequency['source'] = frequency['source'].map(index_fips_dict)

frequency['target'] = frequency['target'].map(fips_state_dict)
frequency['source'] = frequency['source'].map(fips_state_dict)
# frequency.to_csv('source_target_frequency_table.csv', mode='a')
# frequency.to_csv('source_target_frequency_table_simulation.csv', mode='a')

sim_edges = pd.read_csv('contact_matrix-state-wdeg.csv')
sim_edges['contact_duration_hr'] = np.log(sim_edges['contact_duration_hr'])

sim_edges = sim_edges.sort_values(by='contact_duration_hr',ascending=False)
all_causal_edges_simulation = pd.read_csv('all_causal_edges_simulation.csv').iloc[:,1:]

all_causal_edges_simulation['target'] = all_causal_edges_simulation['target'].map(index_fips_dict)
all_causal_edges_simulation['source'] = all_causal_edges_simulation['source'].map(index_fips_dict)

all_causal_edges_simulation['target'] = all_causal_edges_simulation['target'].map(fips_state_dict)
all_causal_edges_simulation['source'] = all_causal_edges_simulation['source'].map(fips_state_dict)

all_causal_edges_simulation_ca = all_causal_edges_simulation[all_causal_edges_simulation['source'] == 'CA']


all_causal_edges_simulation = all_causal_edges_simulation[frequency['source'] != all_causal_edges_simulation['target']]
te_sum = (
    all_causal_edges_simulation.groupby(['source', 'target'], as_index=False)['TE']
      .sum()
      .rename(columns={'TE': 'TE_sum'})
)



te_sum = te_sum.sort_values(by='TE_sum',ascending=False)

te_sum_FL = te_sum[te_sum['source'] == 'CA']
sim_edges_FL = sim_edges[sim_edges['source_state'] == 'CA']

numbers = pd.read_parquet('exp1-inc-E-state-weekly.parquet')


