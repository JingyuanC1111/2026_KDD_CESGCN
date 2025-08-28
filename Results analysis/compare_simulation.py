import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

Ten_splits = np.load('Ten_splits.npy')

numbers = pd.read_parquet('exp1-inc-E-state-weekly.parquet')
numbers['fips'] = numbers['fips'].astype(int)
strats = np.unique(numbers.cell.values)

covid_case = pd.read_csv(
    'C:\\Users\\6\\PycharmProjects\\FORECASTING_GNN\\epidemic_data\\COVID_CASE\\COVID_CASE_DATA.csv')

covid_case = covid_case[~covid_case['Abbr'].isin(['HI','AK'])]
covid_case = covid_case.reset_index(drop=True)
covid_case['Index'] = covid_case.index

index_fips_dict = covid_case['FIPS'].to_dict()
fips_state_dict = covid_case.set_index('FIPS')['Abbr'].to_dict()
index_state = dict(zip(covid_case['Index'], covid_case['Abbr']))

all_s_t = list()
for idx in range(0, len(Ten_splits)):
    degree_check = Ten_splits[idx]
    for ind in range(4):
        for source in index_state.keys():
            for target in index_state.keys():
                if degree_check[ind, source, target]:
                    temp = [source, 4 - ind, target, idx]
                    all_s_t.append(temp)

all_s_t = pd.DataFrame(all_s_t)
all_s_t.columns = ['source_state', 'lag', 'target_state','strat_id']
all_s_t = all_s_t[all_s_t['source_state'] != all_s_t['target_state']]

all_s_t["source_state"] = all_s_t["source_state"].map(index_state)
all_s_t["target_state"] = all_s_t["target_state"].map(index_state)
all_s_t = all_s_t.sort_values('source_state')
all_s_t.to_csv('all_source_target_pairs_in_8_stratigies.csv')

selected_cols = ['source_state','target_state','strat_id']
all_s_t_src_tgt = all_s_t.loc[:,selected_cols]
all_s_t_src_tgt.drop_duplicates(inplace=True)
pair_counts = all_s_t_src_tgt.value_counts(subset=["source_state", "target_state"]).reset_index(name="count")

frequent_sources = all_s_t['source_state'].value_counts() # This scenario did not filter multiple lags
# NJ: 44, PA: 32, CA: 30, FL: 18, GA: 16


df_max = all_s_t.loc[
    all_s_t.groupby(['source_state', 'target_state','strat_id'])['lag'].transform('max') == all_s_t['lag']
].reset_index(drop=True)

df_max.drop_duplicates(inplace=True)
frequent_sources_unique = df_max['source_state'].value_counts()

pair_counts_top10 = pair_counts.iloc[:10,:]








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

sim_edges = pd.read_csv('contact_matrix-state-wdeg.csv')
sim_edges['contact_duration_hr'] = np.log(sim_edges['contact_duration_hr'])

sim_edges = sim_edges.sort_values(by='contact_duration_hr',ascending=False)

all_causal_edges_simulation = pd.read_csv('all_causal_edges_simulation.csv').iloc[:,1:]

all_causal_edges_simulation['target'] = all_causal_edges_simulation['target'].map(index_fips_dict)
all_causal_edges_simulation['source'] = all_causal_edges_simulation['source'].map(index_fips_dict)

all_causal_edges_simulation['target'] = all_causal_edges_simulation['target'].map(fips_state_dict)
all_causal_edges_simulation['source'] = all_causal_edges_simulation['source'].map(fips_state_dict)


all_causal_edges_simulation = all_causal_edges_simulation[all_causal_edges_simulation['source'] != all_causal_edges_simulation['target']]
all_causal_edges_simulation = all_causal_edges_simulation.sort_values(by=['source','lag'])
all_causal_edges_simulation = all_causal_edges_simulation.iloc[:,[0,1,3,4]]

te_sum = (
    all_causal_edges_simulation.groupby(['source', 'target'], as_index=False)['TE']
      .sum()
      .rename(columns={'TE': 'TE_sum'})
)
te_sum = te_sum.sort_values(by='TE_sum',ascending=False)

state = 'PA'
df_max_sub = pair_counts[pair_counts['source_state'] == state]
sim_edges_sub = sim_edges[sim_edges['source_state'] == state]

numbers = pd.read_parquet('exp1-inc-E-state-weekly.parquet')
strategies = np.unique(numbers.cell.values)

numbers = pd.read_parquet('exp1-inc-E-state-weekly.parquet')
edges = pd.read_csv('contact_matrix-state-wdeg.csv.gz')
numbers['fips'] = numbers['fips'].astype(int)

strategies = np.unique(numbers.cell.values)

all_data = []
for rep in range(1,11):
    tau_high_no = numbers[(numbers['cell'] == 'seeding_emergingID-tau_high-surveil_no') &
                          (numbers['rep'] == rep)]
    df_sorted = tau_high_no.sort_values(by='fips')
    df_sorted["fips"] = pd.to_numeric(df_sorted["fips"])
    df_sorted["week"] = pd.to_numeric(df_sorted["week"])
    df_sorted["new_count"] = pd.to_numeric(df_sorted["new_count"])

    data_rep = df_sorted.pivot(index="fips", columns="week", values="new_count")

    all_data.append(data_rep)

print('data preparation complete')

# Stack over repeats axis (resulting shape: [nodes, 28, repeats])

data_3d = np.stack(all_data, axis=-1)
data_3d = pd.DataFrame(data_3d)
w = list(range(0,10))
numbers = pd.read_parquet('exp1-inc-E-state-weekly.parquet')

strategies = np.unique(numbers.cell.values)
tau_high_no = numbers[(numbers['cell'] == 'seeding_emergingID-tau_high-surveil_no') &
                              (numbers['rep'] == 1.0)]