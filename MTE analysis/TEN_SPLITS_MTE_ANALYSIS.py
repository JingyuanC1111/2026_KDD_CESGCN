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
# NJ: 19, PA: 16, CA: 15, FL: 11, GA: 8
pair_counts.to_csv('8_splits_identified_MTE.csv')
df_max.to_csv('max_lag_source_target_pairs.csv')

pair_counts_top10 = pair_counts.iloc[:10,:]

strats = ['seeding_importedJFK-tau_low-surveil_yes',
              'seeding_importedJFK-tau_low-surveil_no',
              'seeding_importedJFK-tau_high-surveil_yes',
              'seeding_emergingID-tau_low-surveil_no',
              'seeding_emergingID-tau_high-surveil_yes',
              'seeding_emergingID-tau_high-surveil_no',
              'seeding_importedJFK-tau_high-surveil_no',
              'seeding_emergingID-tau_low-surveil_yes']

if __name__ == '__main__':

    source_states = ['NJ','PA','CA','FL','GA']
    numbers['fips'] = numbers['fips'].map(fips_state_dict)
    for src in source_states:
        tgt_ls = df_max[df_max['source_state'] == src]['target_state'].values
        for tgt in tgt_ls:
            strat_ids = df_max[(df_max['source_state'] == src) & (df_max['target_state'] == tgt)]['strat_id'].values
            for strat_id in strat_ids:
                lag = df_max[(df_max['source_state'] == src) & (df_max['target_state'] == tgt) & (df_max['strat_id'] == strat_id)]['lag'].values[0]
                sub_number = numbers[numbers['cell'] == strats[strat_id]]
                for rep in range(1, 11):
                    plotting_data = numbers[(numbers['cell'] == strats[strat_id]) & (numbers['rep'] == rep)]

                    src_values = plotting_data[plotting_data['fips'] == src]['new_count'].values
                    tgt_values = plotting_data[plotting_data['fips'] == tgt]['new_count'].values

                    # Compute peak indices
                    src_peak_idx = np.argmax(src_values)
                    tgt_peak_idx = np.argmax(tgt_values)

                    # Create the plot
                    plt.figure(figsize=(10, 6))
                    plt.plot(src_values, label='Source State {}'.format(src), marker='o')
                    plt.plot(tgt_values, label='Target State {}'.format(tgt), marker='x')

                    # Add vertical lines for peaks
                    plt.axvline(src_peak_idx, color='blue', linestyle='--', alpha=0.6, label='Source Peak')
                    plt.axvline(tgt_peak_idx, color='orange', linestyle='--', alpha=0.6, label='Target Peak')

                    # Annotate the peaks
                    plt.annotate('Peak: {}'.format(src_peak_idx),
                                 xy=(src_peak_idx, src_values[src_peak_idx]),
                                 xytext=(src_peak_idx, src_values[src_peak_idx] + max(src_values) * 0.05),
                                 arrowprops=dict(arrowstyle='->', color='blue'),
                                 color='blue')

                    plt.annotate('Peak: {}'.format(tgt_peak_idx),
                                 xy=(tgt_peak_idx, tgt_values[tgt_peak_idx]),
                                 xytext=(tgt_peak_idx, tgt_values[tgt_peak_idx] + max(tgt_values) * 0.05),
                                 arrowprops=dict(arrowstyle='->', color='orange'),
                                 color='orange')

                    # Labels and legend
                    plt.xlabel("Time Index")
                    plt.ylabel("New Count")
                    plt.title("{} leads {} By {} in repeat {} of stratification {}".format(src, tgt, lag, rep,
                                                                                           strats[strat_id]))
                    plt.legend()
                    plt.grid(True)

                    # Save the figure
                    filename = '10splits_MTE_validation/{} leads {} By {} in repeat {} of stratification {}.png'.format(
                        src, tgt, lag, rep, strats[strat_id]
                    )
                    plt.savefig(filename)
                    plt.close()

