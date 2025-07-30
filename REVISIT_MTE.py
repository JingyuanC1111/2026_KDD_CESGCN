import networkx as nx
import pandas as pd
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import wilcoxon, spearmanr
import matplotlib.dates as mdates

degree_check = np.load('MTE_matrices_flu_hosp.npy')

covid_case = pd.read_csv(
    'C:\\Users\\6\\PycharmProjects\\FORECASTING_GNN\\epidemic_data\\COVID_CASE\\COVID_CASE_DATA.csv')

covid_case = covid_case.reset_index().rename(columns={'index': 'Index'})
index_state = dict(zip(covid_case['Index'], covid_case['Abbr']))
index_state[51] = 'PR'

hhs_map = pd.read_csv('us_subplot_grid.csv')
state_hhs = dict(zip(hhs_map['State'], hhs_map['HHS']))

# Reverse the key-value pairs
state_hhs_reverse = {}
for key, value in state_hhs.items():
    state_hhs_reverse.setdefault(value, []).append(key)

# Replace with your actual data
ls = list()
for t in range(11, 172):
    t = degree_check[(t - 4):t, t, :, :]
    ls.append(t)

tem = ls[-27:]
selected_regions = [4,9,43,32]
result = list()

case = pd.read_csv('CDC_DATA/transferred_hospital_admission_0524.csv')
current_season_dates = case['date'].values[-27:]
starting_date = datetime.strptime(current_season_dates[0], '%Y-%m-%d').date()

for ls in tem:
    subresult = []
    for sr in selected_regions:
        cur = ls[:, sr, :]
        lags = 4 - np.where(cur)[0]
        index_target_state = np.where(cur)[1]

        for ind in range(len(lags)):
            temp = [starting_date.strftime('%Y-%m-%d'),
                    sr,
                    lags[ind],
                    index_target_state[ind]]
            subresult.append(temp)

    result.append(subresult)

    # ➜ move to the next epidemiological week
    starting_date += timedelta(days=7)

flattened = [entry for sublist in result for entry in sublist]
# Create DataFrame
df = pd.DataFrame(flattened, columns=["date", "source_state", "lag", "target_state"])

df["source_state"] = df["source_state"].map(index_state)
df["target_state"] = df["target_state"].map(index_state)

df['lag'] = pd.to_numeric(df['lag'], errors='coerce')
df_max = df[df['source_state'] != df['target_state']]

# ------------------------------------------------------------------
# KEEP max-lag row for every (source_state , target_state) pair
# ------------------------------------------------------------------
df_max = df_max.loc[
    df_max.groupby(['source_state', 'target_state','date'])['lag'].transform('max') == df_max['lag']
].reset_index(drop=True)








if __name__ == '__main__':
    case = pd.read_csv('CDC_DATA/transferred_hospital_admission_0524.csv')
    covid_case = pd.read_csv(
        'C:\\Users\\6\\PycharmProjects\\FORECASTING_GNN\\epidemic_data\\COVID_CASE\\COVID_CASE_DATA.csv')

    covid_case = covid_case.reset_index().rename(columns={'index': 'Index'})
    index_state = dict(zip(covid_case['Index'], covid_case['Abbr']))
    index_state[51] = 'PR'

    source_states = ['CA','FL', 'TX', 'NY']

    new_columns = ['date']
    new_columns.extend(index_state.values())
    case.columns = new_columns

    current_season_dates = pd.to_datetime(current_season_dates)
    df_max['date'] = pd.to_datetime(df_max['date'])

    current_season_dates = pd.to_datetime(current_season_dates)
    case['date'] = pd.to_datetime(case['date'])

    for src in source_states:
        for date in current_season_dates:
            sub_df = df_max[(df_max['source_state'] == src) & (df_max['date'] == date)]
            if sub_df.empty:
                continue

            tgt_state_ls = sub_df['target_state'].values
            for tgt in tgt_state_ls:
                cur_lag = sub_df[sub_df['target_state'] == tgt]['lag'].values[0]

                # Align series by current_season_dates
                src_values = case.loc[case['date'].isin(current_season_dates), src].values
                tgt_values = case.loc[case['date'].isin(current_season_dates), tgt].values

                if len(src_values) != len(current_season_dates) or len(tgt_values) != len(current_season_dates):
                    print(f"Skipping {src}→{tgt} due to length mismatch.")
                    continue

                # ------------------------------
                # 1. Global Peaks (full curve)
                # ------------------------------
                src_global_peak_idx = np.argmax(src_values)
                tgt_global_peak_idx = np.argmax(tgt_values)
                src_global_peak_date = current_season_dates[src_global_peak_idx]
                tgt_global_peak_date = current_season_dates[tgt_global_peak_idx]

                # ------------------------------
                # 2. Local Peaks Before Cutoff
                # ------------------------------
                cutoff_date = pd.to_datetime(date)
                pre_mask = current_season_dates <= cutoff_date

                src_local_peak_date = tgt_local_peak_date = None
                if pre_mask.sum() > 0:
                    pre_dates = current_season_dates[pre_mask]
                    src_pre_values = src_values[pre_mask]
                    tgt_pre_values = tgt_values[pre_mask]

                    src_local_peak_idx = np.argmax(src_pre_values)
                    tgt_local_peak_idx = np.argmax(tgt_pre_values)

                    src_local_peak_date = pre_dates[src_local_peak_idx]
                    tgt_local_peak_date = pre_dates[tgt_local_peak_idx]

                # ------------------------------
                # Plotting
                # ------------------------------
                plt.figure(figsize=(12, 6))
                plt.plot(current_season_dates, src_values, label=f'Source State {src}', marker='o')
                plt.plot(current_season_dates, tgt_values, label=f'Target State {tgt}', marker='x')

                # Global peak vertical lines
                plt.axvline(src_global_peak_date, color='blue', linestyle='--', alpha=0.6, label='Source Peak')
                plt.axvline(tgt_global_peak_date, color='orange', linestyle='--', alpha=0.6, label='Target Peak')

                # Local peak before date (dashed-dot)
                if src_local_peak_date:
                    plt.axvline(src_local_peak_date, color='blue', linestyle='-.', alpha=0.6,
                                label='Source Peak before {}'.format(date.strftime('%Y-%m-%d')))
                    plt.annotate('Source Peak by {}'.format(date.strftime('%Y-%m-%d')),
                                 xy=(src_local_peak_date, src_pre_values[src_local_peak_idx]),
                                 xytext=(
                                 src_local_peak_date, src_pre_values[src_local_peak_idx] + max(src_values) * 0.05),
                                 arrowprops=dict(arrowstyle='->', color='blue'),
                                 color='blue')

                if tgt_local_peak_date:
                    plt.axvline(tgt_local_peak_date, color='orange', linestyle='-.', alpha=0.6,
                                label='Target Peak before {}'.format(date.strftime('%Y-%m-%d')))
                    plt.annotate('Target Peak by {}'.format(date.strftime('%Y-%m-%d')),
                                 xy=(tgt_local_peak_date, tgt_pre_values[tgt_local_peak_idx]),
                                 xytext=(
                                 tgt_local_peak_date, tgt_pre_values[tgt_local_peak_idx] + max(tgt_values) * 0.05),
                                 arrowprops=dict(arrowstyle='->', color='orange'),
                                 color='orange')

                # Annotations for global peaks
                plt.annotate(f'Source Global Peak: {src_global_peak_idx}',
                             xy=(src_global_peak_date, src_values[src_global_peak_idx]),
                             xytext=(src_global_peak_date, src_values[src_global_peak_idx] + max(src_values) * 0.05),
                             arrowprops=dict(arrowstyle='->', color='blue'),
                             color='blue')

                plt.annotate(f'Target Global Peak: {tgt_global_peak_idx}',
                             xy=(tgt_global_peak_date, tgt_values[tgt_global_peak_idx]),
                             xytext=(tgt_global_peak_date, tgt_values[tgt_global_peak_idx] + max(tgt_values) * 0.05),
                             arrowprops=dict(arrowstyle='->', color='orange'),
                             color='orange')

                # Format x-axis
                ax = plt.gca()
                ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                plt.xticks(rotation=45)

                # Labels and title
                plt.xlabel("Date Time")
                plt.ylabel("Hospitalization Number")
                plt.title(f"{src} leads {tgt} By {cur_lag} by {cutoff_date.strftime('%Y-%m-%d')}")

                # Clean legend (remove duplicate labels)
                handles, labels = ax.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                plt.legend(by_label.values(), by_label.keys())

                plt.grid(True)
                plt.tight_layout()
                # Save the figure
                filename = 'Revisit_MTE/{} leads {} with a lag of {} by {}'.format(
                    src, tgt, cur_lag, cutoff_date.strftime('%Y-%m-%d')
                )
                plt.savefig(filename)
                plt.close()