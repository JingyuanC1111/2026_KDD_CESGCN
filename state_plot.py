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

covid_case= covid_case.reset_index().rename(columns={'index': 'Index'})
index_state = dict(zip(covid_case['Index'], covid_case['Abbr']))
index_state[51] = 'PR'
count_source_target = list()
temp_source_target = list()
minus_4 = len(degree_check) - 4
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
frequency['target'] = frequency['target'].map(index_state)
frequency['source'] = frequency['source'].map(index_state)
# frequency.to_csv('source_target_frequency_table.csv', mode='a')
frequency.to_csv('source_target_frequency_table_simulation.csv', mode='a')

sim_edges = pd.read_csv()




lag_df = list()
for index in range(7, len(degree_check)):
    temp_source_target = list()
    minus_4 = index - 4
    cur_MTE = degree_check[minus_4:index, index, :, :]
    for lag in range(0, 4):
        cur_lag = 4 - lag
        for i in range(52):
            for j in range(52):
                if i == j:
                    continue
                else:
                    if cur_MTE[lag, i, j]:
                        date = datetime.strptime('2022-02-12', "%Y-%m-%d") + timedelta(days= index * 7)
                        lag_df.append([i, j, cur_lag, date.strftime("%Y-%m-%d")])
lag_df = pd.DataFrame(lag_df)
lag_df.columns = ['source','target','lag','date']
lag_df['source'] = lag_df['source'].map(index_state)
lag_df['target'] = lag_df['target'].map(index_state)
ls = list()
hhs_map = pd.read_csv('us_subplot_grid.csv')
state_hhs = dict(zip(hhs_map['State'], hhs_map['HHS']))
start_case_index = 7
hhs_colormap = {
    10: 'red',
    1: 'blue',
    8: 'green',
    5: 'orange',
    2: 'purple',
    9: 'yellow',
    7: 'pink',
    3: 'cyan',
    4: 'brown',
    6: 'gray'
}
# Reverse the key-value pairs
state_hhs_reverse = {}
for key, value in state_hhs.items():
    state_hhs_reverse.setdefault(value, []).append(key)
for t in range(11, 153):
    t = degree_check[t - 4:t, t, :, :]
    ls.append(t)
aggregated_df = list()
for TIME in range(len(ls)):
    data = ls[TIME]  # Replace with real data
    lags, num_nodes, _ = data.shape
    G = nx.DiGraph()  # Create a directed graph

    current_time = 5

    # Add nodes and edges based on the causal relationships
    for t in range(1, current_time + 1):
        for n in range(num_nodes):
            G.add_node(f"{index_state[n]}_T{t}")

    for lag in range(lags):
        for i in range(num_nodes):
            for j in range(num_nodes):
                if data[lag, i, j] != 0:
                    for k in range(1, lag + 2):
                        source_time = current_time - k
                        if source_time > 0:
                            G.add_edge(f"{index_state[i]}_T{source_time}", f"{index_state[j]}_T{current_time}")

    # Define positions and colors
    pos = {}
    x_spacing = 2
    y_spacing = 1

    grouped_nodes = {hhs_group: [] for hhs_group in hhs_colormap.keys()}
    for node in G.nodes():
        state_code = node.split('_')[0]
        hhs_group = state_hhs.get(state_code, None)
        if hhs_group is not None:
            grouped_nodes[hhs_group].append(node)

    for t_idx, t in enumerate(range(1, current_time + 1)):
        y_offset = 0
        for hhs_group, nodes in grouped_nodes.items():
            for node in nodes:
                if f"_T{t}" in node:
                    pos[node] = (t_idx * x_spacing, -y_offset)
                    y_offset += y_spacing

    node_colors = []
    for node in G.nodes():
        state_code = node.split('_')[0]
        hhs_group = state_hhs.get(state_code, None)
        color = hhs_colormap.get(hhs_group, 'black')
        node_colors.append(color)

    # plt.figure(figsize=(15, 10))
    #nx.draw_networkx_nodes(G, pos, node_size=300, node_color=node_colors, edgecolors="black")
    #nx.draw_networkx_edges(G, pos, edge_color="blue", style="dashed", arrowsize=10)
    #nx.draw_networkx_labels(
    #    G, pos, labels={n: n.split('_')[0] for n in G.nodes()}, font_size=8, font_color="black"
    #)

    start_date = datetime.strptime('2022-04-02', "%Y-%m-%d") + timedelta(days=TIME * 7)
    target_date = start_date + timedelta(days=4 * 7)
    '''
    for t_idx, t in enumerate(range(0, current_time)):
        x = t_idx * x_spacing
        date_label = start_date + timedelta(days=t_idx * 7)
        plt.text(x, 1, date_label.strftime("%Y-%m-%d"), fontsize=12, fontweight="bold", ha="center")
    '''
    # plt.title(
        # "Spatiotemporal Causal Graph with Grouped Nodes by HHS Group by {}".format(target_date.strftime("%Y-%m-%d")))
    #plt.axis('off')

    # Save the current plot to the PDF
    # pdf.savefig()
    # plt.show()
    # plt.close()

    # Calculate additional metrics
    causal_in_strength = {node: 0 for node in G.nodes()}
    causal_out_strength = {node: 0 for node in G.nodes()}

    for lag in range(lags):
        for i in range(num_nodes):
            for j in range(num_nodes):
                if data[lag, i, j] != 0:
                    for k in range(1, lag + 2):
                        source_time = current_time - k
                        if source_time > 0:
                            source_node = f"{index_state[i]}_T{source_time}"
                            target_node = f"{index_state[j]}_T{current_time}"
                            weight = data[lag, i, j]
                            if i != j:
                                causal_in_strength[target_node] += weight
                                causal_out_strength[source_node] += weight

    node_trend = {}
    for i in range(num_nodes):
        source_case = case.iloc[start_case_index, i + 1]
        end_case_index = start_case_index + 4
        target_case = case.iloc[end_case_index, i + 1]
        node_trend[f"{index_state[i]}"] = (target_case - source_case) / 4

    start_case_index = start_case_index + 1

    temporal_reachability = {node: len(nx.descendants(G, node)) / len(G.nodes) for node in G.nodes()}

    # Aggregate metrics
    metrics = {
        'in_degree': {node: G.in_degree(node) for node in G.nodes()},
        'out_degree': {node: G.out_degree(node) for node in G.nodes()},
        'closeness_centrality': nx.closeness_centrality(G),
        'betweenness_centrality': nx.betweenness_centrality(G),
        'density': nx.density(G),
        'causal_in_strength': causal_in_strength,
        'causal_out_strength': causal_out_strength,
        'temporal_reachability': temporal_reachability,
        'node_trend': node_trend,
    }
    # Display metrics
    metrics_df = pd.DataFrame(metrics)
    in_degree = metrics_df['in_degree'].values[52 * 4:52 * 5]
    causal_in_strength_values = metrics_df['causal_in_strength'].values[52 * 4:52 * 5]

    node_trend_value = metrics_df['node_trend'].values[52 * 5:]

    stat1, p_value_in_degree = spearmanr(in_degree, node_trend_value, alternative='greater')
    stat2, p_value_in_causal_strength = spearmanr(causal_in_strength_values, node_trend_value, alternative='greater')

    metrics_df['causal_in_degree_p_value'] = p_value_in_degree
    metrics_df['causal_in_degree_correlation'] = stat1
    metrics_df['causal_in_strength_p_value'] = p_value_in_causal_strength
    metrics_df['causal_in_strength_correlation'] = stat2
    # metrics_df.to_csv('metrics_of_MTE_pruned_by_5/metrics_by_{}.csv'.format(target_date.strftime("%Y-%m-%d")))

    # Extract unique base node names without timestamps
    base_nodes = list(set(index.split("_")[0] for index in metrics_df.index))

    # Create a new dataframe to hold the aggregated data
    aggregated_data = pd.DataFrame(index=base_nodes)

    # Aggregate the values as per instructions
    for node in base_nodes:
        # Filter data for the current node across all timestamps
        node_data = metrics_df.filter(like=node, axis=0)

        # Handle in-degree: use the value from T5
        aggregated_data.loc[node, "in_degree"] = node_data.loc[f"{node}_T5", "in_degree"]

        # Handle out-degree: sum values across T1 to T4
        aggregated_data.loc[node, "out_degree"] = node_data.loc[
            [f"{node}_T{i}" for i in range(1, 5)], "out_degree"].sum()

        # Handle closeness and betweenness centrality: use T5 values
        aggregated_data.loc[node, "closeness_centrality"] = node_data.loc[f"{node}_T5", "closeness_centrality"]
        aggregated_data.loc[node, "betweenness_centrality"] = node_data.loc[f"{node}_T5", "betweenness_centrality"]

        # Handle causal in strength: use T5 values
        aggregated_data.loc[node, "causal_in_strength"] = node_data.loc[f"{node}_T5", "causal_in_strength"]

        # Handle causal out strength: sum values across T1 to T4
        aggregated_data.loc[node, "causal_out_strength"] = node_data.loc[
            [f"{node}_T{i}" for i in range(1, 5)], "causal_out_strength"].sum()

        # Handle temporal reachability: sum values across T1 to T4
        aggregated_data.loc[node, "temporal_reachability"] = node_data.loc[
            [f"{node}_T{i}" for i in range(1, 5)], "temporal_reachability"].sum()

        # Handle node trend: use T5 value
        aggregated_data.loc[node, "node_trend"] = node_data.loc[f"{node}", "node_trend"]

        # Keep density unchanged from T5
        aggregated_data.loc[node, "density"] = node_data.loc[f"{node}_T5", "density"]

        aggregated_data.loc[node, 'causal_in_degree_p_value'] = node_data.loc[f"{node}", "causal_in_degree_p_value"]
        aggregated_data.loc[node, 'causal_in_degree_correlation'] = node_data.loc[
            f"{node}", "causal_in_degree_correlation"]
        aggregated_data.loc[node, 'causal_in_strength_p_value'] = node_data.loc[f"{node}", "causal_in_strength_p_value"]
        aggregated_data.loc[node, 'causal_in_strength_correlation'] = node_data.loc[
            f"{node}", "causal_in_strength_correlation"]

    causal_out_strength_values = aggregated_data['causal_out_strength'].values

    node_trend_value = aggregated_data['node_trend'].values

    stat3, p_value_out_causal_strength = spearmanr(causal_out_strength_values, node_trend_value, alternative='greater')

    aggregated_data['causal_out_strength_p_value'] = p_value_out_causal_strength
    aggregated_data['causal_out_strength_correlation'] = stat3

    aggregated_df.append(aggregated_data)

case = pd.read_csv('CDC_DATA/transferred_hospital_admission_0111_w_US.csv')

US_cases = case['US'].values[11:]
density_ls = list()
data_ls = list()
outdegree_ls = list()
for index, df in enumerate(aggregated_df):
    density_ls.append(df['density'].values[0])
    outdegree_ls.append(sum(df['out_degree'].values))
    start_date = datetime.strptime('2022-04-30', "%Y-%m-%d") + timedelta(days=index * 7)
    data_ls.append(start_date)

state_index = {value: key for key, value in index_state.items()}
# Take Florida as an example, we analyze the correspondence of lead/lag for other states
target_case = case.iloc[:,state_index['TX']+1].values[11:]
source_case = case.iloc[:,state_index['FL']+1].values[11:]

# Sample trajectory data in DataFrame format

trajectory_df = lag_df[(lag_df['source'] == 'FL') & (lag_df['target'] == 'TX')]
fig, ax = plt.subplots(figsize=(12, 6))

# Plot TX and FL case numbers
ax.plot(data_ls, target_case, label='TX Hospitalizations', color='blue', linewidth=2, marker='o')
ax.plot(data_ls, source_case, label='FL Hospitalizations', color='red', linewidth=2, marker='s')

# Add arrows for trajectory relationships from the DataFrame
for _, row in trajectory_df.iterrows():
    source = row['source']
    target = row['target']
    lag = row['lag']
    target_date_str = row['date']  # Keep target_date as string

    # Convert target_date to datetime for calculations
    target_date = pd.to_datetime(target_date_str, format='%Y-%m-%d')

    # Calculate the source date based on the lag
    source_date = target_date - pd.Timedelta(days=lag)

    # Convert lists of string dates to datetime for accurate comparison
    data_ls_dt = [pd.to_datetime(date, format='%Y-%m-%d') for date in data_ls]

    # Check if source_date and target_date exist in the dataset
    if source_date in data_ls_dt and target_date in data_ls_dt:
        source_idx = data_ls_dt.index(source_date)
        target_idx = data_ls_dt.index(target_date)

        source_value = source_case[source_idx]
        target_value = target_case[target_idx]

        # Example output for verification
        if source_date in data_ls_dt and target_date in data_ls_dt:
            print(f"Plotting arrow: {source_date} -> {target_date}")
        # Add an arrow showing the lead-lag relationship
        ax.annotate(
            f'{source} leads {target}\nlag={lag} days',
            xy=(target_date, target_value),
            xytext=(target_date, target_value + 100),  # Offset the text position
            arrowprops=dict(arrowstyle="->", color='black', lw=1.5),
            fontsize=10, color='black',
            ha='center', va='bottom'
        )

        # Annotate with text to explain the relationship
        ax.text(
            source_date, source_value,
            f'{source} leads {target}\nlag={lag} days',
            fontsize=10, color='black'
        )

# Properly scale the Y-axis for better visibility of arrows
y_min = min(min(source_case), min(target_case))
y_max = max(max(source_case), max(target_case))
padding = (y_max - y_min) * 0.1  # Add 10% padding

ax.set_ylim(y_min - padding, y_max + padding)

# Formatting x-axis to show dates nicely
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.xticks(rotation=45)

# Add labels and legend
ax.set_xlabel('Date')
ax.set_ylabel('Case Numbers')
ax.set_title('Case Numbers with Trajectory Relationships')
ax.legend()

# Improve grid appearance
ax.grid(which='major', linestyle='--', linewidth=0.5, alpha=0.7)

# Show the plot
plt.tight_layout()
plt.show()