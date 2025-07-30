import networkx as nx
import pandas as pd
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import wilcoxon, spearmanr

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
for t in range(11, 153):
    t = degree_check[(t - 4):t, t, :, :]
    ls.append(t)

tem = ls[-8:]
selected_regions = [4,9,43,32]
result = list()
starting_date = datetime.strptime('2024-11-23', '%Y-%m-%d').date()
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
    df_max.groupby(['source_state', 'target_state'])['lag'].transform('max') == df_max['lag']
].reset_index(drop=True)

# df_max.to_csv('all_pairs_first_8_weeks.csv')

case = pd.read_csv('CDC_DATA/transferred_hospital_admission_0524.csv')

source_states = ['CA','FL', 'TX', 'NY']

for src in source_states:
    tgt_ls = df_max[df_max['source_state'] == src]['target_state'].values
    for tgt in tgt_ls:
        strat_ids = df_max[(df_max['source_state'] == src) & (df_max['target_state'] == tgt)]['strat_id'].values
        for strat_id in strat_ids:
            lag = df_max[
                (df_max['source_state'] == src) & (df_max['target_state'] == tgt) & (df_max['strat_id'] == strat_id)][
                'lag'].values[0]
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

'''
start_case_index = 7
hhs_regions = {
    1: ['CT', 'ME', 'MA', 'NH', 'RI', 'VT'],
    2: ['NJ', 'NY', 'PR'],
    3: ['DC', 'DE', 'MD', 'PA', 'VA', 'WV'],
    4: ['AL', 'FL', 'GA', 'KY', 'MS', 'NC', 'SC', 'TN'],
    5: ['IL', 'IN', 'MI', 'MN', 'OH', 'WI'],
    6: ['AR', 'LA', 'NM', 'OK', 'TX'],
    7: ['IA', 'KS', 'MO', 'NE'],
    8: ['CO', 'MT', 'ND', 'SD', 'UT', 'WY'],
    9: ['AZ', 'CA', 'HI', 'NV'],
    10: ['AK', 'ID', 'OR', 'WA']
}
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


if __name__ == '__main__':

    aggregated_df = list()
    pdf_filename = "metrics_of_MTE/spatiotemporal_causal_graphs.pdf"
    with PdfPages(pdf_filename) as pdf:
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
                            source_time = current_time - lag - 1
                            G.add_edge(f"{index_state[i]}_T{current_time - source_time}",
                                       f"{index_state[j]}_T{current_time}")

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

            sorted_hhs_order = sorted(hhs_regions.keys())

            # Assign positions based on the sorted HHS order (from top to bottom)
            y_offset = 0
            for hhs_group in sorted_hhs_order:
                if hhs_group in grouped_nodes:
                    for node in grouped_nodes[hhs_group]:
                        for t_idx, t in enumerate(range(1, current_time + 1)):
                            if f"_T{t}" in node:
                                pos[node] = (t_idx * x_spacing, -y_offset)
                        y_offset += y_spacing * len(grouped_nodes[hhs_group])

            for t_idx, t in enumerate(range(1, current_time + 1)):
                y_offset = 0
                for hhs_group in sorted(grouped_nodes.keys()):  # Sort keys from 1 to 10
                    nodes = grouped_nodes[hhs_group]
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

            plt.figure(figsize=(15, 10))
            nx.draw_networkx_nodes(G, pos, node_size=300, node_color=node_colors, edgecolors="black")
            nx.draw_networkx_edges(G, pos, edge_color="blue", style="dashed", arrowsize=10)
            nx.draw_networkx_labels(
                G, pos, labels={n: n.split('_')[0] for n in G.nodes()}, font_size=8, font_color="black"
            )

            start_date = datetime.strptime('2022-04-02', "%Y-%m-%d") + timedelta(days=TIME * 7)
            target_date = start_date + timedelta(days=4 * 7)
            for t_idx, t in enumerate(range(0, current_time)):
                x = t_idx * x_spacing
                date_label = start_date + timedelta(days=t_idx * 7)
                plt.text(x, 1, date_label.strftime("%Y-%m-%d"), fontsize=12, fontweight="bold", ha="center")

            plt.title("Spatiotemporal Causal Graph with Grouped Nodes by HHS Group by {}".format(
                target_date.strftime("%Y-%m-%d")))
            plt.axis('off')

            # Save the current plot to the PDF
            pdf.savefig()
            plt.close()

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
            stat2, p_value_in_causal_strength = spearmanr(causal_in_strength_values, node_trend_value,
                                                          alternative='greater')

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
                aggregated_data.loc[node, 'causal_in_strength_p_value'] = node_data.loc[
                    f"{node}", "causal_in_strength_p_value"]
                aggregated_data.loc[node, 'causal_in_strength_correlation'] = node_data.loc[
                    f"{node}", "causal_in_strength_correlation"]

            causal_out_strength_values = aggregated_data['causal_out_strength'].values

            node_trend_value = aggregated_data['node_trend'].values

            stat3, p_value_out_causal_strength = spearmanr(causal_out_strength_values, node_trend_value,
                                                           alternative='greater')

            aggregated_data['causal_out_strength_p_value'] = p_value_out_causal_strength
            aggregated_data['causal_out_strength_correlation'] = stat3
            aggregated_data = aggregated_data.apply(pd.to_numeric, errors='coerce')
            aggregated_df.append(aggregated_data)
            aggregated_data.to_csv('metrics_of_MTE/metrics_by_{}.csv'.format(target_date.strftime("%Y-%m-%d")))
    import pickle
    with open('aggregated_df.pkl', 'wb') as f:
        pickle.dump(aggregated_df, f)
'''