import networkx as nx
import pandas as pd
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import wilcoxon, spearmanr

degree_check = np.load('Simulation_repeat_from_1_to_10_all_data.npy')

covid_case = pd.read_csv(
    'C:\\Users\\6\\PycharmProjects\\FORECASTING_GNN\\epidemic_data\\COVID_CASE\\COVID_CASE_DATA.csv')

covid_case = covid_case[~covid_case['Abbr'].isin(['HI','AK'])]
covid_case = covid_case.reset_index(drop=True)
covid_case['index'] = covid_case.index
index_fips_dict = covid_case['FIPS'].to_dict()
fips_state_dict = covid_case.set_index('FIPS')['Abbr'].to_dict()
index_state = dict(zip(covid_case['index'], covid_case['Abbr']))


hhs_map = pd.read_csv('us_subplot_grid.csv')
state_hhs = dict(zip(hhs_map['State'], hhs_map['HHS']))

# Reverse the key-value pairs
state_hhs_reverse = {}
for key, value in state_hhs.items():
    state_hhs_reverse.setdefault(value, []).append(key)

# Replace with your actual data
ls = list()
ls.append(degree_check)
case = pd.read_csv('CDC_DATA/transferred_hospital_admission_0111.csv')
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

    list_of_causal_rel = list()

    aggregated_df = list()
    pdf_filename = "metrics_of_MTE/Simulations_MTE_all_data.pdf"
    with PdfPages(pdf_filename) as pdf:
        data = ls[0]  # Replace with real data
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
                        temp = list([i, j, 28, lag+1, data[lag, i, j]])
                        list_of_causal_rel.append(temp)
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

        start_date = datetime.strptime('2022-04-02', "%Y-%m-%d") + timedelta(days=1 * 1)
        target_date = start_date + timedelta(days=4 * 1)
        for t_idx, t in enumerate(range(0, current_time)):
            x = t_idx * x_spacing
            date_label = start_date + timedelta(days=t_idx * 1)
            plt.text(x, 1, date_label.strftime("%Y-%m-%d"), fontsize=12, fontweight="bold", ha="center")

        plt.title("Spatiotemporal Causal Graph with Grouped Nodes by HHS Group by {}".format(
            target_date.strftime("%Y-%m-%d")))
        plt.axis('off')

        # Save the current plot to the PDF
        pdf.savefig()
        plt.close()

    all_causal_edges_df = pd.DataFrame(list_of_causal_rel)
    all_causal_edges_df.columns = ['source','target','time','lag','TE']


    all_causal_edges_df.to_csv('all_causal_edges_simulation_all_data.csv')
