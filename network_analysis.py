import networkx as nx
import pandas as pd
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import wilcoxon, spearmanr
from collections import defaultdict, Counter
import pickle
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import matplotlib.ticker as ticker


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

hhs_colors = {
    1: '#8c7853',  # Brown
    2: '#b63b8a',  # Pink
    3: '#213970',  # Navy blue
    4: '#d62f2f',  # Red
    5: '#2a6d8f',  # Blue
    6: '#56712e',  # Green
    7: '#e8aa2b',  # Yellow
    8: '#6c5e9c',  # Purple
    9: '#aa3e3e',  # Dark red
    10: '#2f77b4'  # Light blue
}


def state_plots(source_state, target_state, lag_df, data_ls, case):
    target_case = case.iloc[:, state_index[target_state] + 1].values[11:]
    source_case = case.iloc[:, state_index[source_state] + 1].values[11:]

    # Sample trajectory data in DataFrame format

    trajectory_df = lag_df[(lag_df['source'] == source_state) & (lag_df['target'] == target_state)]

    # Convert all dates to string format for lookup
    data_ls_str = [date.strftime('%Y-%m-%d') for date in data_ls]

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot TX and FL case numbers
    ax.plot(data_ls, target_case, label='{} Hospitalizations'.format(target_state), color='orange', linewidth=2,
            marker='o')
    ax.plot(data_ls, source_case, label='{} Hospitalizations'.format(source_state), color='royalblue', linewidth=2,
            marker='s')

    # Debug: Print available dates for comparison
    print("Available dates:", data_ls_str)

    # Add arrows for trajectory relationships from the DataFrame
    for _, row in trajectory_df.iterrows():
        source = row['source']
        target = row['target']
        lag = row['lag']
        target_date_str = row['date']  # Target date as string

        # Convert target date and calculate source date
        target_date_str = pd.to_datetime(target_date_str).strftime('%Y-%m-%d')
        source_date_str = (pd.to_datetime(target_date_str) - pd.Timedelta(days=lag * 7)).strftime('%Y-%m-%d')

        # Debug: Print source and target dates
        print(f"Checking source: {source_date_str}, target: {target_date_str}")

        if source_date_str in data_ls_str and target_date_str in data_ls_str:
            source_idx = data_ls_str.index(source_date_str)
            target_idx = data_ls_str.index(target_date_str)

            source_value = source_case[source_idx]
            target_value = target_case[target_idx]

            print(f"Plotting arrow: {source} leads {target} on {target_date_str}: {source_value} -> {target_value}")

            # Convert back to datetime for plotting
            source_date_plot = pd.to_datetime(source_date_str)
            target_date_plot = pd.to_datetime(target_date_str)

            # Add an arrow showing the lead-lag relationship
            ax.annotate(
                '',
                xy=(target_date_plot, target_value),
                xytext=(source_date_plot, source_value),
                arrowprops=dict(arrowstyle="->", color='black', lw=2),
            )

            # Annotate with text to explain the relationship
            ax.text(
                target_date_plot, target_value + 50,
                f'{source} leads {target} {lag} weeks',
                fontsize=10, color='black',
                ha='center', va='bottom'
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
    ax.set_ylabel('Hospitalization Numbers')
    ax.set_title('Hospitalization Numbers with Trajectory Relationships')
    ax.legend()

    # Improve grid appearance
    ax.grid(which='major', linestyle='--', linewidth=0.5, alpha=0.7)

    # Show the plot
    plt.tight_layout()
    plt.show()


def topk_state_plots(source_state, target_state_ls, lag_df, data_ls, case):
    source_case = case.iloc[:, state_index[source_state] + 1].values[11:]
    data_ls_str = [date.strftime('%Y-%m-%d') for date in data_ls]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data_ls, source_case, label='{} Hospitalizations'.format(source_state), color='red', linewidth=2,
            marker='s')
    colors = ['#ff7f0e', '#2ca02c', '#9467bd', '#8c564b', '#e377c2']
    all_target_case_no = list()
    for i, target_state in enumerate(target_state_ls):
        target_case = case.iloc[:, state_index[target_state] + 1].values[11:]
        all_target_case_no.extend(target_case)
        # Sample trajectory data in DataFrame format
        trajectory_df = lag_df[(lag_df['source'] == source_state) & (lag_df['target'] == target_state)]
        # Convert all dates to string format for lookup
        ax.plot(data_ls, target_case, label='{} Hospitalizations'.format(target_state), color=colors[i], linewidth=2,
                marker='o')

        # Debug: Print available dates for comparison
        # Add arrows for trajectory relationships from the DataFrame
        for _, row in trajectory_df.iterrows():
            source = row['source']
            target = row['target']
            lag = row['lag']
            target_date_str = row['date']  # Target date as string

            # Convert target date and calculate source date
            target_date_str = pd.to_datetime(target_date_str).strftime('%Y-%m-%d')
            source_date_str = (pd.to_datetime(target_date_str) - pd.Timedelta(days=lag * 7)).strftime('%Y-%m-%d')

            if source_date_str in data_ls_str and target_date_str in data_ls_str:
                source_idx = data_ls_str.index(source_date_str)
                target_idx = data_ls_str.index(target_date_str)

                source_value = source_case[source_idx]
                target_value = target_case[target_idx]

                print(f"Plotting arrow: {source} leads {target} on {target_date_str}: {source_value} -> {target_value}")

                # Convert back to datetime for plotting
                source_date_plot = pd.to_datetime(source_date_str)
                target_date_plot = pd.to_datetime(target_date_str)

                # Add an arrow showing the lead-lag relationship
                ax.annotate(
                    '',
                    xy=(target_date_plot, target_value),
                    xytext=(source_date_plot, source_value),
                    arrowprops=dict(arrowstyle="->", color='black', lw=2),
                )

                # Annotate with text to explain the relationship
                ax.text(
                    target_date_plot, target_value + 50,
                    f'{source} leads {target} {lag} weeks',
                    fontsize=10, color='black',
                    ha='center', va='bottom'
                )

    # Properly scale the Y-axis for better visibility of arrows
    y_min = min(min(source_case), min(all_target_case_no))
    y_max = max(max(source_case), max(all_target_case_no))
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


def density_plot(density_ls, US_cases, data_ls):
    fig, ax1 = plt.subplots()
    # Plot first curve
    ax1.plot(data_ls, US_cases, label='US Hospitalizations', color='royalblue', linewidth=2, marker='o')
    ax1.set_ylabel('US Hospitalizations', color='royalblue', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='royalblue')

    # Create secondary axis
    ax2 = ax1.twinx()
    ax2.plot(data_ls, density_ls, label='Density of Causal Edges', color='crimson', linewidth=2, marker='s')
    ax2.set_ylabel('Density of Causal Edges', color='crimson', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='crimson')

    # Formatting x-axis to show dates
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)

    # Add grid and title
    ax1.grid()
    # Set title with better font and spacing
    plt.title('Density of Causal Edges VS US Hospitalizations', fontsize=14, fontweight='bold')

    # Tight layout for better spacing
    plt.tight_layout()

    # Show the plot
    plt.show()


def outdegree_plot(outdegree_ls, US_cases, data_ls):
    fig, ax1 = plt.subplots()

    # Plot first curve
    ax1.plot(data_ls, US_cases, label='US Hospitalizations', color='royalblue', linewidth=2, marker='o')
    ax1.set_ylabel('US Hospitalizations', color='royalblue', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='royalblue')

    # Create secondary axis
    ax2 = ax1.twinx()
    ax2.plot(data_ls, outdegree_ls, label='# Outgoing Causal Edges', color='crimson', linewidth=2, marker='s')
    ax2.set_ylabel('# Outgoing Causal Edges', color='crimson', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='crimson')

    # Formatting x-axis to show dates
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)

    # Add grid and title
    ax1.grid()
    # Set title with better font and spacing
    plt.title('# Outgoing Causal Edges VS US Hospitalizations', fontsize=14, fontweight='bold')

    # Tight layout for better spacing
    plt.tight_layout()

    # Show the plot
    plt.show()


def heatmap_plot(degree_check):
    outgoing_ls = defaultdict(list)
    for index in range(11, len(degree_check)):
        cur_MTE = degree_check[index - 4: index, index, :, :]
        sum_outdegree_single_time = defaultdict(list)
        for i in range(0, 4):
            cur_i_lag_MTE = cur_MTE[i]
            for each_state in range(0, 52):
                state_name = index_state[each_state]
                out_degrees = len(cur_i_lag_MTE[each_state].nonzero()[0])
                sum_outdegree_single_time[state_name].append(out_degrees)
        for state_name in sum_outdegree_single_time.keys():
            outgoing_ls[state_name].append(sum(sum_outdegree_single_time[state_name]))

    sorted_states = []
    region_colors = []

    for region, states in hhs_regions.items():
        sorted_states.extend(states)
        region_colors.extend([hhs_colors[region]] * len(states))  # Add states only without HHS prefix

    # Convert defaultdict to 2D NumPy array with states ordered by HHS region
    heatmap_array = np.array([outgoing_ls[state] for state in sorted_states if state in outgoing_ls])

    # Example datetime list
    data_ls = pd.date_range(start='2022-04-30', periods=142, freq='7D').tolist()
    formatted_dates = [date.strftime('%Y-%m-%d') for date in data_ls]

    # Create the heatmap
    fig, ax = plt.subplots(figsize=(14, 12))
    cax = ax.imshow(heatmap_array, aspect='auto', cmap='viridis')

    # Set x-axis labels with an interval to avoid overcrowding
    num_dates = len(formatted_dates)
    interval = max(num_dates // 20, 1)  # Show at most 10 date labels
    selected_dates = formatted_dates[::interval]

    # Format x-axis labels to 'YYYY-MM-DD'
    ax.set_xticks(np.arange(0, num_dates, interval))
    ax.set_xticklabels(selected_dates, rotation=45, ha="right")

    # Manually add colored state labels next to the heatmap with correct alignment
    for i, state in enumerate(sorted_states):
        ax.text(-1.2, i, state, color=region_colors[i], fontsize=10, va='center', ha='right')

    # Add thick boundary lines to separate HHS regions
    region_start_index = 0
    for region, states in hhs_regions.items():
        region_start_index += len(states)
        ax.axhline(y=region_start_index - 0.5, color='white', linestyle='-', linewidth=2)

    # Add color bar
    plt.colorbar(cax, label="Value Scale")

    # Add labels and title
    plt.xlabel('Date')
    plt.ylabel('States by HHS Regions')
    plt.title('Heatmap for # Outgoing Causal Edges with HHS Group Boundaries')

    # Enhance spacing
    plt.tight_layout()
    plt.show()


def snap_shot(source_state,target_state, case, lag_df, data_ls):
    source_state_index = state_index[source_state]
    case_start_index = 7
    end_index = case_start_index + 4
    twelve_weeks_after_index = end_index + 11
    pdf_filename = "spatiotemporal_causal_graphs_for_{}.pdf".format(source_state)
    with PdfPages(pdf_filename) as pdf:
        for index in range(len(data_ls) - 12):
            fig, ax = plt.subplots(figsize=(12, 8))
            x_date = data_ls[index:index + 12]
            last_date = x_date[-1]
            current_time = x_date[0]
            # Generate the previous 4 weekly timestamps
            previous_timestamps = [(current_time - timedelta(weeks=i)).strftime("%Y-%m-%d") for i in range(4, 0, -1)]
            x_date = [da.strftime("%Y-%m-%d") for da in x_date]
            previous_timestamps.extend(x_date)

            soruce_state_case = case.iloc[:, source_state_index + 1].values[
                                case_start_index:twelve_weeks_after_index + 1]

            ax.plot(previous_timestamps, soruce_state_case, label='{} Hospitalizations'.format(source_state),
                    color='royalblue', linewidth=2,
                    marker='s')
            plt.xticks(rotation=45, ha='right')
            sub_lag_df = lag_df[
                (lag_df['source'] == source_state) & (lag_df['date'] <= x_date[-1]) & (lag_df['target'] == target_state)
                & (lag_df['date'] >= previous_timestamps[0])]
            if len(sub_lag_df) == 0:
                plt.close()
                continue
                case_start_index = case_start_index + 1
                end_index = case_start_index + 4
                twelve_weeks_after_index = end_index + 11
            else:
                all_target_case_no = list()
                target_state_ls = set(sub_lag_df['target'].values)
                for i, target_state in enumerate(target_state_ls):
                    target_case = case.iloc[:, state_index[target_state] + 1].values[
                                  case_start_index:twelve_weeks_after_index + 1]
                    all_target_case_no.extend(target_case)
                    # Sample trajectory data in DataFrame format
                    # Convert all dates to string format for lookup
                    ax.plot(previous_timestamps, target_case, label='{} Hospitalizations'.format(target_state), color='orange',
                            linewidth=2,
                            marker='o')

                    # Debug: Print available dates for comparison
                    # Add arrows for trajectory relationships from the DataFrame
                    for _, row in sub_lag_df.iterrows():
                        source = row['source']
                        target = row['target']
                        lag = row['lag']
                        target_date_str = row['date']  # Target date as string

                        # Convert target date and calculate source date
                        target_date_str = pd.to_datetime(target_date_str).strftime('%Y-%m-%d')
                        source_date_str = (pd.to_datetime(target_date_str) - pd.Timedelta(days=lag * 7)).strftime(
                            '%Y-%m-%d')

                        if source_date_str in previous_timestamps and target_date_str in previous_timestamps:
                            source_idx = previous_timestamps.index(source_date_str)
                            target_idx = previous_timestamps.index(target_date_str)

                            source_value = soruce_state_case[source_idx]
                            target_value = target_case[target_idx]

                            print(
                                f"Plotting arrow: {source} leads {target} on {target_date_str}: {source_value} -> {target_value}")

                            # Convert back to datetime for plotting
                            source_date_plot = source_date_str
                            target_date_plot = target_date_str

                            # Add an arrow showing the lead-lag relationship
                            ax.annotate(
                                '',
                                xy=(target_date_plot, target_value),
                                xytext=(source_date_plot, source_value),
                                arrowprops=dict(arrowstyle="->", color='black', lw=2),
                            )

                            # Annotate with text to explain the relationship
                            ax.text(
                                target_date_plot, target_value + 50,
                                f'{source} leads {target} {lag} weeks',
                                fontsize=10, color='black',
                                ha='center', va='bottom'
                            )

                    # Properly scale the Y-axis for better visibility of arrows
                y_min = min(min(soruce_state_case), min(all_target_case_no))
                y_max = max(max(soruce_state_case), max(all_target_case_no))
                padding = (y_max - y_min) * 0.1  # Add 10% padding

                ax.set_ylim(y_min - padding, y_max + padding)

                plt.xticks(rotation=45)

                # Add labels and legend
                ax.set_xlabel('Date')
                ax.set_ylabel('Hospitalization Numbers')
                ax.set_title('Hospitalization Numbers with Trajectory Relationships')
                ax.legend()

                # Improve grid appearance
                ax.grid(which='major', linestyle='--', linewidth=0.5, alpha=0.7)

                # Show the plot
                plt.tight_layout()
                # Save the current plot to the PDF
                pdf.savefig()
                plt.close()

            case_start_index = case_start_index + 1
            end_index = end_index + 1
            twelve_weeks_after_index = twelve_weeks_after_index + 1


def plot_edge_categorization(strong_to_weak, strong_to_strong, weak_to_strong, weak_to_weak, date_ls, US_cases):
    fig, ax1 = plt.subplots()
    # Plot first curve
    ax1.plot(date_ls, US_cases, label='US Hospitalizations', color='royalblue', linewidth=2, marker='o')
    ax1.set_ylabel('US Hospitalizations', color='royalblue', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='royalblue')

    # Create secondary axis
    ax2 = ax1.twinx()
    ax2.plot(date_ls, strong_to_weak, label='Strong to Weak', color='crimson', linewidth=2)
    ax2.plot(date_ls, strong_to_strong, label='Strong to Strong', color='green', linewidth=2)
    ax2.plot(date_ls, weak_to_strong, label='Weak to Strong', color='purple', linewidth=2)
    ax2.plot(date_ls, weak_to_weak, label='Weak to Weak', color='orange', linewidth=2)

    ax2.set_ylabel('Number of Outgoing Edges', color='crimson', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='crimson')

    # Formatting x-axis to show dates
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)

    # Add grid and title
    ax1.grid()
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    # Set title with better font and spacing
    plt.title('Number of Outgoing Causal Edges VS US Hospitalizations', fontsize=14, fontweight='bold')

    # Tight layout for better spacing
    plt.tight_layout()

    # Show the plot
    plt.show()


def outgoing_causal_edge_distribution_plot(outgoing_lag_df, date_ls):
    pdf_filename = 'TE_box_plot_across_time.pdf'
    plt.figure(figsize=(15, 6))  # Adjust figure size for better readability

    positions = range(len(date_ls))  # Define x-axis positions
    mean_values = []  # To store the mean of each box plot

    # Iterate over each timestamp to create and plot its corresponding box plot
    for i, date in enumerate(date_ls):
        sub_outgoing_lag_df = outgoing_lag_df[outgoing_lag_df['date'] == date]
        outgoing_lag_df_MTE_values = sub_outgoing_lag_df['MTE'].values
        if len(outgoing_lag_df_MTE_values) > 0:
            # Create a box plot at a specific position on the x-axis
            plt.boxplot(outgoing_lag_df_MTE_values, positions=[i], widths=0.6)

            # Calculate and store the mean value of the current box
            mean_values.append(np.mean(outgoing_lag_df_MTE_values))
        else:
            mean_values.append(np.nan)  # Handle missing data points

    # Plot the curve using the mean values
    plt.plot(positions, mean_values, color='red', marker='o', linewidth=2, label='Mean TE Values')

    # Use formatted and reduced date labels for readability
    plt.xticks(ticks=positions[::5], labels=[date_ls[i] for i in range(0, len(date_ls), 5)], rotation=45, ha='right')

    plt.ylabel('TE Value')
    plt.xlabel('Date')
    plt.title('Box Plot of TE values across time with Mean Trend')

    plt.grid(axis='y')  # Add grid for better visibility
    plt.legend()  # Add legend to identify mean curve

    # Save and show the plot
    plt.tight_layout()
    plt.savefig(pdf_filename)
    plt.show()


def each_state_MTE_density_plot(outgoing_lag_df):
    state_population = {
        'CA': 39538223, 'TX': 29145505, 'FL': 21538187, 'NY': 20201249, 'PA': 13002700,
        'IL': 12812508, 'OH': 11799448, 'GA': 10711908, 'NC': 10439388, 'MI': 10077331,
        'NJ': 9288994, 'VA': 8631393, 'WA': 7693612, 'AZ': 7151502, 'MA': 7029917,
        'TN': 6910840, 'IN': 6785528, 'MO': 6154913, 'MD': 6177224, 'WI': 5893718,
        'CO': 5773714, 'MN': 5706494, 'SC': 5118425, 'AL': 5024279, 'LA': 4657757,
        'KY': 4505836, 'OR': 4237256, 'OK': 3959353, 'CT': 3605944, 'UT': 3271616,
        'IA': 3190369, 'NV': 3104614, 'AR': 3011524, 'MS': 2961279, 'KS': 2937880,
        'NM': 2117522, 'NE': 1961504, 'WV': 1793716, 'ID': 1839106, 'HI': 1455271,
        'NH': 1377529, 'ME': 1362359, 'MT': 1084225, 'RI': 1097379, 'DE': 989948,
        'SD': 886667, 'ND': 779094, 'AK': 733391, 'VT': 643077, 'WY': 576851
    }
    # Extract unique state names from the DataFrame
    states = outgoing_lag_df['source'].unique().tolist()
    # Sort the states by population
    states_sorted_by_population = sorted(states, key=lambda state: state_population.get(state, 0), reverse=True)
    pdf_filename = 'TE_distribution_for_all_states.pdf'
    with PdfPages(pdf_filename) as pdf:
        for source_state in states_sorted_by_population:
            plt.figure(figsize=(15, 6))  # Adjust figure size for better readability
            sub_outgoing_lag_df = outgoing_lag_df[outgoing_lag_df['source'] == source_state]
            outgoing_lag_df_MTE_values = sub_outgoing_lag_df['MTE'].values
            if len(outgoing_lag_df_MTE_values) > 0:
                # Plot the KDE
                sns.kdeplot(outgoing_lag_df_MTE_values, fill=True, color='blue', alpha=0.6)
                # Add labels and title
                plt.xlabel('Value')
                plt.ylabel('Density')
                plt.title('General TE Distribution of {}'.format(source_state))
                # Show the plot
                plt.tight_layout()
                # Save the current plot to the PDF
                pdf.savefig()
                plt.close()
            else:
                continue


def each_state_MTE_density_plot_across_time(source_state, outgoing_lag_df, date_ls):
    pdf_filename = 'TE_distribution_across_time_for_{}.pdf'.format(source_state)
    plt.figure(figsize=(15, 6))  # Adjust figure size for better readability
    positions = range(len(date_ls))  # Define x-axis positions
    mean_values = []  # To store the mean of each box plot
    with PdfPages(pdf_filename) as pdf:
        # Iterate over each timestamp to create and plot its corresponding box plot
        for i, date in enumerate(date_ls):
            sub_outgoing_lag_df = outgoing_lag_df[outgoing_lag_df['date'] == date]
            outgoing_lag_df_MTE_values = sub_outgoing_lag_df['MTE'].values

            if len(outgoing_lag_df_MTE_values) > 0:
                # Plot the KDE
                sns.kdeplot(outgoing_lag_df_MTE_values, fill=True, color='blue', alpha=0.6)

                # Add labels and title
                plt.xlabel('Value')
                plt.ylabel('Density')
                plt.title('TE Distribution of {} By {}'.format(source_state, date))
                # Show the plot
                plt.tight_layout()
                # Save the current plot to the PDF
                pdf.savefig()
                plt.close()
            else:
                continue


def each_state_MTE_box_plot(source_state, outgoing_lag_df, date_ls):
    plt.figure(figsize=(15, 6))  # Adjust figure size for better readability
    positions = range(len(date_ls))  # Define x-axis positions
    mean_values = []  # To store the mean of each box plot
    # Iterate over each timestamp to create and plot its corresponding box plot
    pdf_filename = 'TE_box_plot_across_time_for_{}.pdf'.format(source_state)
    with PdfPages(pdf_filename) as pdf:
        for i, date in enumerate(date_ls):
            sub_outgoing_lag_df = outgoing_lag_df[
                (outgoing_lag_df['source'] == source_state) & (outgoing_lag_df['date'] == date)]
            outgoing_lag_df_MTE_values = sub_outgoing_lag_df['MTE'].values

            if len(outgoing_lag_df_MTE_values) > 0:
                # Create a box plot at a specific position on the x-axis
                plt.boxplot(outgoing_lag_df_MTE_values, positions=[i], widths=0.6)

                # Calculate and store the mean value of the current box
                mean_values.append(np.mean(outgoing_lag_df_MTE_values))
            else:
                mean_values.append(np.nan)  # Handle missing data points

        # Plot the curve using the mean values
        plt.plot(positions, mean_values, color='red', marker='o', linewidth=2, label='Mean TE Values')

        # Use formatted and reduced date labels for readability
        plt.xticks(ticks=positions[::5], labels=[date_ls[i] for i in range(0, len(date_ls), 5)], rotation=45,
                   ha='right')

        plt.ylabel('TE Value')
        plt.xlabel('Date')
        plt.title('Box Plot of TE values For {} across time with Mean Trend'.format(source_state))

        plt.grid(axis='y')  # Add grid for better visibility
        plt.legend()  # Add legend to identify mean curve

        # Save and show the plot
        plt.tight_layout()
        # Save the current plot to the PDF
        pdf.savefig()
        plt.close()


def state_box_plot_per_time_ranked_by_population(outgoing_lag_df, date_ls):
    state_population = {
        'CA': 39538223, 'TX': 29145505, 'FL': 21538187, 'NY': 20201249, 'PA': 13002700,
        'IL': 12812508, 'OH': 11799448, 'GA': 10711908, 'NC': 10439388, 'MI': 10077331,
        'NJ': 9288994, 'VA': 8631393, 'WA': 7693612, 'AZ': 7151502, 'MA': 7029917,
        'TN': 6910840, 'IN': 6785528, 'MO': 6154913, 'MD': 6177224, 'WI': 5893718,
        'CO': 5773714, 'MN': 5706494, 'SC': 5118425, 'AL': 5024279, 'LA': 4657757,
        'KY': 4505836, 'OR': 4237256, 'OK': 3959353, 'CT': 3605944, 'UT': 3271616,
        'IA': 3190369, 'NV': 3104614, 'AR': 3011524, 'MS': 2961279, 'KS': 2937880,
        'NM': 2117522, 'NE': 1961504, 'WV': 1793716, 'ID': 1839106, 'HI': 1455271,
        'NH': 1377529, 'ME': 1362359, 'MT': 1084225, 'RI': 1097379, 'DE': 989948,
        'SD': 886667, 'ND': 779094, 'AK': 733391, 'VT': 643077, 'WY': 576851
    }

    # Extract unique state names from the DataFrame
    states = outgoing_lag_df['source'].unique().tolist()

    # Sort the states by population
    states_sorted_by_population = sorted(states, key=lambda state: state_population.get(state, 0), reverse=True)

    # Define x-axis positions based on the sorted order
    positions = range(len(states_sorted_by_population))

    for i, date in enumerate(date_ls):
        plt.figure(figsize=(24, 6))  # Adjust figure size for better readability
        pdf_filename = 'TE_box_plot_across_state_by_{}.pdf'.format(date)
        mean_values = []
        state_list = []

        with PdfPages(pdf_filename) as pdf:
            for index, state in enumerate(states_sorted_by_population):
                state_list.append(state)
                sub_outgoing_lag_df = outgoing_lag_df[
                    (outgoing_lag_df['source'] == state) & (outgoing_lag_df['date'] == date)
                    ]
                outgoing_lag_df_MTE_values = sub_outgoing_lag_df['MTE'].values

                if len(outgoing_lag_df_MTE_values) > 0:
                    # Create a box plot at a specific position on the x-axis
                    plt.boxplot(outgoing_lag_df_MTE_values, positions=[index], widths=0.6)

                    # Calculate and store the mean value of the current box
                    mean_values.append(np.mean(outgoing_lag_df_MTE_values))
                else:
                    mean_values.append(np.nan)  # Handle missing data points

            # Plot the curve using the mean values
            plt.plot(positions, mean_values, color='red', marker='o', linewidth=2, label='Mean TE Values')

            # Set x-axis labels to sorted state names
            plt.xticks(ticks=positions, labels=state_list, rotation=45, ha='right', fontsize=12)

            plt.ylabel('TE Value', fontsize=14)
            plt.xlabel('States (sorted by population)', fontsize=14)
            plt.title('Box Plot of TE values across States with Mean Trend by {}'.format(date), fontsize=16)

            plt.grid(axis='y')  # Add grid for better visibility
            plt.legend()  # Add legend to identify mean curve

            # Save and show the plot
            plt.tight_layout()
            pdf.savefig()
            plt.close()


def state_box_plot_ranked_by_population(outgoing_lag_df):
    state_population = {
        'CA': 39538223, 'TX': 29145505, 'FL': 21538187, 'NY': 20201249, 'PA': 13002700,
        'IL': 12812508, 'OH': 11799448, 'GA': 10711908, 'NC': 10439388, 'MI': 10077331,
        'NJ': 9288994, 'VA': 8631393, 'WA': 7693612, 'AZ': 7151502, 'MA': 7029917,
        'TN': 6910840, 'IN': 6785528, 'MO': 6154913, 'MD': 6177224, 'WI': 5893718,
        'CO': 5773714, 'MN': 5706494, 'SC': 5118425, 'AL': 5024279, 'LA': 4657757,
        'KY': 4505836, 'OR': 4237256, 'OK': 3959353, 'CT': 3605944, 'UT': 3271616,
        'IA': 3190369, 'NV': 3104614, 'AR': 3011524, 'MS': 2961279, 'KS': 2937880,
        'NM': 2117522, 'NE': 1961504, 'WV': 1793716, 'ID': 1839106, 'HI': 1455271,
        'NH': 1377529, 'ME': 1362359, 'MT': 1084225, 'RI': 1097379, 'DE': 989948,
        'SD': 886667, 'ND': 779094, 'AK': 733391, 'VT': 643077, 'WY': 576851
    }
    # Extract unique state names from the DataFrame
    states = outgoing_lag_df['source'].unique().tolist()
    # Sort the states by population
    states_sorted_by_population = sorted(states, key=lambda state: state_population.get(state, 0), reverse=True)
    pdf_filename = 'TE_box_plot_across_time_all_states.pdf'
    plt.figure(figsize=(24, 6))  # Adjust figure size for better readability
    positions = range(len(states_sorted_by_population))  # Define x-axis positions
    mean_values = []  # To store the mean of each box plot
    # Iterate over each timestamp to create and plot its corresponding box plot
    for i, state in enumerate(states_sorted_by_population):
        sub_outgoing_lag_df = outgoing_lag_df[outgoing_lag_df['source'] == state]
        outgoing_lag_df_MTE_values = sub_outgoing_lag_df['MTE'].values

        if len(outgoing_lag_df_MTE_values) > 0:
            # Create a box plot at a specific position on the x-axis
            plt.boxplot(outgoing_lag_df_MTE_values, positions=[i], widths=0.6)

            # Calculate and store the mean value of the current box
            mean_values.append(np.mean(outgoing_lag_df_MTE_values))
        else:
            mean_values.append(np.nan)  # Handle missing data points

    # Plot the curve using the mean values
    plt.plot(positions, mean_values, color='red', marker='o', linewidth=2, label='Mean TE Values')

    # Use formatted and reduced date labels for readability
    plt.xticks(ticks=positions, labels=states_sorted_by_population, rotation=45, ha='right')

    plt.ylabel('TE Value')
    plt.xlabel('States')
    plt.title('Box Plot of TE values across states with Mean Trend')

    plt.grid(axis='y')  # Add grid for better visibility
    plt.legend()  # Add legend to identify mean curve

    # Save and show the plot
    plt.tight_layout()
    plt.savefig(pdf_filename)
    plt.show()


if __name__ == '__main__':
    degree_check = np.load('MTE_matrices_flu_hosp.npy')
    covid = pd.read_csv(
        'C:\\Users\\6\\PycharmProjects\\FORECASTING_GNN\\epidemic_data\\COVID_CASE\\COVID_CASE_DATA.csv')
    covid = covid.reset_index().rename(columns={'index': 'Index'})
    case = pd.read_csv('CDC_DATA/transferred_hospital_admission_0111_w_US.csv')
    index_state = dict(zip(covid['Index'], covid['Abbr']))
    index_state[51] = 'PR'
    count_source_target = list()
    # select states whose value > 1000 as strong
    strong_states = ['CA', 'FL', 'TX', 'NY', 'PA', 'AZ', 'NJ']
    weak_states = [st for st in index_state.values() if st not in strong_states]
    for index in range(11, len(degree_check)):
        temp_source_target = list()
        cur_MTE = degree_check[index - 4:index, index, :, :]
        cur_MTE_agg = np.sum(cur_MTE, axis=0)
        for i in range(len(cur_MTE_agg)):
            for j in range(len(cur_MTE_agg)):
                if cur_MTE_agg[i][j]:
                    if i == j:
                        continue
                    else:
                        temp_source_target.append([i, j])
        count_source_target.extend(temp_source_target)

    df = pd.DataFrame(count_source_target)
    frequency = df.value_counts().reset_index(name='count')
    frequency.columns = ['source', 'target', 'count']
    frequency['target'] = frequency['target'].map(index_state)
    frequency['source'] = frequency['source'].map(index_state)
    frequency.to_csv('source_target_frequency_table.csv', mode='a')

    lag_df = list()
    for index in range(11, len(degree_check)):
        minus_4 = index - 4
        cur_MTE = degree_check[minus_4:index, index, :, :]
        for lag in range(0, 4):
            real_lag = 4 - lag
            for i in range(52):
                for j in range(52):
                    if cur_MTE[lag, i, j]:
                        date = datetime.strptime('2022-02-12', "%Y-%m-%d") + timedelta(days=index * 7)
                        lag_df.append([i, j, real_lag, cur_MTE[lag, i, j], date.strftime("%Y-%m-%d")])
    lag_df = pd.DataFrame(lag_df)
    lag_df.columns = ['source', 'target', 'lag', 'MTE', 'date']
    lag_df['source'] = lag_df['source'].map(index_state)
    lag_df['target'] = lag_df['target'].map(index_state)
    # lag_df.to_csv('lag_df.csv', mode='a')

    outgoing_lag_df = lag_df[lag_df['MTE'] != 1]

    all_date = set(lag_df['date'].values)
    date_ls = sorted(list(all_date))
    '''
    each_state_MTE_density_plot(outgoing_lag_df)

    # for source_state in index_state.values():
    #    each_state_MTE_density_plot_across_time(source_state, outgoing_lag_df, date_ls)

    # states plot per timestamp, ranked by population
    # state_box_plot_per_time_ranked_by_population(outgoing_lag_df, date_ls)

    # states plot across timestamp, ranked by population
    state_box_plot_ranked_by_population(outgoing_lag_df)

    # for source_state in index_state.values():
    #    each_state_MTE_box_plot(source_state, outgoing_lag_df, date_ls)
    outgoing_causal_edge_distribution_plot(outgoing_lag_df, date_ls)

    strong_connections_df = lag_df[(lag_df['source'].isin(strong_states)) & (lag_df['target'].isin(strong_states))]

    strong_connections_ratio = len(strong_connections_df) / len(lag_df)

    strong_to_weak = list()
    strong_to_strong = list()
    weak_to_strong = list()
    weak_to_weak = list()

    strong_to_weak_df = lag_df[(lag_df['source'].isin(strong_states)) & (lag_df['target'].isin(weak_states))]
    strong_to_strong_df = lag_df[(lag_df['source'].isin(strong_states)) & (lag_df['target'].isin(strong_states))]
    weak_to_strong_df = lag_df[(lag_df['source'].isin(weak_states)) & (lag_df['target'].isin(strong_states))]
    weak_to_weak_df = lag_df[(lag_df['source'].isin(weak_states)) & (lag_df['target'].isin(weak_states))]

    for index, dt in enumerate(date_ls):
        strong_to_weak_df = lag_df[
            (lag_df['source'].isin(strong_states)) & (lag_df['target'].isin(weak_states)) & (lag_df['date'] == dt)]
        strong_to_strong_df = lag_df[
            (lag_df['source'].isin(strong_states)) & (lag_df['target'].isin(strong_states)) & (lag_df['date'] == dt)]
        weak_to_strong_df = lag_df[
            (lag_df['source'].isin(weak_states)) & (lag_df['target'].isin(strong_states)) & (lag_df['date'] == dt)]
        weak_to_weak_df = lag_df[
            (lag_df['source'].isin(weak_states)) & (lag_df['target'].isin(weak_states)) & (lag_df['date'] == dt)]

        start_date = datetime.strptime('2022-04-30', "%Y-%m-%d") + timedelta(days=index * 7)
        date_ls[index] = start_date

        strong_to_weak.append(len(strong_to_weak_df))
        strong_to_strong.append(len(strong_to_strong_df))
        weak_to_strong.append(len(weak_to_strong_df))
        weak_to_weak.append(len(weak_to_weak_df))
    '''
    US_cases = case['US'].values[11:]
    # plot_edge_categorization(strong_to_weak, strong_to_strong, weak_to_strong, weak_to_weak, date_ls, US_cases)


    hhs_map = pd.read_csv('us_subplot_grid.csv')

    state_hhs = dict(zip(hhs_map['State'], hhs_map['HHS']))
    state_hhs_reverse = {}
    for key, value in state_hhs.items():
        state_hhs_reverse.setdefault(value, []).append(key)

    data_ls = list()
    density_ls = list()
    outdegree_ls = list()
    with open('aggregated_df.pkl', 'rb') as f:
        aggregated_df = pickle.load(f)
    for index, df in enumerate(aggregated_df):
        density_ls.append(df['density'].values[0])
        outdegree_ls.append(sum(df['out_degree'].values))
        start_date = datetime.strptime('2022-04-30', "%Y-%m-%d") + timedelta(days=index * 7)
        data_ls.append(start_date)

    source_state = 'CA'
    target_state = 'TX'
    state_index = {value: key for key, value in index_state.items()}

    source_state_hosp = case['06'].values
    target_state_hosp = case['48'].values
    data_x_axis = case['date'].values

    plt.figure(figsize=(12, 6))
    plt.plot(data_x_axis, source_state_hosp, color='royalblue', label="CA Hospitalizations", linestyle='-',
             linewidth=2.5)
    plt.plot(data_x_axis, target_state_hosp, color='orange', label="TX Hospitalizations", linestyle='-', linewidth=2.5)

    # Formatting the plot
    plt.xlabel("Date", fontsize=16, fontweight='bold')
    plt.ylabel("Hospitalizations", fontsize=16, fontweight='bold')
    plt.title("Hospitalization Trends for CA and TX States", fontsize=18, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=18, loc="upper left", frameon=True)

    # Adjust x-axis tick frequency for better spacing
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(nbins=20))

    # Increase grid visibility
    plt.grid(True, linestyle="--", linewidth=0.7, alpha=0.7)

    # Show the plot
    plt.show()


    '''
    sub_lag_df = lag_df[(lag_df['source'] == source_state) & (lag_df['target'] != source_state)]
    counts = Counter(sub_lag_df['target'].values).most_common(5)
    target_state_ls = [item[0] for item in counts]
    snap_shot(source_state, target_state, case, lag_df, data_ls)
    '''
    sub_lag_df = lag_df[(lag_df['source'] == source_state) & (lag_df['target'] == target_state)]
    state_plots(source_state, target_state, sub_lag_df, data_ls, case)

    topk_state_plots(source_state, target_state_ls, sub_lag_df, data_ls, case)

    density_plot(density_ls, US_cases, data_ls)
    outdegree_plot(outdegree_ls, US_cases, data_ls)

    heatmap_plot(degree_check)

    # network analysis on source to different HHS region
    hhs_regions_reversed = {st: reg for reg, sta in hhs_regions.items() for st in sta}
    data_ls_str = [date.strftime('%Y-%m-%d') for date in data_ls]
    summary_table = list()
    date_index = 0
    for index in range(11, len(degree_check)):
        cur_degree_check = degree_check[(index - 4):index, index, :, :]
        datetime_cur = data_ls_str[date_index]
        for state in range(len(cur_degree_check[0])):
            cur_state_degree_matrix = cur_degree_check[:, state, :]
            state_name = index_state[state]
            hhs_belonging = hhs_map[hhs_map['State'] == state_name]['HHS'].values[0]
            for lag in range(1, 5):
                lag_index = lag - 1
                real_lag = 4 - lag_index
                current_lag_MTE = cur_state_degree_matrix[lag_index]
                indices = [index for index, value in enumerate(current_lag_MTE) if value != 0]
                outgoing_state_names = [index_state[i] for i in indices]
                hhs_frequency = defaultdict()
                for st_name in outgoing_state_names:
                    target_hhs = hhs_regions_reversed[st_name]
                    if target_hhs not in hhs_frequency.keys():
                        hhs_frequency[target_hhs] = 1
                    else:
                        hhs_frequency[target_hhs] = hhs_frequency[target_hhs] + 1
                total_intra_connections = 0
                for key in hhs_frequency.keys():
                    if hhs_belonging == key:
                        temp_hhs_ls = [state_name, hhs_belonging, real_lag, hhs_frequency[hhs_belonging], 0, 0,
                                       datetime_cur]
                    else:
                        temp_hhs_ls = [state_name, hhs_belonging, real_lag, 0, hhs_frequency[key], key, datetime_cur]
                    # state, hhs group, current lead lag, in-hhs connection number, out-hhs connection number, outgoing edge group, datetime
                    summary_table.append(temp_hhs_ls)
        date_index = date_index + 1

    summary_table = pd.DataFrame(summary_table)
    summary_table.columns = ['State', 'Belonging HHS', 'Lead by weeks', 'Inter-HHS Connections',
                             'Intra-HHS Connections', 'Outgoing HHS Group', 'Date']
    summary_table.replace(0, '', inplace=True)
    summary_table.to_csv('Inter and Intra-HHS Connections.csv', mode='a')
