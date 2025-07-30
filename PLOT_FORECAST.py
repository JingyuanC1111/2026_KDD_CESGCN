import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
last_season = ['2023-11-18','2023-11-25','2023-12-02','2023-12-09',
               '2023-12-16','2023-12-23','2023-12-30','2024-01-06',
               '2024-01-13', '2024-01-20', '2024-01-27','2024-02-03']

latest_season = ['2024-11-23','2024-11-30','2024-12-07','2024-12-14',
                 '2024-12-21','2024-12-28','2025-01-04','2025-01-11',
                 '2025-01-18']

flu = pd.read_csv('CDC_DATA/transferred_hospital_admission_0125.csv')
first_index = np.where(flu['date'].values == latest_season[0])[0]
last_index = np.where(flu['date'].values == latest_season[-1])[0]
# CA
CA_truth = flu.iloc[first_index[0]:last_index[0]+1,5].values
TX_truth = flu.iloc[first_index[0]:last_index[0]+1,-9].values
GA_truth = flu.iloc[first_index[0]:last_index[0]+1,11].values

lowest = 0.01
highest = 0.99
CA_base_low = defaultdict(list)
CA_base_high = defaultdict(list)
CA_ensemble_low = defaultdict(list)
CA_ensemble_high = defaultdict(list)

TX_base_low = defaultdict(list)
TX_base_high = defaultdict(list)
TX_ensemble_low = defaultdict(list)
TX_ensemble_high = defaultdict(list)

GA_base_low = defaultdict(list)
GA_base_high = defaultdict(list)
GA_ensemble_low = defaultdict(list)
GA_ensemble_high = defaultdict(list)

for dt in latest_season:
    CAbaseline = pd.read_csv('{}-FluSight-baseline.csv'.format(dt))
    CAensemble = pd.read_csv('{}-FluSight-ensemble.csv'.format(dt))

    CAbaseline = CAbaseline[CAbaseline['location'] == '06']
    CAbaseline = CAbaseline[(CAbaseline['output_type_id'] == str(lowest)) | (CAbaseline['output_type_id'] == str(highest))]
    CAbaseline = CAbaseline[CAbaseline['horizon'] >= 0]
    CAbaseline_low = CAbaseline[CAbaseline['output_type_id'] == str(lowest)]['value'].values
    CAbaseline_hight = CAbaseline[CAbaseline['output_type_id'] == str(highest)]['value'].values
    CA_base_low[dt].extend(CAbaseline_low)
    CA_base_high[dt].extend(CAbaseline_hight)

    CAensemble = CAensemble[CAensemble['location'] == '06']
    CAensemble = CAensemble[(CAensemble['output_type_id'] == str(lowest)) | (CAensemble['output_type_id'] == str(highest))]
    CAensemble = CAensemble[CAensemble['horizon'] >= 0]
    CAensemble_low = CAensemble[CAensemble['output_type_id'] == str(lowest)]['value'].values
    CAensemble_hight = CAensemble[CAensemble['output_type_id'] == str(highest)]['value'].values
    CA_ensemble_low[dt].extend(CAensemble_low)
    CA_ensemble_high[dt].extend(CAensemble_hight)

    TXbaseline = pd.read_csv('{}-FluSight-baseline.csv'.format(dt))
    TXensemble = pd.read_csv('{}-FluSight-ensemble.csv'.format(dt))

    TXbaseline = TXbaseline[TXbaseline['location'] == '48']
    TXbaseline = TXbaseline[
        (TXbaseline['output_type_id'] == str(lowest)) | (TXbaseline['output_type_id'] == str(highest))]
    TXbaseline = TXbaseline[TXbaseline['horizon'] >= 0]
    TXbaseline_low = TXbaseline[TXbaseline['output_type_id'] == str(lowest)]['value'].values
    TXbaseline_hight = TXbaseline[TXbaseline['output_type_id'] == str(highest)]['value'].values
    TX_base_low[dt].extend(TXbaseline_low)
    TX_base_high[dt].extend(TXbaseline_hight)

    TXensemble = TXensemble[TXensemble['location'] == '48']
    TXensemble = TXensemble[
        (TXensemble['output_type_id'] == str(lowest)) | (TXensemble['output_type_id'] == str(highest))]
    TXensemble = TXensemble[TXensemble['horizon'] >= 0]
    TXensemble_low = TXensemble[TXensemble['output_type_id'] == str(lowest)]['value'].values
    TXensemble_hight = TXensemble[TXensemble['output_type_id'] == str(highest)]['value'].values
    TX_ensemble_low[dt].extend(TXensemble_low)
    TX_ensemble_high[dt].extend(TXensemble_hight)

    GAbaseline = pd.read_csv('{}-FluSight-baseline.csv'.format(dt))
    GAensemble = pd.read_csv('{}-FluSight-ensemble.csv'.format(dt))

    GAbaseline = GAbaseline[GAbaseline['location'] == '13']
    GAbaseline = GAbaseline[
        (GAbaseline['output_type_id'] == str(lowest)) | (GAbaseline['output_type_id'] == str(highest))]
    GAbaseline = GAbaseline[GAbaseline['horizon'] >= 0]
    GAbaseline_low = GAbaseline[GAbaseline['output_type_id'] == str(lowest)]['value'].values
    GAbaseline_hight = GAbaseline[GAbaseline['output_type_id'] == str(highest)]['value'].values
    GA_base_low[dt].extend(GAbaseline_low)
    GA_base_high[dt].extend(GAbaseline_hight)

    GAensemble = GAensemble[GAensemble['location'] == '13']
    GAensemble = GAensemble[
        (GAensemble['output_type_id'] == str(lowest)) | (GAensemble['output_type_id'] == str(highest))]
    GAensemble = GAensemble[GAensemble['horizon'] >= 0]
    GAensemble_low = GAensemble[GAensemble['output_type_id'] == str(lowest)]['value'].values
    GAensemble_hight = GAensemble[GAensemble['output_type_id'] == str(highest)]['value'].values
    GA_ensemble_low[dt].extend(GAensemble_low)
    GA_ensemble_high[dt].extend(GAensemble_hight)






# Define actual prediction timestamps for visualization
prediction_dates_1123 = ['2024-11-30', '2024-12-07', '2024-12-14','2024-12-21']
prediction_dates_1221 = ['2024-12-21', '2024-12-28', '2025-01-04','2025-01-11']

# Extract values from dictionaries
predicted_values_low_1123 = CA_ensemble_low['2024-11-30']
predicted_values_high_1123 = CA_ensemble_high['2024-11-30']
predicted_values_low_1221 = CA_ensemble_low['2024-12-21']
predicted_values_high_1221 = CA_ensemble_high['2024-12-21']

predicted_values_baseline_low_1123 = CA_base_low['2024-11-30']
predicted_values_baseline_high_1123 = CA_base_high['2024-11-30']
predicted_values_baseline_low_1221 = CA_base_low['2024-12-21']
predicted_values_baseline_high_1221 = CA_base_high['2024-12-21']

predicted_value_low_1130 = [392,460,478, 495]
predicted_value_high_1130 = [1026, 1231, 1678, 1740]
predicted_value_low_1221 = [160, 184, 271, 341]
predicted_value_high_1221 = [3118,2870,3278, 2969]
# Plot the main truth line
plt.figure(figsize=(10, 5))
plt.plot(latest_season, CA_truth, marker='o', color='green', label="Truth")

# Plot the shaded uncertainty regions for ensemble predictions
plt.fill_between(prediction_dates_1123, predicted_values_low_1123, predicted_values_high_1123,
                 color='green', alpha=0.5)
plt.fill_between(prediction_dates_1221, predicted_values_low_1221, predicted_values_high_1221,
                 color='green', alpha=0.5, label="Ensemble Prediction Interval")

# Plot the shaded uncertainty regions for baseline predictions
plt.fill_between(prediction_dates_1123, predicted_value_low_1130, predicted_value_high_1130,
                 color='blue', alpha=0.3)
plt.fill_between(prediction_dates_1221, predicted_value_low_1221, predicted_value_high_1221,
                 color='blue', alpha=0.3, label="CESGCN Prediction Interval")

plt.fill_between(prediction_dates_1123, predicted_values_baseline_low_1123, predicted_values_baseline_high_1123,
                 color='yellow', alpha=0.5)
plt.fill_between(prediction_dates_1221, predicted_values_baseline_low_1221, predicted_values_baseline_high_1221,
                 color='yellow', alpha=0.5, label="Baseline Prediction Interval")

# Enhancing visualization
plt.xlabel("Date", fontsize=14)
plt.ylabel("# Flu Hospitalizations", fontsize=14)
plt.title("CA Flu Hospitalization", fontsize=16, fontweight='bold')
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=10, loc="upper left")
plt.grid(True, linestyle="--", alpha=0.5)

# Display the plot
plt.tight_layout()
plt.show()

########################################################

# Define actual prediction timestamps for visualization
prediction_dates_1123 = ['2024-11-30', '2024-12-07', '2024-12-14','2024-12-21']
prediction_dates_1221 = ['2024-12-21', '2024-12-28', '2025-01-04','2025-01-11']

# Extract values from dictionaries
predicted_values_low_1123 = TX_ensemble_low['2024-11-30']
predicted_values_high_1123 = TX_ensemble_high['2024-11-30']
predicted_values_low_1221 = TX_ensemble_low['2024-12-21']
predicted_values_high_1221 = TX_ensemble_high['2024-12-21']

predicted_values_baseline_low_1123 = TX_base_low['2024-11-30']
predicted_values_baseline_high_1123 = TX_base_high['2024-11-30']
predicted_values_baseline_low_1221 = TX_base_low['2024-12-21']
predicted_values_baseline_high_1221 = TX_base_high['2024-12-21']

predicted_value_low_1130 = [188,238,288, 261]
predicted_value_high_1130 = [620, 777, 1024, 1083]
predicted_value_low_1221 = [593, 407, 379, 317]
predicted_value_high_1221 = [1509,1821,2464, 2781]
# Plot the main truth line
plt.figure(figsize=(10, 5))
plt.plot(latest_season, TX_truth, marker='o', color='green', label="True Data")

# Plot the shaded uncertainty regions for ensemble predictions
plt.fill_between(prediction_dates_1123, predicted_values_low_1123, predicted_values_high_1123,
                 color='green', alpha=0.5)
plt.fill_between(prediction_dates_1221, predicted_values_low_1221, predicted_values_high_1221,
                 color='green', alpha=0.5, label="Ensemble Prediction Interval")

# Plot the shaded uncertainty regions for baseline predictions
plt.fill_between(prediction_dates_1123, predicted_value_low_1130, predicted_value_high_1130,
                 color='blue', alpha=0.3)
plt.fill_between(prediction_dates_1221, predicted_value_low_1221, predicted_value_high_1221,
                 color='blue', alpha=0.3, label="CESGCN Prediction Interval")

plt.fill_between(prediction_dates_1123, predicted_values_baseline_low_1123, predicted_values_baseline_high_1123,
                 color='yellow', alpha=0.5)
plt.fill_between(prediction_dates_1221, predicted_values_baseline_low_1221, predicted_values_baseline_high_1221,
                 color='yellow', alpha=0.5, label="Baseline Prediction Interval")

# Enhancing visualization
plt.xlabel("Date", fontsize=14)
plt.ylabel("# Flu Hospitalizations", fontsize=14)
plt.title("TX Flu Hospitalization", fontsize=16, fontweight='bold')
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=10, loc="upper left")
plt.grid(True, linestyle="--", alpha=0.5)

# Display the plot
plt.tight_layout()
plt.show()





########################################################

# Define actual prediction timestamps for visualization
prediction_dates_1123 = ['2024-11-30', '2024-12-07', '2024-12-14','2024-12-21']
prediction_dates_1221 = ['2024-12-21', '2024-12-28', '2025-01-04','2025-01-11']

# Extract values from dictionaries
predicted_values_low_1123 = GA_ensemble_low['2024-11-30']
predicted_values_high_1123 = GA_ensemble_high['2024-11-30']
predicted_values_low_1221 = GA_ensemble_low['2024-12-21']
predicted_values_high_1221 = GA_ensemble_high['2024-12-21']

predicted_values_baseline_low_1123 = GA_base_low['2024-11-30']
predicted_values_baseline_high_1123 = GA_base_high['2024-11-30']
predicted_values_baseline_low_1221 = GA_base_low['2024-12-21']
predicted_values_baseline_high_1221 = GA_base_high['2024-12-21']

predicted_value_low_1130 = [20,23,24, 28]
predicted_value_high_1130 = [148,264,354, 422]
predicted_value_low_1221 = [307,211,196,164]
predicted_value_high_1221 = [659,731,1043,1058]
# Plot the main truth line
plt.figure(figsize=(10, 5))
plt.plot(latest_season, GA_truth, marker='o', color='green', label="True Data")

# Plot the shaded uncertainty regions for ensemble predictions
plt.fill_between(prediction_dates_1123, predicted_values_low_1123, predicted_values_high_1123,
                 color='green', alpha=0.5)
plt.fill_between(prediction_dates_1221, predicted_values_low_1221, predicted_values_high_1221,
                 color='green', alpha=0.5, label="Ensemble Prediction Interval")

# Plot the shaded uncertainty regions for baseline predictions
plt.fill_between(prediction_dates_1123, predicted_value_low_1130, predicted_value_high_1130,
                 color='blue', alpha=0.3)
plt.fill_between(prediction_dates_1221, predicted_value_low_1221, predicted_value_high_1221,
                 color='blue', alpha=0.3, label="CESGCN Prediction Interval")

plt.fill_between(prediction_dates_1123, predicted_values_baseline_low_1123, predicted_values_baseline_high_1123,
                 color='yellow', alpha=0.5)
plt.fill_between(prediction_dates_1221, predicted_values_baseline_low_1221, predicted_values_baseline_high_1221,
                 color='yellow', alpha=0.5, label="Baseline Prediction Interval")

# Enhancing visualization
plt.xlabel("Date", fontsize=14)
plt.ylabel("# Flu Hospitalizations", fontsize=14)
plt.title("GA Flu Hospitalization", fontsize=16, fontweight='bold')
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=10, loc="upper left")
plt.grid(True, linestyle="--", alpha=0.5)

# Display the plot
plt.tight_layout()
plt.show()

print('w')

