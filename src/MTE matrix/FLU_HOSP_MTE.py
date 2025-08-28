import pickle
from idtxl.multivariate_te import MultivariateTE
from idtxl.data import Data
import pandas as pd
import os, sys

print(sys.argv[1])
time = int(sys.argv[1])
if __name__ == '__main__':
    cases = pd.read_csv('transferred_hospital_admission_0524.csv')
    cases = cases.iloc[:, 1:] 
    total_timestamps = cases.shape[0]


    print('=================================== Running ' + str(time) + '-th timestamp, total  timestamps')
    # Arrange the data in a 2D array
    data = cases[time-12:time]
    data = data.T  # now it's regions * time

    # Convert this into an IDTxl Data object
    data_idtxl = Data(data, dim_order='ps', normalise=False)  # use readily available data to calculate transfer entropy

    # Initialize the MultivariateTE analysis object
    network_analysis = MultivariateTE()

    # We should be able to check multiple timestamps, and record the transfer entropy of each [i,j] pair
    # Set some parameters
    # print(f'Number of cores available: {multiprocessing.cpu_count()}')

    settings = {'cmi_estimator': 'JidtKraskovCMI',
                'max_lag_sources': 4,
                'min_lag_sources': 1,
                'verbose': False}

    # Run the analysis
    results = network_analysis.analyse_network(settings=settings, data=data_idtxl)

    with open('output/rolling_window_4_lags_with_12_timestamps/MTE_flu_from_{}_to_{}'.format(str(time-12), str(time)), 'wb') as f:
        pickle.dump(results._single_target, f)
