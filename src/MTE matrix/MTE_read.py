import pickle
import numpy as np
import pandas as pd
import torch

if __name__ == '__main__':
    res_list = list()
    MTE_static_matrices = torch.zeros((200, 200, 52, 52))
    for time in range(12, 201):
        with open('MTE_FLU_4_LAG_12_TIME\MTE_flu_from_{}_to_{}'.format(str(time - 12), str(time)), 'rb') as file:
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
    np.save('MTE_matrices_flu_hosp.npy', MTE_static_matrices)
