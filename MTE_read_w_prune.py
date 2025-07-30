import pickle

import numpy as np
import torch

if __name__ == '__main__':
    res_list = list()
    MTE_static_matrices = torch.zeros((153, 153, 52, 52))
    for time in range(12, 154):
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
                    current_index = time - 1  # 23
                    index_ls = list(range(1, source_process_lag + 1))  # lags start from 1
                    selected_index = [current_index - ind for ind in index_ls]  # 22
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

    # Assume `tensor` is your 4D tensor
    def restrict_outgoing_edges(tensor, max_edges=5):
        num_source_times, num_target_times, num_source_nodes, num_target_nodes = tensor.shape

        for t_s in range(7, num_source_times-4):  # Source time
            for i in range(num_source_nodes):  # Source node
                # Extract all outgoing edges
                for j in range(0,4):
                    weights = tensor[t_s+j, t_s+4, i, :]

                    # Flatten and sort by magnitude
                    flattened = weights.flatten()
                    sorted_indices = np.argsort(-np.abs(flattened))  # Descending order by absolute weight

                    # Retain top `max_edges`
                    top_indices = sorted_indices[:max_edges]
                    # row, col = np.unravel_index(top_indices, weights.shape)
                    mask = np.zeros_like(weights, dtype=bool)
                    mask[top_indices] = True

                    # Update restricted tensor
                    # restricted_weights = np.zeros_like(weights)
                    # restricted_weights[row, col] = weights[row, col]

                    # Assign back to the restricted tensor
                    tensor[t_s+j, t_s + 4, i, ~mask] = 0

        return tensor

    max_edges = 5
    # Apply the function
    restricted_tensor = restrict_outgoing_edges(MTE_static_matrices, max_edges=max_edges)

    MTE_static_matrices = restricted_tensor.numpy()
    np.save('MTE_matrices_flu_hosp_pruned_by_{}.npy'.format(max_edges), MTE_static_matrices)
