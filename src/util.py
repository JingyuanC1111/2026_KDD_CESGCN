import numpy as np
import torch
from torch.autograd import Variable
import pandas as pd

from CESGCN_realtime import train


class dataloader_pipeline(object):
   def __init__(self, data, train, device, horizon, window):
        self.n = data.shape[1]
        self.horizon = horizon
        self.P = window
        self.device = device
        trainset = list()
        for each in train:
            each = np.expand_dims(each, -1)
            trainset.append(each)
        self.feature_dim = 1
        self.train = self.batchify(trainset)
        #self.test1 = self.train[0][-1][1:]
        #self.test2 = self.train[1][-1]
        self.test = self.train[1][-1][0:-1].unsqueeze(0)

        length_train = len(self.train[0])
        self.length = length_train
        val_split = int(length_train * 0.8)
        self.split = val_split
        self.valid_x = self.train[0][val_split:]
        self.valid_y = self.train[1][val_split:]

        self.train[0] = self.train[0][:val_split]
        self.train[1] = self.train[1][:val_split]



   def batchify(self, data):
       n = len(data)
       X = torch.zeros((n, self.P, self.n, self.feature_dim))
       Y = torch.zeros((n, self.horizon, self.n, 1))
       for i in range(n):
           tempX = torch.from_numpy(data[i][0:self.P])
           tempY = torch.from_numpy(data[i][self.P:])
           X[i, :] = tempX
           if tempY.shape[0] == 0:
               break
           Y[i, :] = tempY  # 62
       return [X, Y]

   def get_batches(self, inputs, targets, batch_size):
       length = len(inputs)
       start_idx = 0
       count = 0
       while (start_idx < length):
           X = inputs[start_idx:start_idx + batch_size]
           Y = targets[start_idx:start_idx + batch_size]
           X = X.to(self.device)
           Y = Y.to(self.device)
           count = count + 1
           yield Variable(X), Variable(Y), Variable(torch.tensor((start_idx, start_idx + batch_size)))
           start_idx += batch_size


class Consecutive_dataloader(object):
    def __init__(self, data, train, valid, device, horizon, window):
        self.P = window
        self.h = horizon
        self.training_portion = train
        self.valid_portion = valid
        self.rawdat = data.astype(float)
        self.m, self.n = self.rawdat.shape
        self._normalized()
        if len(self.rawdat.shape) == 2:
            self.training_data = np.expand_dims(self.training_data, -1)
            self.valid_data = np.expand_dims(self.valid_data, -1)
            self.testing_data = np.expand_dims(self.testing_data, -1)
            self.all_data = np.expand_dims(self.all_data, -1)
            self.feature_dim = 1
        else:
            self.feature_dim = self.rawdat.shape[-1]
        # timestamps, num_nodes, feature dimension
        self._split(self.training_data, self.valid_data, self.testing_data)
        self.device = device


    def _normalized(self):
        # normalized by the maximum value of training set
        training_data = self.rawdat[0: int(self.m * self.training_portion)]
        num_nodes,time = training_data.shape[1], training_data.shape[0]
        mean = training_data.mean(axis=0, keepdims=True)
        self.mean = mean
        mean = np.repeat(mean, time, axis=0)
        std = training_data.std(axis=0, keepdims = True)
        self.std = std
        std = np.repeat(std, time, axis=0)
        training_data = (training_data - mean) / std

        # normalize for validation and testing as well
        valid_data = self.rawdat[int(self.m * self.training_portion): int(self.m * (self.training_portion + self.valid_portion))]
        testing_data = self.rawdat[int(self.m * (self.training_portion + self.valid_portion)):]

        mean_valid = np.repeat(self.mean, valid_data.shape[0], axis=0)
        std_valid = np.repeat(self.std, valid_data.shape[0], axis=0)
        valid_data = (valid_data - mean_valid) / std_valid

        mean_test = np.repeat(self.mean, testing_data.shape[0], axis=0)
        std_test = np.repeat(self.std, testing_data.shape[0], axis=0)
        testing_data = (testing_data - mean_test) / std_test

        self.training_data = training_data
        self.valid_data = valid_data
        self.testing_data = testing_data
        concatenated_array = np.concatenate((valid_data, testing_data), axis=0)
        self.all_data = concatenated_array



    def _split(self, train, valid, test):
        self.train = self._batchify(train, self.h)
        self.valid = self._batchify(valid, self.h)
        self.testset = self._batchify(test, self.h)
        self.probabilistic = self._batchify(self.all_data, self.h)
    def _batchify(self, idx_set, horizon):
        n = idx_set.shape[0] - self.P - self.h + 1
        X = torch.zeros((n, self.P, self.n, self.feature_dim))
        Y = torch.zeros((n, horizon,self.n, 1))
        for i in range(n):
            start = i
            tempX = torch.from_numpy(idx_set[start: start+self.P])
            tempY = torch.from_numpy((idx_set[start+self.P:start+self.P+self.h]))
            X[i,:] = tempX
            Y[i,:] = tempY # 62
        return [X, Y]

    def get_batches(self, inputs, targets, batch_size):
        length = len(inputs)
        start_idx = 0
        count = 0
        while (start_idx < length):
            X = inputs[start_idx:start_idx+batch_size]
            Y = targets[start_idx:start_idx+batch_size]
            X = X.to(self.device)
            Y = Y.to(self.device)
            count = count + 1
            yield Variable(X), Variable(Y), Variable(torch.tensor((start_idx, start_idx+batch_size)))
            start_idx += batch_size
def mape_loss(output, target):
    """
    Calculate Mean Absolute Percentage Error (MAPE) loss

    Args:
    output (torch.Tensor): The predictions
    target (torch.Tensor): The ground truth values

    Returns:
    torch.Tensor: MAPE loss
    """
    loss = torch.abs((target - output) / (target+1))
    return torch.mean(loss)

def rmse_loss(output, target):
    mse_loss = torch.nn.MSELoss()
    mse = mse_loss(output, target)
    rmse = torch.sqrt(mse)

    return rmse

class Scaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, tensor):
        # Compute the mean and std along the batch and sequence dimensions
        self.mean = tensor.mean(dim=[0, 1], keepdim=True)
        self.std = tensor.std(dim=[0, 1], keepdim=True)

    def transform(self, tensor):
        # Check if the scaler is fitted
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler has not been fitted.")

        # Perform standard scaling
        return (tensor - self.mean) / self.std

    def restore(self, tensor):
        # Check if the scaler is fitted
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler has not been fitted.")

        # Reverse the scaling
        return tensor * self.std + self.mean

def get_adjacency_matrix(dataset_name):
    '''
    Parameters
    ----------
    distance_df_filename: str, path of the csv file contains edges information

    num_of_vertices: int, the number of vertices

    Returns
    ----------
    A: np.ndarray, adjacency matrix
    '''



    dist_A = pd.read_csv('epidemic_data/geo_adj_matrix/state_distance.csv')
    dist_A = dist_A.iloc[:,1:].values
    if dataset_name == 'covid_hosp':
        A = np.load('COVID_HOSP_MTE_RESULT/matrix_post_processing/MTE_matrices_covid_hosp.npy', allow_pickle=True)
        return A, dist_A
    elif dataset_name == 'covid_case':
        A = np.load('COVID_CASE_MTE_RESULT/matrix_post_processing/MTE_matrices_covid_case.npy', allow_pickle=True)
        return A, dist_A
    else:
        current_A = np.load('FLU_HOSP_MTE_RESULT/matrix_post_processing/MTE_matrices_flu_hosp.npy', allow_pickle=True)
        return current_A, dist_A

def load_from_mte_adj():
    degree_check = np.load('MTE_matrices_flu_hosp.npy')
    ls = list()
    for t in range(11, len(degree_check)):
        t = degree_check[t - 4:t, t, :, :]
        ls.append(t)
    return np.stack(ls, axis=0)
