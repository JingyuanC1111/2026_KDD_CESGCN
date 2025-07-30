import argparse
import os
from operator import index
from os.path import split
from datetime import date

import numpy as np
import json
import pandas as pd
import torch
import torch.nn as nn
from net import MTE_STJGC
import networkx as nx
from util import *
from trainer import Optim
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cpu', help='')
parser.add_argument('--data', type=str, default='flu_hosp', choices=['covid_case', 'covid_hosp'
    , 'flu_hosp'], help='data path')
parser.add_argument('--lr', type=float, default=0.02, help='learning rate')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--seq_in_len', type=int, default=4, help='input sequence length')
parser.add_argument('--horizon', type=int, default=5)
parser.add_argument('--save', type=str, default='model/model.pt',
                    help='path to save the final model')
parser.add_argument('--version', type=str, default='V1')
parser.add_argument('--num_nodes', type=int, default=52, help='number of nodes/variables')
parser.add_argument('--num_MC_dropout', type=int, default=100,
                    help='paramater for case forecasting, how many times does '
                         'MC dropout predict')
parser.add_argument('--node_dim', type=int, default=12, help='dim of nodes')
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--in_dim', type=int, default=1, help='inputs dimension')
parser.add_argument('--layers', type=int, default=2, help='number of layers')
parser.add_argument('--weight_decay', type=float, default=0.02, help='weight decay rate')
parser.add_argument('--clip', type=int, default=5, help='clip')
parser.add_argument('--dilation_rates', type=list, default=[1,2], help='dilation of each STJGC layer')
parser.add_argument('--L1Loss', type=bool, default=True)
parser.add_argument('--epochs', type=int, default=100, help='')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--dropout_rate', type=float, default=0.2, help='dropout rate')
parser.add_argument('--multivariate_TE_enhanced', type=bool, default=True, help='Multivariate Transfer Entropy')
parser.add_argument('--early_stopping', type=bool, default=True)
parser.add_argument('--kernel_size', type=int, default=2)
parser.add_argument('--probabilistic', type=bool, default=True, help='probabilistic forecast or point-wise forecast')
parser.add_argument('--use_gnn', type=bool, default=True, help='Whether we use GNN')
parser.add_argument('--use_temporal', type=bool, default=True, help='Whether we use causal dilated')
parser.add_argument('--use_attention', type=bool, default=True, help='Whether we use attention')
parser.add_argument('--use_dynamic_adj', type=bool, default=True, help='Whether we use dynamic spatio-temporal matrix')
parser.add_argument('--use_multirange', type=bool, default=True, help='Whether we use multi-range to calculate')
parser.add_argument('--use_MTE', type=bool, default=True, help='Whether we use MTE as starting point')
parser.add_argument('--patience', type=int, default=20, help='patience of training')
parser.add_argument('--unified_graph', type=bool, default=False, help='Decide whether takes MTE as initialization on '
                                                                      'graph')
args = parser.parse_args()
torch.set_num_threads(3)


def evaluate(X, Y, model, criterion, starting_ind, ending_ind, scaler_min, scaler_max):
    model.eval()
    total_loss = 0
    X = X.transpose(2, 3)
    index_MTE_matrices = [starting_ind, ending_ind]
    '''
    with torch.no_grad():
        output = model(X, index_MTE_matrices).squeeze(-1)
    original_Y = Y.transpose(1, 2).squeeze(-1)
    '''
    with torch.no_grad():
        output = model(X, index_MTE_matrices, 0, False).squeeze(-1)
    original_Y = Y.transpose(1, 2).squeeze(-1)
    output = torch.tensor(inverse_transform(output,scaler_min, scaler_max))
    original_Y =torch.tensor(inverse_transform(original_Y,scaler_min, scaler_max))


    if criterion is None:
        loss = mape_loss(output, original_Y)
    else:
        loss = criterion(output, original_Y)
    total_loss += loss.item()
    return total_loss


def train(data, X, Y, model, optim, batch_size, criterion, device, scaler_min, scaler_max):
    model.train()
    torch.autograd.set_detect_anomaly(True)
    total_loss = 0
    idx_count = 0
    for X_batch, Y_batch, batch_index in data.get_batches(X, Y, batch_size):
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
        idx_count = idx_count + 1
        model.zero_grad()
        X_batch = X_batch.transpose(2, 3)  # batch * input_length * 1 * num_nodes
        if len(X_batch) != batch_size:
            index_MTE_matrices = [(idx_count - 1) * batch_size, len(X)]
        else:
            index_MTE_matrices = [(idx_count - 1) * batch_size, idx_count * batch_size]

        #output = model(X_batch, index_MTE_matrices).squeeze(-1)
        #original_Y = Y_batch.transpose(1, 2).squeeze(-1)

        output = model(X_batch, index_MTE_matrices, idx_count, True).squeeze(-1)
        output = inverse_transform(output, scaler_min, scaler_max)# Assuming `scaler` supports PyTorch tensors directly
        original_Y = inverse_transform(Y_batch.transpose(1, 2).squeeze(-1), scaler_min, scaler_max)

        if criterion is None:
            loss = mape_loss(output, original_Y)
        else:
            loss = criterion(output, original_Y)
        total_loss += loss.item()
        loss.backward()
        optim.step()
    return total_loss


def mean_absolute_percentage_error(y_true, y_pred):
    """
    Calculate Mean Absolute Percentage Error (MAPE).

    Parameters:
    - y_true (numpy array or list): Array of true values.
    - y_pred (numpy array or list): Array of predicted values.

    Returns:
    - float: MAPE value.
    """
    # Convert to numpy arrays for ease of computation
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Ensure non-zero true values to avoid division by zero
    non_zero_mask = y_true != 0
    y_true = y_true[non_zero_mask]
    y_pred = y_pred[non_zero_mask]

    # Calculate MAPE
    raw_mape = np.abs((y_true - y_pred) / y_true) * 100
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return raw_mape, mape


def all_metrics(true, pred):
    time = true.size(2)
    samples = true.size(0)
    nodes = true.size(1)
    MAE = list()
    rmse_list = list()
    mape_list = list()
    for time_idx in range(time):
        true_time = true[:, :, time_idx]
        pred_time = pred[:, :, time_idx]
        MAE_time = torch.abs(true_time - pred_time)
        MAE_time = torch.mean(MAE_time, dim=0)
        MAE.append(torch.mean(MAE_time).numpy())
        rmse = np.sqrt(mean_squared_error(true_time.numpy(), pred_time.numpy()))
        raw_mape, mape = mean_absolute_percentage_error(true_time.numpy(), pred_time.numpy())
        rmse_list.append(rmse)
        mape_list.append(mape)

    return np.mean(MAE), np.mean(rmse_list), np.mean(mape_list)

def inverse_transform(tensor, scaler_min, scaler_max):
    scaler_min = scaler_min.view(1, 52, 1)
    scaler_max = scaler_max.view(1, 52, 1)
    return tensor * (scaler_max - scaler_min) + scaler_min
def predict(best_model, testX, adj, scaler_min, scaler_max):
    best_model.train()
    loss = dict()
    loss['mae'] = list()
    loss['mape'] = list()
    X = testX
    X = X.transpose(2, 3)
    data_index = [len(adj) - 1, len(adj)]
    res = list()
    with torch.no_grad():
        output = best_model(X, data_index, 0, False).squeeze(-1)
        # output = np.transpose(output).reshape(1, -1)
        res = inverse_transform(output, scaler_min, scaler_max)
    '''
    while horizon:
        with torch.no_grad():
            output = best_model(X, data_index).squeeze().numpy()
            output = np.transpose(output).reshape(1, -1)
            next_output = torch.tensor(output).unsqueeze(0).unsqueeze(0)
        output = torch.tensor(scaler.inverse_transform(output))
        res.append(output.squeeze())
        new_input = torch.concatenate((X[:,1:,:,:],next_output),dim=1)
        horizon = horizon - 1
        X = new_input
    '''
    # res = torch.stack(res, dim=1)

    return res


def reframe(raw_data, data, train_window, forecasting_window):
    processed_data = list()
    raw_data_list = list()
    for i in range(12, data.shape[0] + 1):
        starting = i - train_window - 1
        ending = starting + train_window + forecasting_window
        if ending > len(data):
            break
        processed_data.append(data[starting:ending, :])
        raw_data_list.append(raw_data.iloc[starting:ending, :])
    trainset = processed_data

    return trainset


def main(ref_date, fct_date, train_window, forecasting_window):
    df = pd.read_csv('CDC_DATA/target-hospital-admissions_0524.csv')
    removable = ['US']
    df['date'] = pd.to_datetime(df['date']).dt.date
    df = df[~df['location'].isin(removable)]
    df['location'] = df['location'].astype(int).astype(str).str.zfill(2)
    available_data = df.pivot(index='location', columns='date', values='value')

    # available_data = available_data[:,:-4]
    available_data = available_data.T
    available_data.fillna(0, inplace=True)
    available_data = available_data.iloc[1:,:]
    # available_data.to_csv('CDC_DATA/transferred_hospital_admission_0524.csv')
    scaler = MinMaxScaler()
    scaler.fit(available_data.values)

    scaler_min = torch.tensor(scaler.data_min_, dtype=torch.float32, device=args.device)
    scaler_max = torch.tensor(scaler.data_max_, dtype=torch.float32, device=args.device)

    # Reimplement the inverse transform function


    processed_data = scaler.transform(available_data.values)

    # available_data.to_csv('time_series_US.csv')
    regions = available_data.columns.values
    # load MTE_matrices here
    training_set = reframe(available_data, processed_data, train_window, forecasting_window)
    # raw_train.to_csv('raw_data_for_MTE.csv')
    # get corresponding data for valid MTE, to guarantee valid MTE using idtxl, we need to set the starting
    # timestamp at least twice as the lags.
    '''
    file_path = "time_range_file.txt"
    # Writing the numbers to the file
    with open(file_path, "w") as file:
        for number in range(staring_time, len(raw_train)):
            file.write(f"{number}\n")
    '''
    # adj, distance_mx = get_adjacency_matrix(args.data) # load FLU_hosp MTE matrices

    # We make a toy sample first
    sample_size = len(training_set)
    num_regions = available_data.shape[1]
    adj = load_from_mte_adj()
    temp = adj[-1]

    # adj = np.random.rand(sample_size + len(validation_set) + forecasting_window, train_window, num_regions, num_regions)

    device = torch.device(args.device)
    adj = torch.from_numpy(adj)
    adj = adj.float()
    adj = adj.to(device)
    metrics = ['MSE']
    res = dict()
    res['mape'] = list()
    res['mae'] = list()
    mc_num = 50
    # Perform normalization
    # Data = Consecutive_dataloader(available_data, 0.8, 0.2, args.device, args.horizon, args.seq_in_len)
    Data = dataloader_pipeline(processed_data, training_set, args.device, args.horizon,
                               args.seq_in_len)

    for iter_index in range(0,10):
        args.version = 'V'+str(iter_index+1)
        for metric in metrics:
            # Restore: X_train_original = X_train_normalized * (max_val - min_val) + min_val
            model = MTE_STJGC(args.batch_size, args.kernel_size, args.dilation_rates, args.dropout_rate, device,
                              args.data, args.num_nodes, args.horizon, args.use_MTE, args.use_multirange
                              , args.use_dynamic_adj, args.use_attention, args.use_temporal, args.use_gnn,
                              adj, args.unified_graph, node_dim=args.node_dim, seq_length=args.seq_in_len,
                              in_dim=args.in_dim, layers=args.layers)
            model = model.to(args.device)

            nParams = sum([p.nelement() for p in model.parameters()])
            print('Number of model parameters is', nParams, flush=True)

            if metric == 'MAE':
                criterion = nn.L1Loss().to(args.device)
            else:
                criterion = nn.MSELoss()  # use mape function instead
            '''
            optim = Optim(
                model.parameters(), args.optim, args.lr, args.clip, lr_decay=args.weight_decay
            )
            '''
            optim = Optim(
                model.parameters(), args.optim, args.lr, args.clip, lr_decay=args.weight_decay
            )

            print('begin training')

            best_model_path = '0503/best_model_0426_CESGCN_LR_0.02_2025-04-26_quantile_0.2_dim_12_onego_100_epochs_MSE_weight_decay_0.02_V13.pth'

            best_model = MTE_STJGC(args.batch_size, args.kernel_size, args.dilation_rates, args.dropout_rate, device,
                                   args.data, args.num_nodes, args.horizon, args.use_MTE, args.use_multirange
                                   , args.use_dynamic_adj, args.use_attention, args.use_temporal, args.use_gnn,
                                   adj, args.unified_graph, node_dim=args.node_dim, seq_length=args.seq_in_len, in_dim=args.in_dim,
                                   layers=args.layers)
            best_model = best_model.to(args.device)
            best_model.load_state_dict(torch.load(best_model_path))

            res = list()

            for mc in range(0, mc_num):
                point_forecasts = predict(best_model, Data.test, adj,
                                          scaler_min, scaler_max)
                res.append(point_forecasts[0])

        predict_all = list()
        horizon = 5
        r = mc_num * horizon
        for idx, region in enumerate(regions):
            predict_vec = np.zeros([mc_num * horizon, 5])
            predict_vec[:r, 0:1] = np.array([region] * r).reshape(-1, 1)
            predict_vec[:r, 1:2] = np.array([0] * r).reshape(-1, 1)
            for mc in range(mc_num):
                pt_pd = res[mc][idx,:]
                r3 = horizon
                predict_vec[mc * r3:(mc + 1) * r3, 2:3] = np.array(range(horizon)).reshape(-1, 1)
                predict_vec[mc * r3:(mc + 1) * r3, 3:4] = np.array([mc] * horizon).reshape(-1, 1)
                predict_vec[mc * r3:(mc + 1) * r3, 4:5] = np.array(pt_pd).reshape(-1, 1)
            predict_all.append(predict_vec)
        result = np.vstack(predict_all)
        pd_df = pd.DataFrame(result, columns=['location', 'date', 'ahead', 'mc', 'value'])
        pd_df['location'] = pd_df['location'].astype(int).astype(str).str.zfill(2)
        # pd_df.to_csv('prediction-hosp.csv', mode='a', index=False)
        # file = 'prediction-hosp.csv'
        pdf = pd_df

        pdf['reference_date'] = ref_date
        pdf['ahead'] = pdf['ahead'].astype(int)
        pdf['fct_date'] = pdf.apply(lambda x: fct_date[int(x['ahead'])], axis=1)
        fct = pdf.groupby(['location', 'fct_date', 'ahead', 'reference_date']).value.mean().reset_index()
        # fct = fct.sort_values(by=["location", "ahead"], ascending=[True, True])
        fct['fct_std'] = pdf.groupby(['location', 'fct_date', 'ahead', 'reference_date']).value.std().values
        fct.loc[:, 'horizon'] = (pd.to_datetime(fct['fct_date']) - pd.to_datetime(fct['reference_date'])).dt.days // 7
        fct['reference_date'] = ref_date
        fct['method'] = 'CESGCN'
        fct = fct.drop('ahead', axis=1)
        # fct.to_csv('prediction-fmt.csv', header=True, index=False, mode='w')

        quantiles = [0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8,
                     0.85, 0.9, 0.95, 0.975, 0.99]
        quantile_res = list()
        fct['location'] = pd.to_numeric(fct['location'], errors='coerce')
        fct['location'] = fct['location'].astype(int).astype(str).str.zfill(2)
        pd_df['location'] = pd.to_numeric(pd_df['location'], errors='coerce')
        pd_df['location'] = pd_df['location'].astype(int).astype(str).str.zfill(2)
        pd_df['ahead'] = pd.to_numeric(pd_df['ahead'], errors='coerce')

        for idx, quantile in enumerate(quantiles):
            quantile_df = np.zeros([horizon * num_regions, 12])
            expanded_list = [x for x in regions for _ in range(horizon)]
            quantile_df[:, 0:1] = np.array(expanded_list).reshape(-1, 1)
            quantile_regions = list()
            for idx, region in enumerate(regions):
                temp_value = fct[fct['location'] == region]['value'].values
                temp_std = fct[fct['location'] == region]['fct_std'].values
                quantile_df[idx * horizon:(idx + 1) * horizon, 3] = temp_value
                quantile_df[idx * horizon:(idx + 1) * horizon, 4] = temp_std
                quantile_df[idx * horizon:(idx + 1) * horizon, 5] = np.arange(-1, 4)
                quantile_res_temp = list()
                for ahead in range(0, horizon):
                    temp_value_quantiles = \
                        pd_df[pd_df['location'] == region][pd_df['ahead'] == ahead]['value'].values
                    quantile_number = np.quantile(temp_value_quantiles, quantile)
                    quantile_res_temp.append(quantile_number)
                quantile_regions.extend(quantile_res_temp)
            quantile_df[:, -2] = quantile
            quantile_df[:, -1] = quantile_regions
            quantile_df = pd.DataFrame(quantile_df,
                                       columns=['location', 'target_end_date', 'avl_date', 'point', 'fct_std', 'horizon',
                                                'reference_date',
                                                'method', 'target', 'output_type', 'output_type_id', 'value'])
            quantile_df['target_end_date'] = fct_date * num_regions
            quantile_df['target_end_date'] = pd.to_datetime(quantile_df['target_end_date']).dt.date
            quantile_df['avl_date'] = fct_date[0]
            quantile_df['reference_date'] = fct_date[1]
            quantile_df['target'] = ['wk inc flu hosp'] * num_regions * horizon
            quantile_df['output_type'] = ['quantile'] * num_regions * horizon
            quantile_res.append(quantile_df)

        quantiled_version = pd.concat(quantile_res, axis=0)
        quantiled_version['location'] = quantiled_version['location'].astype(int).astype(str).str.zfill(2)
        quantiled_version.to_csv('0531/CESGCN_LR_{}_{}_quantile_{}_dim_{}_onego_{}_epochs_MSE_weight_decay_{}_PREDICT_{}.csv'.
                                 format(args.lr,ref_date,args.dropout_rate,args.node_dim,
                                        args.epochs,args.weight_decay,args.version), header=True, index=False, mode='w')

        fct.to_csv('0531/CESGCN_LR_{}_{}_{}_dim_{}_onego_{}_MSE_weight_decay_{}_PREDICT_{}.csv'.format(args.lr,ref_date, args.dropout_rate,args.node_dim,args.epochs,args.weight_decay,args.version), header=True, index=False, mode='w')


if __name__ == "__main__":
    with open('cfg/cfg') as json_file:
        data = json.load(json_file)
    ref_date = data['reference_date']
    fct_date = data['fct_date']
    print('Preparing ground truth.')


    def get_week(date, weeks):
        for week in weeks:
            s, e = week.split('_')
            if s <= date <= e:
                return week


    # Confirmed cases
    temp_out_dir = data['temp_output_path']
    out_dir = data['output_path']
    out_file = temp_out_dir + '/prediction-hosp.csv'
    if os.path.isfile(out_file):
        os.remove(out_file)
    train_window = args.seq_in_len
    horizon = args.horizon
    main(ref_date, fct_date, train_window, horizon)
