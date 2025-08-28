from layer import *
import torch
import numpy as np

class MTE_STJGC(nn.Module):
    def __init__(self, batch_size, kernel_size, dilation_rates, dropout_rate, device, target, num_nodes, horizon,
                 use_MTE, use_multirange
                 , use_dynamic_adj, use_attention, use_temporal, use_gnn, adj,unified_graph,
                 node_dim=16, seq_length=12, in_dim=1, layers=4, ):
        super(MTE_STJGC, self).__init__()
        # self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.target = target
        self.device = device
        self.num_nodes = num_nodes
        self.use_MTE = use_MTE
        self.use_multirange = use_multirange
        self.use_dynamic_adj = use_dynamic_adj
        self.use_attention = use_attention
        self.use_temporal = use_temporal
        self.use_gnn = use_gnn
        self.predefined = adj
        self.horizon = horizon
        self.layers = layers
        self.skip_convs = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.unified_graph = unified_graph
        self.feature_transform = nn.Conv2d(in_channels=in_dim, out_channels=node_dim, kernel_size=1)
        self.softmax = nn.Softmax()
        self.relu = nn.ReLU()
        self.multi_interaction = torch.nn.Parameter(torch.Tensor(node_dim, node_dim))
        self.seq_length = seq_length
        self.kernel_size = kernel_size
        self.dilation_rates = dilation_rates
        self.STJGC = DilatedSpatioTemporalGCN(node_dim, self.dropout_rate, self.layers, self.device, self.target,
                                              self.kernel_size, self.dilation_rates,
                                              self.use_multirange, self.use_dynamic_adj, self.use_attention,
                                              self.use_temporal, self.use_gnn, self.use_MTE)
        if self.unified_graph:
            self.STJGC_UNIFIED = UNIFIED_DilatedSpatioTemporalGCN(node_dim, self.dropout_rate, self.layers, self.device, self.target,
                                              self.kernel_size, self.dilation_rates,
                                              self.use_multirange, self.use_dynamic_adj, self.use_attention,
                                              self.use_temporal, self.use_gnn, self.use_MTE, seq_length, num_nodes, batch_size)

        self.output_module = nn.ModuleList()
        for _ in range(self.horizon):
            linear1 = nn.Linear(node_dim, node_dim)

            # Second linear layer (W2, b2)
            linear2 = nn.Linear(node_dim, 1)

            # Combine the layers with an activation function in between
            module = nn.Sequential(
                linear1,
                nn.ReLU(),  # Example activation function
                linear2,
            )
            self.output_module.append(module)

    def forward(self, input, index_of_MTE_matrices, idx_count, is_training):
        seq_len = input.size(1)
        input = input.float()
        '''
        MTE_static_matrices = self.softmax(MTE_static_matrices)

        MTE_static_matrices = torch.transpose(MTE_static_matrices, 1, 2, 0)
        '''
        MTE_static_matrices = self.predefined
        MTE_matrix_list_for_the_batch = list()
        index_of_MTE_matrices = np.arange(index_of_MTE_matrices[0], index_of_MTE_matrices[1])
        MTE_matrix_list_for_the_batch.extend(MTE_static_matrices[index_of_MTE_matrices])
        MTE_matrix_list_for_the_batch = torch.stack(MTE_matrix_list_for_the_batch, dim=0)
        # matrix normalization to guarantee that the sum of each row is 1
        MTE_static_matrices = MTE_matrix_list_for_the_batch
        # we process MTE matrix here, the data_index shows
        input = input.permute(0, 2, 1, 3)
        # Pass through MLP
        embedding_U = self.feature_transform(input)
        # Reshape x back to: [batch_size, seq_length, new_feature_dim, num_nodes]
        embedding_U = embedding_U.permute(0, 2, 1, 3)
        # Forward pass through the model
        # adaptive_matrix = self.adaptive_adj_matrix
        if self.unified_graph:
            output = self.STJGC_UNIFIED(embedding_U, self.multi_interaction, MTE_static_matrices, self.use_MTE, idx_count, is_training)
        else:
            output = self.STJGC(embedding_U, self.multi_interaction, MTE_static_matrices, self.use_MTE)

        # OUTPUT Predictions
        res_list = []
        for idx in range(self.horizon):
            res_list.append(self.output_module[idx](output))
        res_list = torch.stack(res_list, dim=2)
        return res_list
