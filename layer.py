from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Batch, Data

'''
class MultiRangeAttention(nn.Module):
    def __init__(self, node_dim, num_layers, device):
        super(MultiRangeAttention, self).__init__()
        self.node_dim = node_dim
        self.num_layers = num_layers
        self.device = device

        # Define learnable parameters
        self.Wa = nn.Parameter(torch.randn(node_dim, node_dim))
        self.ba = nn.Parameter(torch.randn(node_dim))
        self.v = nn.Parameter(torch.randn(node_dim))

    def forward(self, x):
        # x shape: [batch, num_nodes, node_dim, num_layers]

        batch_size, num_nodes, _, num_layers = x.shape

        # Initialize output tensor
        Y = torch.zeros(batch_size, num_nodes, self.node_dim).to(self.device)

        # Process each layer
        for m in range(num_layers):
            # Select the m-th layer for all nodes
            z = x[:, :, :, m]  # Shape: [batch, num_nodes, node_dim]

            # Compute attention scores
            s = torch.tanh(torch.matmul(z, self.Wa) + self.ba)
            s = torch.matmul(s, self.v.unsqueeze(-1)).squeeze(-1)  # Shape: [batch, num_nodes]

            # Apply softmax across layers for each node
            a = F.softmax(s, dim=-1)  # Shape: [batch, num_nodes]

            # Aggregate information with attention scores
            Y += a.unsqueeze(-1) * z

        return Y
'''
class MultiRangeAttention(nn.Module):
    def __init__(self, node_dim, num_layers, device):
        super(MultiRangeAttention, self).__init__()
        self.node_dim = node_dim
        self.num_layers = num_layers
        self.device = device

        # Define learnable parameters
        self.Wa = nn.Parameter(torch.Tensor(node_dim, node_dim))
        self.ba = nn.Parameter(torch.Tensor(node_dim))
        self.v = nn.Parameter(torch.Tensor(node_dim, 1))

        # Parameter initialization
        nn.init.xavier_uniform_(self.Wa)
        nn.init.zeros_(self.ba)
        nn.init.xavier_uniform_(self.v)

    def forward(self, x):
        # x shape: [batch, num_nodes, node_dim, num_layers]
        batch_size, num_nodes, _, num_layers = x.shape

        # Reshape x to process all layers at once: [batch*num_nodes*num_layers, node_dim]
        x_reshaped = x.permute(0, 3, 1, 2).reshape(-1, self.node_dim)
        # B * M * N, d
        # Compute attention scores for all layers at once
        s = torch.tanh(torch.matmul(x_reshaped, self.Wa) + self.ba)
        s = torch.matmul(s, self.v).squeeze(-1).reshape(batch_size, num_nodes, num_layers)

        # Apply softmax across the num_layers dimension
        a = F.softmax(s, dim=2)  # Shape: [batch, num_nodes, num_layers]

        # Aggregate information with attention scores: [batch, num_nodes, node_dim]
        Y = torch.einsum('bnm,bndm->bnd', a, x)

        return Y


class DilatedSpatioTemporalGCN(nn.Module):
    def __init__(self, node_features, dropout_rate, num_layers, device, target, kernel_size, dilation_rates,
                 use_multirange, use_dynamic_adj, use_attention, use_temporal, use_gnn, use_mte):
        super().__init__()
        self.target = target
        self.dilation_rates = dilation_rates
        self.kernel_size = kernel_size
        self.device = device
        self.use_multirange = use_multirange  # directly use the last layer
        self.use_dynamic_adj = use_dynamic_adj  # use geolocation as the predefined adj
        self.use_attention = use_attention  # if not, we skip the attention and directly use the stacked output from
        self.use_temporal = use_temporal  # if not, we use the output from GNN module
        self.use_gnn = use_gnn  # if not, we will only use MTE matrix
        self.temporal_convs = nn.ModuleList()
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout_rate)
        for dilation in dilation_rates:
            self.temporal_convs.append(nn.Conv2d(node_features, node_features, (1, kernel_size), stride=(1, 1),
                                                 dilation=dilation))

        # Initialize GCN layers for each dilation rate
        self.gcn_layers = nn.ModuleList()
        if self.target:
            self.gcn_layers_static = nn.ModuleList()
            self.sig = nn.Sigmoid()
            self.gating_linear = nn.Linear(node_features * 2, node_features)
        for _ in dilation_rates:  # Subsequent layers
            if self.target:
                self.gcn_layers_static.append(GCNConv(in_channels=node_features, out_channels=node_features))
            if self.use_gnn:
                self.gcn_layers.append(GCNConv(in_channels=node_features, out_channels=node_features))

        self.attention_mechanism = MultiRangeAttention(node_features, self.num_layers, self.device)

    def forward(self, node_embeddings, B, static_MTE_matrix, use_MTE):
        '''
        :param node_embeddings: node embeddings for N nodes
        :param B: the interactive matrix ,a d by d matrix, d is the latent dimension
        :param static_MTE_matrix: static MTE matrix from Multivariate Transfer Entropy
        :param alpha: the weight to control the matrix
        :return: the representation of N nodes at timestamp t from each STJCN layer
        '''
        res_aggregation = []
        batch_size, sequence_length, feature_dim, num_nodes = node_embeddings.shape
        # dynamic_adj_matrix = torch.zeros(batch_size, sequence_length, num_nodes, num_nodes).to(self.device)
        #dynamic_adj_matrix = adaptive_matrix
        # Extract U_t (features at the last timestamp)
        U_t = node_embeddings[:, -1, :, :].permute(0, 2, 1)  # Shape: [batch_size, num_nodes, feature_dim]
        dynamic_adj_matrix = list()
        for k in range(sequence_length):
            # Extract U_{t-k}
            U_t_k = node_embeddings[:, sequence_length - 1 - k, :, :].permute(0, 2,
                                                                              1)  # Shape: [batch_size, num_nodes, feature_dim]
            # Compute U_{t-k} * B * U_{t}^Transpose
            intermediate_product = torch.matmul(U_t_k, B)  # Shape: [batch_size, num_nodes, feature_dim]
            product = torch.matmul(intermediate_product,
                                   U_t.transpose(1, 2))  # Shape: [batch_size, num_nodes, num_nodes]
            # Apply phi (ReLU) and softmax
            temp_matrix = F.relu(product)
            temp_matrix = F.softmax(temp_matrix, dim=-1)
            # dynamic_adj_matrix[:, -(k + 1), :, :] = temp_matrix
            dynamic_adj_matrix.append(temp_matrix)
        dynamic_adj_matrix.reverse()
        dynamic_adj_matrix = torch.stack(dynamic_adj_matrix, dim=0)
        dynamic_adj_matrix = dynamic_adj_matrix.transpose(1,0)
        # Compute all dynamic adjacency matrices simultaneously
        # predefined_matrix is adaptive_adj
        predefined = F.softmax(F.relu(static_MTE_matrix), dim=-1)

        # Batch-wise GCN operations for all timestamps
        for layer_idx, temporal_conv in enumerate(zip(self.temporal_convs)):
            new_gcn_outputs = []
            residual = node_embeddings
            x_processed = []
            for time_step in range(sequence_length):
                current_embeddings = node_embeddings[:, time_step]  # Shape: [batch, node_dim, num_nodes]
                # The next is the GNN on adaptive matrix
                if self.use_gnn:
                    # Convert adjacency matrix to edge index and create a batched graph
                    if self.use_dynamic_adj:
                        batched_graph = []
                        for i in range(batch_size):
                            edge_index, _ = dense_to_sparse(dynamic_adj_matrix[i, time_step])
                            edge_index = edge_index.to(self.device)
                            node_embeddings_batch = current_embeddings[i].t().to(self.device)
                            data = Data(x=node_embeddings_batch, edge_index=edge_index)
                            batched_graph.append(data)
                        batched_graph = Batch.from_data_list(batched_graph)
                        # Apply GCNConv on the batched graph
                        gcn_output = self.gcn_layers[layer_idx](batched_graph.x, batched_graph.edge_index)
                        gcn_output = self.dropout(gcn_output)
                        # Reshape the output back to [batch, num_nodes, node_dim]
                        gcn_output = gcn_output.view(batch_size, num_nodes, -1)
                        # The following is the GNN on MTE matrices
                    else:
                        gcn_output = None
                    if use_MTE:
                        static_batched_graph = []
                        for i in range(batch_size):
                            edge_index_static, _ = dense_to_sparse(predefined[i, time_step])
                            edge_index_static = edge_index_static.to(self.device)
                            node_embeddings_batch_static = current_embeddings[i].t().to(self.device)
                            data_static = Data(x=node_embeddings_batch_static, edge_index=edge_index_static)
                            static_batched_graph.append(data_static)
                        static_batched_graph = Batch.from_data_list(static_batched_graph)
                        # Apply GCNConv on the batched graph
                        gcn_output_static = self.gcn_layers_static[layer_idx](static_batched_graph.x,
                                                                              static_batched_graph.edge_index)
                        gcn_output_static = self.dropout(gcn_output_static)
                        gcn_output_static = gcn_output_static.view(batch_size, num_nodes, -1)
                    if self.use_gnn and use_MTE:
                        if self.use_dynamic_adj:
                            concatenated_embeddings = torch.concatenate((gcn_output_static, gcn_output), dim=-1)
                            gcn_output = self.sig(self.gating_linear(concatenated_embeddings))
                        else:
                            gcn_output = gcn_output_static
                    else:
                        gcn_output = gcn_output
                new_gcn_outputs.append(gcn_output)
                # new_gcn_outputs.append(gcn_output_static)
                '''
                if layer_idx == 0:
                    dynamic_adj_matrix = alpha * static_MTE_matrix[time_step] + (1 - alpha) * dynamic_adj_matrix
                '''
            x = torch.stack(new_gcn_outputs)  # input_seq_length * batch size * number_of_nodes * dimension
            x = x.transpose(1, 0)  # [batch, sequence, num_nodes, node_dim]
            if self.use_temporal:
                # Apply padding only on the left
                left_padding = (self.kernel_size - 1) * self.dilation_rates[layer_idx]
                x_padded = F.pad(x, (0, 0, 0, 0, left_padding, 0), "constant", 0)
                x_padded = x_padded.permute(0, 3, 2, 1)
                x_processed = F.relu(self.temporal_convs[layer_idx](x_padded))  # B * D * N * T
                res_aggregation.append(x_processed[:, :, :, -1])  # B,D,N
                x_processed = x_processed.permute(0, 3, 1, 2)  # BTDN
                x_processed = self.dropout(x_processed)
            else:
                x_processed = x.transpose(3,2)
                res_aggregation.append(x_processed[:,-1,:,:])
            # Update node_embeddings for next iteration if not the last layer
            node_embeddings = x_processed + residual  # 2,12,32,49
        if self.use_multirange:
            res_aggregation = torch.stack(res_aggregation, dim=3)
        else:
            Y = res_aggregation[-1]
            return Y
        # Apply attention mechanism to the output of the last temporal convolution layer
        if self.use_attention:
            res_aggregation = res_aggregation.permute(0, 2, 1, 3)
            Y = self.attention_mechanism(res_aggregation)  # INPUT IS [batch, num_nodes, dim, num_layers]
        else:
            Y = torch.sum(res_aggregation, dim=3)
        Y = self.dropout(Y)
        return Y


class UNIFIED_DilatedSpatioTemporalGCN(nn.Module):
    def __init__(self, node_features, dropout_rate, num_layers, device, target, kernel_size, dilation_rates,
                 use_multirange, use_dynamic_adj, use_attention, use_temporal, use_gnn, use_mte, seq_length, num_nodes
                 ,batch_size):
        super().__init__()
        self.target = target
        self.dilation_rates = dilation_rates
        self.kernel_size = kernel_size
        self.device = device
        self.use_multirange = use_multirange  # directly use the last layer
        self.use_dynamic_adj = use_dynamic_adj  # use geolocation as the predefined adj
        self.use_attention = use_attention  # if not, we skip the attention and directly use the stacked output from
        self.use_temporal = use_temporal  # if not, we use the output from GNN module
        self.use_gnn = use_gnn  # if not, we will only use MTE matrix
        self.temporal_convs = nn.ModuleList()
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout_rate)
        self.learnable_dynamic_adj = nn.Parameter(torch.randn(seq_length, num_nodes, num_nodes))
        for dilation in dilation_rates:
            self.temporal_convs.append(nn.Conv2d(node_features, node_features, (1, kernel_size), stride=(1, 1),
                                                 dilation=dilation))

        # Initialize GCN layers for each dilation rate
        self.gcn_layers = nn.ModuleList()
        if self.target:
            self.gcn_layers_static = nn.ModuleList()
            self.sig = nn.Sigmoid()
            self.gating_linear = nn.Linear(node_features, node_features)
        for _ in dilation_rates:  # Subsequent layers
            if self.target:
                self.gcn_layers_static.append(GCNConv(in_channels=node_features, out_channels=node_features))
            if self.use_gnn:
                self.gcn_layers.append(GCNConv(in_channels=node_features, out_channels=node_features))

        self.attention_mechanism = MultiRangeAttention(node_features, self.num_layers, self.device)

    def forward(self, node_embeddings, B, static_MTE_matrix, use_MTE, batch_index, is_training):
        '''
        :param node_embeddings: node embeddings for N nodes
        :param B: the interactive matrix ,a d by d matrix, d is the latent dimension
        :param static_MTE_matrix: static MTE matrix from Multivariate Transfer Entropy
        :param alpha: the weight to control the matrix
        :return: the representation of N nodes at timestamp t from each STJCN layer
        '''
        res_aggregation = []
        batch_size, sequence_length, feature_dim, num_nodes = node_embeddings.shape
        dynamic_adj_matrix = list()
        # Compute adjacency matrix for each time step
        for k in range(sequence_length):
            U_t_k = node_embeddings[:, sequence_length - 1 - k, :, :].permute(0, 2, 1)  # [batch, nodes, features]
            U_t = node_embeddings[:, -1, :, :].permute(0, 2, 1)  # [batch, nodes, features]
            # Compute spatiotemporal adjacency
            intermediate_product = torch.matmul(U_t_k, B)  # [batch, nodes, features]
            product = torch.matmul(intermediate_product, U_t.transpose(1, 2))  # [batch, nodes, nodes]
            temp_matrix = F.relu(product)
            temp_matrix = F.softmax(temp_matrix, dim=-1)  # Normalize adjacency matrix
            dynamic_adj_matrix.append(temp_matrix)
        dynamic_adj_matrix.reverse()
        dynamic_adj_matrix = torch.stack(dynamic_adj_matrix, dim=1)

        if batch_index == 1:
            static_MTE_matrix = torch.tensor(static_MTE_matrix[0], dtype=torch.float32,
                                             device=self.device)
            predefined = F.softmax(F.relu(static_MTE_matrix), dim=-1)
            self.learnable_dynamic_adj = nn.Parameter(predefined.clone())
            batch_dynamic_adj = self.learnable_dynamic_adj.unsqueeze(0).expand(batch_size, -1, -1, -1)
        else:
            with torch.no_grad():
                if len(self.learnable_dynamic_adj.size()) == 3:
                    self.learnable_dynamic_adj.data = self.learnable_dynamic_adj.unsqueeze(0).expand(batch_size, -1, -1, -1).data + \
                                                   dynamic_adj_matrix
                batch_dynamic_adj = self.learnable_dynamic_adj
        # Expand learnable adjacency matrix across batch: [batch_size, seq_length, num_nodes, num_nodes]
        # batch_dynamic_adj = self.learnable_dynamic_adj.unsqueeze(0).expand(batch_size, -1, -1, -1)

        if not is_training:
            batch_dynamic_adj = static_MTE_matrix


        # Batch-wise GCN operations for all timestamps
        for layer_idx, temporal_conv in enumerate(zip(self.temporal_convs)):
            new_gcn_outputs = []
            residual = node_embeddings
            x_processed = []
            for time_step in range(sequence_length):
                current_embeddings = node_embeddings[:, time_step]  # Shape: [batch, node_dim, num_nodes]
                # The next is the GNN on adaptive matrix
                if self.use_gnn:
                    # Convert adjacency matrix to edge index and create a batched graph
                    if self.use_dynamic_adj:
                        batched_graph = []
                        for i in range(batch_size):
                            edge_index, _ = dense_to_sparse(batch_dynamic_adj[i, time_step])
                            edge_index = edge_index.to(self.device)
                            node_embeddings_batch = current_embeddings[i].t().to(self.device)
                            data = Data(x=node_embeddings_batch, edge_index=edge_index)
                            batched_graph.append(data)
                        batched_graph = Batch.from_data_list(batched_graph)
                        # Apply GCNConv on the batched graph
                        gcn_output = self.gcn_layers[layer_idx](batched_graph.x, batched_graph.edge_index)
                        gcn_output = self.dropout(gcn_output)
                        # Reshape the output back to [batch, num_nodes, node_dim]
                        gcn_output = gcn_output.view(batch_size, num_nodes, -1)
                        # The following is the GNN on MTE matrices
                    else:
                        gcn_output = None
                    if self.use_gnn and use_MTE:
                        if self.use_dynamic_adj:
                            gcn_output = self.sig(self.gating_linear(gcn_output))
                    else:
                        gcn_output = gcn_output
                new_gcn_outputs.append(gcn_output)
                # new_gcn_outputs.append(gcn_output_static)
                '''
                if layer_idx == 0:
                    dynamic_adj_matrix = alpha * static_MTE_matrix[time_step] + (1 - alpha) * dynamic_adj_matrix
                '''
            x = torch.stack(new_gcn_outputs)  # input_seq_length * batch size * number_of_nodes * dimension
            x = x.transpose(1, 0)  # [batch, sequence, num_nodes, node_dim]
            if self.use_temporal:
                # Apply padding only on the left
                left_padding = (self.kernel_size - 1) * self.dilation_rates[layer_idx]
                x_padded = F.pad(x, (0, 0, 0, 0, left_padding, 0), "constant", 0)
                x_padded = x_padded.permute(0, 3, 2, 1)
                x_processed = F.relu(self.temporal_convs[layer_idx](x_padded))  # B * D * N * T
                res_aggregation.append(x_processed[:, :, :, -1])  # B,D,N
                x_processed = x_processed.permute(0, 3, 1, 2)  # BTDN
                x_processed = self.dropout(x_processed)
            else:
                x_processed = x.transpose(3,2)
                res_aggregation.append(x_processed[:,-1,:,:])
            # Update node_embeddings for next iteration if not the last layer
            node_embeddings = x_processed + residual  # 2,12,32,49
        if self.use_multirange:
            res_aggregation = torch.stack(res_aggregation, dim=3)
        else:
            Y = res_aggregation[-1]
            return Y
        # Apply attention mechanism to the output of the last temporal convolution layer
        if self.use_attention:
            res_aggregation = res_aggregation.permute(0, 2, 1, 3)
            Y = self.attention_mechanism(res_aggregation)  # INPUT IS [batch, num_nodes, dim, num_layers]
        else:
            Y = torch.sum(res_aggregation, dim=3)
        Y = self.dropout(Y)
        return Y
