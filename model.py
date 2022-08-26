import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.nn.models import GIN, GCN
# from dgl.nn.pytorch.conv import GraphConv
import numpy as np
import dgl


# class GCNLayer(nn.Module):
#     def __init__(self, in_features, out_features, bias=True):
#         super(GCNLayer, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.weight = Parameter(torch.FloatTensor(in_features, out_features))
#         if bias:
#             self.bias = Parameter(torch.FloatTensor(out_features))
#         else:
#             self.register_parameter('bias', None)
#
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         stdv = 1. / math.sqrt(self.weight.size(1))
#         self.weight.data.uniform_(-stdv, stdv)
#         if self.bias is not None:
#             self.bias.data.uniform_(-stdv, stdv)  # bias均匀分布随机初始化
#
#     def forward(self, input, adj):
#         # Calculate the laplacian matrix
#         # D = torch.sum(adj, 0)
#         # D = torch.diag(torch.pow(D, -1.0/2))
#         # adj_norm = torch.mm(torch.mm(D, adj), D)
#
#         support = torch.mm(input, self.weight)
#         output = torch.spmm(adj, support)
#
#         if self.bias is not None:
#             output = output + self.bias  # 返回（系数*输入*权重+偏置）
#         else:
#             output = output  # 返回（系数*输入*权重）无偏置
#
#         return F.relu(output)
#
#
# class GCN(nn.Module):
#     def __init__(self, in_features, hidden_features, num_layers, out_features):
#         super(GCN, self).__init__()
#         self.GCNConv_list = nn.ModuleList()
#         self.num_layers = num_layers
#         if num_layers == 1:
#             self.GCNConv_list.append(GCNLayer(in_features, out_features))
#         else:
#             if num_layers > 0:
#                 self.GCNConv_list.append(GCNLayer(in_features, hidden_features))
#             for i in range(num_layers-2):
#                 self.GCNConv_list.append(GCNLayer(hidden_features, hidden_features))
#             if num_layers > 1:
#                 self.GCNConv_list.append(GCNLayer(hidden_features, out_features))
#
#     def forward(self, h, adj):
#         for i in range(self.num_layers):
#             h = self.GCNConv_list[i](h, adj)
#         return h


# class LMPool(nn.Module):
#     def __init__(self, device, num_landmarks, hidden_dim):
#         super(LMPool, self).__init__()
#         self.device = device
#         self.hidden_dim = hidden_dim
#         self.num_landmarks = num_landmarks
#         self.feat_landmars = Parameter(torch.FloatTensor(self.num_landmarks * (self.hidden_dim - 1) // 2))
#
#         self.model_init()
#
#         # attention
#         self.attention_W = nn.Linear(hidden_dim, hidden_dim, bias=False)
#         self.attention_alpha = nn.Linear(hidden_dim*2, 1, bias=False)
#
#         def model_init(self):
#             # self.adj_landmark_element.data.uniform_(-1, 1)
#             self.adj_landmark_element.data.uniform_(0, 1)


class SDPool(nn.Module):
    def __init__(self, device, size_landmark_graphs, num_landmark_graphs, hidden_dim, num_gcn_layers):
        super(SDPool, self).__init__()
        self.device = device
        # self.num_coarsen_graphs = num_coarsen_graphs
        self.hidden_dim = hidden_dim
        self.size_landmark_graphs = size_landmark_graphs
        self.num_landmark_graphs = num_landmark_graphs
        self.adj_landmark_element_list = nn.ParameterList()
        encoder = GCN
        # encoder = GIN
        self.landmark_gcn_layer = encoder(self.size_landmark_graphs, self.hidden_dim, num_gcn_layers, self.hidden_dim)

        # self.landmark_gcn_layer_list = nn.ModuleList()
        for i in range(self.num_landmark_graphs):
            self.adj_landmark_element_list.append(Parameter(torch.FloatTensor(self.size_landmark_graphs * (self.size_landmark_graphs - 1) // 2)))
        # self.landmark_gcn_layer_list = encoder(self.hidden_dim, self.hidden_dim, num_gcn_layers, self.hidden_dim)
        #     self.landmark_gcn_layer_list.append(encoder(self.size_landmark_graphs, self.hidden_dim, num_gcn_layers, self.hidden_dim))


        self.model_init()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

        # attention
        self.attention_W = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.attention_alpha = nn.Linear(hidden_dim*2, 1, bias=False)

        # assignment-only
        self.assign_W = nn.Linear(hidden_dim, 8)
        self.assign_soft = nn.Softmax(dim=1)
        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.2)

        # self.gcn_layer = GraphConv(args.input_dim, args.hidden_dim, activation=F.relu)
        # self.gcn_layer_2nd = GraphConv(args.hidden_dim, args.hidden_dim, activation=F.relu)
        # self.gcn_layer_3rd = GraphConv(args.hidden_dim, args.hidden_dim, activation=F.relu)
        # self.classify = nn.Linear(args.hidden_dim, args.out_dim)
        # self.final_dropout = args.final_dropout

    def model_init(self):
        # self.adj_landmark_element.data.uniform_(-1, 1)
        for i in range(self.num_landmark_graphs):
            self.adj_landmark_element_list[i].data.uniform_(0, 1)

    def lap_eigen_cal(self, g_adj):
        # adj = g_adj.to('cpu')
        # deg = torch.sum(adj, 0).to('cpu')
        #
        # deg_ng_half = torch.pow(deg, -1 / 2)
        # laplacian_g = torch.eye(len(deg_ng_half)).to('cpu') - torch.diag(deg_ng_half).matmul(adj).matmul(torch.diag(deg_ng_half))

        adj = g_adj
        deg = torch.sum(adj, 0)
        deg_ng_half = torch.pow(deg, -1 / 2)
        laplacian_g = torch.eye(len(deg_ng_half)).to(self.device) - torch.diag(deg_ng_half).matmul(adj).matmul(torch.diag(deg_ng_half))


        # laplacian_g = torch.eye(len(g_adj)).to(self.device) - torch.diag(torch.pow(torch.sum(g_adj, 0), -1 / 2)).matmul(g_adj).matmul(torch.diag(torch.pow(torch.sum(g_adj, 0), -1 / 2)))

        # t = torch.isnan(laplacian_g)
        # if True in t:
        #     print('ok')

        # u, _ = torch.eig(laplacian_g, eigenvectors=True)
        # u = torch.linalg.eigvals(laplacian_g).real

        # ensure the symmetric
        laplacian_g = (laplacian_g + torch.transpose(laplacian_g, 0, 1))/2

        u = torch.linalg.eigvalsh(laplacian_g)
        # u = u[:, 0].reshape(-1)
        u, _ = torch.sort(u, descending=True)
        return u.to(self.device)

    def spectral_distance(self, a, b):
        sp_dist = torch.sum(torch.abs(a-b))
        return sp_dist

    # def avoid_isolated(self, adj):
    #     deg = torch.sum(adj, 0)
    #     indices = deg < 0.01
    #     if True in indices:
    #         zero_indices = torch.nonzero(indices)
    #         for idx in zero_indices:
    #             # random.choice()
    #             t = list(range(len(adj)))
    #             t.remove(idx.item())
    #             random_neighbor_idx = random.choice(t)
    #             random_value = random.random()
    #             adj[idx.item()][random_neighbor_idx] = adj[idx.item()][random_neighbor_idx] + random_value
    #             adj[random_neighbor_idx][idx.item()] = adj[idx.item()][random_neighbor_idx]
    #
    #     return adj

    def forward(self, h, batch_g):
        e_landmark_list = list()
        batch_num_nodes = batch_g.batch_num_nodes()
        for i in range(self.num_landmark_graphs):
            # generate the landmark adj
            adj_landmark = torch.zeros(self.size_landmark_graphs, self.size_landmark_graphs).to(self.device)
            idx = torch.triu_indices(self.size_landmark_graphs, self.size_landmark_graphs, 1)
            # adj_landmark[idx[0], idx[1]] = F.relu(self.adj_landmark_element)
            adj_landmark[idx[0], idx[1]] = self.adj_landmark_element_list[i]
            adj_landmark = adj_landmark + torch.transpose(adj_landmark, 0, 1)
            adj_landmark = adj_landmark.to(self.device)
            features_hidden = torch.eye(self.size_landmark_graphs).to(self.device)
            # features_hidden = torch.ones(self.size_landmark_graphs, self.hidden_dim).to(self.device)
            # features_hidden = self.landmark_gcn_layer_list[i](features_hidden, adj_landmark)

            adj_landmark = torch.nonzero(adj_landmark).transpose(0, 1).to(dtype=torch.long)

            features_hidden = self.landmark_gcn_layer(features_hidden, adj_landmark)

            # Calculate the attention
            # adj = batch_g.adj().to(self.device)
            orig_feat = h
            coar_feat = features_hidden
            orig_feat = self.attention_W(orig_feat)
            coar_feat = self.attention_W(coar_feat)
            cat_dump_coar = coar_feat.repeat(1, len(orig_feat)).reshape(-1, self.hidden_dim)
            cat_dump_orig = orig_feat.repeat(len(coar_feat), 1)
            e = torch.cat((cat_dump_coar, cat_dump_orig), 1)
            # e = self.attention_alpha(e).reshape(-1, self.size_landmark_graphs)
            e = self.attention_alpha(self.LeakyReLU(e)).reshape(-1, self.size_landmark_graphs)
            e = self.softmax(self.sigmoid(e))
            # e = self.softmax(e)

            # # Do not use attention
            # feat = self.assign_W(h)
            # e = self.assign_soft(feat)

            # pool graph
            graph_list = dgl.unbatch(batch_g)
            e_list = torch.split(e, batch_num_nodes.to('cpu').tolist())
            e_landmark_list.append(e_list)

        pool_graph_list = list()
        batch_idx = 0
        batch_sp_dist = list()

        for i in range(len(graph_list)):
            graph = graph_list[i]
            assign_mtx_list = [e_list[i] for e_list in e_landmark_list]
            min_spectral_dist = float("INF")
            min_spectral_pool_adj = None
            min_spectral_assign_mtx = None

            if graph.num_nodes() < self.size_landmark_graphs:
                pool_adj = graph.adj().to_dense().to(self.device)
                pool_adj = pool_adj - torch.diag_embed(torch.diag(pool_adj))
                sp_dist = torch.tensor([0])
                indices = torch.nonzero(pool_adj).transpose(0, 1)
                pool_g = dgl.graph((indices[0], indices[1]), num_nodes=len(pool_adj))
                pool_g.edata['w'] = pool_adj[indices[0], indices[1]]
                pool_g.ndata['feat'] = h[batch_idx:batch_idx + batch_num_nodes[i].item(), :]
            else:
                for j in range(self.num_landmark_graphs):
                    adj = graph.adj().to_dense().to(self.device)
                    assign_mtx = assign_mtx_list[j]
                    pool_adj = (assign_mtx.transpose(0, 1).matmul(adj)).matmul(assign_mtx)
                    # remove self-loop
                    pool_adj = pool_adj - torch.diag_embed(torch.diag(pool_adj))

                    assign_mtx_nor = assign_mtx / torch.sum(assign_mtx, 0)
                    reverse_adj = (assign_mtx_nor.matmul(pool_adj)).matmul(assign_mtx_nor.transpose(0, 1))
                    # reverse_adj = (assign_mtx.matmul(pool_adj)).matmul(assign_mtx.transpose(0, 1))
                    reverse_adj = reverse_adj - torch.diag_embed(torch.diag(reverse_adj))
                    eig_reverse = self.lap_eigen_cal(reverse_adj)
                    eig_adj = self.lap_eigen_cal(adj)
                    sp_dist = self.spectral_distance(eig_adj, eig_reverse)
                    if sp_dist < min_spectral_dist:
                        min_spectral_dist = sp_dist
                        min_spectral_pool_adj = pool_adj
                        min_spectral_assign_mtx = assign_mtx

                pool_adj = min_spectral_pool_adj
                assign_mtx = min_spectral_assign_mtx
                sp_dist = min_spectral_dist
                indices = torch.nonzero(pool_adj).transpose(0, 1)

                pool_g = dgl.graph((indices[0], indices[1]), num_nodes=len(pool_adj))
                pool_g.edata['w'] = pool_adj[indices[0], indices[1]]
                feat_g = h[batch_idx:batch_idx + batch_num_nodes[i].item(), :]

                pool_g.ndata['feat'] = assign_mtx.transpose(0, 1).matmul(feat_g)

            batch_idx = batch_idx + batch_num_nodes[i].item()
            pool_graph_list.append(pool_g)
            if torch.isnan(sp_dist):
                print('ok')
            batch_sp_dist.append(sp_dist)

        pool_batch_g = dgl.batch(pool_graph_list)
        del pool_graph_list

        return pool_batch_g, pool_batch_g.ndata['feat'], batch_sp_dist


class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.device = args.device
        self.SDPool_list = nn.ModuleList()
        self.GCN_list = nn.ModuleList()
        self.num_pool_layer = args.num_pool_layer
        self.num_coarsen_graphs_list = args.num_coarsen_graphs_list.split(',')
        self.size_coarsen_graphs_list = args.size_coarsen_graphs_list.split(',')

        node_encoder = GCN
        # node_encoder = GIN
        for i in range(self.num_pool_layer):
            if i == 0:
                self.GCN_list.append(node_encoder(args.input_dim, args.hidden_dim, args.num_gcn_layers, args.hidden_dim))
            else:
                self.GCN_list.append(node_encoder(args.hidden_dim, args.hidden_dim, args.num_gcn_layers, args.hidden_dim))
            self.SDPool_list.append(SDPool(args.device, int(self.size_coarsen_graphs_list[i]), int(self.num_coarsen_graphs_list[i]), args.hidden_dim, args.num_landmark_gcn_layers))


        self.classify = nn.Linear(args.hidden_dim, args.out_dim)

    def forward(self, batch_graph):
        h = batch_graph.ndata['feat'].float()
        adj_g = batch_graph.adj().to(self.device)
        g = batch_graph
        sp_dist_list = list()
        for i in range(self.num_pool_layer):
            adj_g = adj_g.coalesce().indices()
            # adj_g = torch.nonzero(adj_g).transpose(0, 1).to(dtype=torch.long)
            h = self.GCN_list[i](h, adj_g)
            g, h, batch_sp_dist = self.SDPool_list[i](h, g)
            sp_dist_list.append(sum(batch_sp_dist)/len(batch_sp_dist))
            adj_g = g.adj().to(self.device)
        h = dgl.readout_nodes(g, 'feat')
        h = self.classify(h)
        if torch.isnan(sum(sp_dist_list)):
            print("OK")
        return h, sum(sp_dist_list)/len(sp_dist_list)


def accuracy(prediction, labels):
    _, indices = torch.max(prediction, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)
