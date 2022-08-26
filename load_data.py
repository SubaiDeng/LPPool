import pickle
import numpy as np
import random
import torch
import dgl
import networkx as nx
import torch.nn.functional as F
from dgl.data import DGLDataset
from dgl.data import LegacyTUDataset
from dgl.dataloading import GraphDataLoader

class MyDataset(DGLDataset):
    def __init__(self,
                 args,
                 dataset,
                 url=None,
                 raw_dir=None,
                 save_dir=None,
                 force_reload=False,
                 verbose=False
                 ):
        super(MyDataset, self).__init__(name=args.bmname,
                                           url=url,
                                           raw_dir=raw_dir,
                                           save_dir=save_dir,
                                           force_reload=force_reload,
                                           verbose=verbose
                                           )
        self.device = args.device
        self.graph_labels = dataset.graph_labels.long()
        self.graph_lists = dataset.graph_lists
        self.num_labels = dataset.num_labels
        self.num_feat_type = args.node_feat_type

        args.out_dim = self.num_labels
        self.remove_isolated()
        # self.add_self_loop()
        self.num_graph = len(self.graph_lists)
        self.node_num_mean = 0
        self.node_num_max = 0
        self.node_num_min = 0
        self.graph_statistic()
        args.input_dim = self.node_feat_selection()
        # self.eigen_cal()
        print('Dataset Finish Loading.')

        # self.num_graph = len(self.graph_list)
        # self.num_attr = len(self.graph_list[0].ndata['attr'][0])
        # self.eigvalue_list = self.eigen_cal(self.graph_list)
        # args.out_dim = len(set(label_list))

    def __getitem__(self, idx):
        return self.graph_lists[idx].to(self.device), self.graph_labels[idx].to(self.device)

    def __len__(self):
        return self.num_graph

    def download(self):
        # download raw data to local disk
        pass

    def process(self):
        # process raw data to graphs, labels, splitting masks
        pass

    def save(self):
        # save processed data to directory `self.save_path`
        pass

    def load(self):
        # load processed data from directory `self.save_path`
        pass

    def has_cache(self):
        # check whether there are processed data in `self.save_path`
        pass

    def add_self_loop(self):
        graph_lists = self.graph_lists
        for i, g in enumerate(graph_lists):
            graph_lists[i] = g.add_self_loop()

    def remove_isolated(self):
        graph_lists = self.graph_lists
        for i, g in enumerate(graph_lists):
            deg = g.in_degrees()
            zero_index = torch.where(deg==0)[0]
            g.remove_nodes(zero_index)

    def graph_statistic(self):
        node_num_sum = 0
        node_num_min = float('inf')
        node_num_max = -1
        for g in self.graph_lists:
            g_len = g.num_nodes()
            node_num_sum += g_len
            if g_len > node_num_max:
                node_num_max = g_len
            if g_len < node_num_min:
                node_num_min = g_len
            self.node_num_min = node_num_min
            self.node_num_max = node_num_max
        self.node_num_mean = node_num_sum/len(self.graph_lists)
        print('ok')

    def node_feat_selection_feat(self, g):
        feat = g.ndata['feat'].float()
        return feat, feat.size()[1]

    def node_feat_selection_deg(self, g):
        deg = g.in_degrees().reshape(-1, 1)
        feat = deg.float()
        return feat, feat.size()[1]

    def node_feat_selection_one(self, g):
        feat = torch.ones_like(g.ndata['feat']).float()
        return feat, feat.size()[1]




    def node_feat_selection(self):
        if self.num_feat_type == 'feat':
            obtain_node_feat = self.node_feat_selection_feat
        elif self.num_feat_type == 'deg':
            obtain_node_feat = self.node_feat_selection_deg
        elif self.num_feat_type == 'one':
            obtain_node_feat = self.node_feat_selection_one

            # for i, g in enumerate(self.graph_lists):
            #     feat = g.ndata['feat'].long()
            #     t = torch.max(feat)
            #     print(t,end=' ')
            #     if t > max_onehot:
            #         max_onehot = t
            #     print(max_onehot)
            #     print(torch.unique(feat))
        if self.num_feat_type == 'onehot':
            graph_len_list = list()
            feat_list = None
            for i, g in enumerate(self.graph_lists):
                feat = g.ndata['feat'].long()
                if feat_list is None:
                    feat_list = feat
                else:
                    feat_list = torch.vstack((feat_list, feat))
                graph_len_list.append(len(feat))
            feat_list = feat_list.reshape(-1)
            min = torch.min(feat_list)
            feat_list = feat_list - min
            feat_onehot_list = F.one_hot(feat_list)
            idxs = torch.sum(feat_onehot_list, 0)
            feat_onehot_list = feat_onehot_list[:, idxs > 0]
            input_dim = feat_onehot_list.size()[1]
            slice_id = 0
            for i, g in enumerate(self.graph_lists):
                self.graph_lists[i].ndata['feat'] = feat_onehot_list[slice_id:slice_id+graph_len_list[i], :]
        else:
            input_dim = 0
            for i, g in enumerate(self.graph_lists):
                feat, input_dim = obtain_node_feat(g)
                self.graph_lists[i].ndata['feat'] = feat

        return input_dim

    def eigen_cal(self):
        # eig_value_list = list()
        for i, g in enumerate(self.graph_lists):
            # build laplacian
            adj = g.adj().to_dense()
            deg_ng_half = torch.pow(g.in_degrees(), -1/2)
            laplacian_g = torch.eye(len(deg_ng_half)) - torch.diag(deg_ng_half).matmul(adj).matmul(torch.diag(deg_ng_half))
            u, _ = torch.eig(laplacian_g)
            u = u[:, 0].reshape(-1)  # remove the imaginary part
            u, _ = torch.sort(u, descending=True)
            g.ndata['eig'] = u
            # eig_value_list.append(u)
            print('Compute the eig of graph {}'.format(i))
        print('FINISH EIGEN CALCULATION.')


def one_hot_embedding(node_label_list, node_label_dict):
    node_feature_list = []
    len_attribute = len(node_label_dict)
    for label in node_label_list:
        node_attribute = np.zeros(len_attribute)
        node_attribute[node_label_dict[label]] = 1
        # node_feature_list = np.append(node_feature_list, node_attribute.reshape(1,-1), axis=0)
        node_feature_list.append(node_attribute)
    node_feature_list = torch.tensor(node_feature_list, dtype=torch.float)
    return node_feature_list


def build_graph(graph, args):
    # edges_list = torch.tensor(graph['edge'], dtype=torch.long)
    edges_list = graph['edge'].copy()
    edges_weight = graph['edge_weight'].copy()

    nx_g = nx.Graph()
    nx_g.add_weighted_edges_from([(edges_list[i][0],
                                   edges_list[i][1],
                                   edges_weight[i]) for i in range(len(edges_list))])

    dgl_g = dgl.from_networkx(nx_g)

    dgl_g.ndata['attr'] = graph['node_attribute']
    dgl_g = dgl_g.to(args.device)
    g_labels = graph['graph_label']

    # forward_edges_list = graph['edge'].copy()
    # # add backward edges
    # for i, edge in enumerate(forward_edges_list):
    #     if edge[1] != edge[0]:  # remove self loop
    #         edges_list.append([edge[1], edge[0]])
    #         edges_weight.append(edges_weight[i])
    #
    # edges_list = torch.tensor(edges_list, dtype=torch.long, device=args.device)
    # edges_weight = torch.tensor(edges_weight, dtype=torch.float, device=args.device)
    # graph_label = torch.tensor(graph['graph_label'], dtype=torch.long, device=args.device)
    # node_feature_list = graph['node_attribute'].to(args.device)
    #
    # g = Data(x=node_feature_list, edge_index=edges_list.t().contiguous(), edge_attr=edges_weight, device=args.device)
    # g.graph_label = graph_label
    return dgl_g, g_labels


def load_graph(args):
    dataset = LegacyTUDataset(name=args.bmname)
    dataset = MyDataset(args, dataset)
    return dataset


def split_dataset(args, train_idx, val_idx, test_idx, dataset):

    train_set = torch.utils.data.dataset.Subset(dataset, train_idx)
    val_set = torch.utils.data.dataset.Subset(dataset, val_idx)
    test_set = torch.utils.data.dataset.Subset(dataset, test_idx)
    # train_val_set = [dataset[i] for i in train_val_idx]
    # test_set = [dataset[i] for i in test_idx]

    # test_loader = DataLoader(test_set,
    #                          batch_size=args.batch_size,
    #                          shuffle=False)
    # train_val_loader = DataLoader(train_val_set,
    #                               batch_size=args.batch_size,
    #                               shuffle=True)

    # val_batch_size = args.batch_size
    val_batch_size = 1

    train_loader = GraphDataLoader(train_set, batch_size=args.batch_size, drop_last=False, shuffle=True)
    val_loader = GraphDataLoader(val_set, batch_size=val_batch_size, drop_last=False, shuffle=True)
    test_loader = GraphDataLoader(test_set, batch_size=val_batch_size, drop_last=False, shuffle=True)

    return train_loader, val_loader, test_loader
