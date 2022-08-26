import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import dgl
import torch
from dgl.data import DGLDataset
from dgl.data import LegacyTUDataset
from dgl.dataloading import GraphDataLoader


class SynthesisDataset:
    def __init__(self, graph_lists, graph_label_list, num_labels):
        super(SynthesisDataset, self).__init__()
        self.graph_labels = graph_label_list
        self.graph_lists = graph_lists
        self.num_labels = num_labels


def adj_concat(a, b):
    len_a = len(a)
    len_b = len(b)
    left = np.row_stack((a, np.zeros((len_b, len_a))))
    right = np.row_stack((np.zeros((len_a,len_b)), b))
    result = np.hstack((left, right))
    return result

def generate_er_graph(n, p):
    adj = np.random.rand(n, n)
    adj = np.where(adj < p, 1, 0)
    adj = np.triu(adj, k=1)
    adj = adj+np.transpose(adj)
    return adj


def generate_er_graph_fix_edges(n, e):
    # e = int(e/2)
    adj = np.zeros((n, n))
    l = int(n*(n-1)/2)
    element_list = np.zeros(l)
    l = list(range(l))
    np.random.shuffle(l)
    l = l[0:e]
    element_list[l] = 1
    idx = np.triu_indices(n, 1)
    adj[idx] = element_list
    adj = adj+np.transpose(adj)
    return adj


def generate_pos_sample(lp_graph_adj, low_rand, up_rand, p):
    lp_graph_adj = np.triu(lp_graph_adj, k=1)
    n = len(lp_graph_adj)
    num_sub_node_list = list()
    num_sub_node_sum_list = list()
    num_sub_node_sum_list.append(0)
    adj = np.zeros((0, 0))
    for i in range(n):
        num_sub_node = np.random.randint(low_rand, up_rand)
        num_sub_node_list.append(num_sub_node)
        num_sub_node_sum_list.append(num_sub_node+num_sub_node_sum_list[i])
        sub_adj = generate_er_graph(num_sub_node, p)
        adj = adj_concat(adj, sub_adj)

    idx = np.nonzero(lp_graph_adj)

    sub_node_list = list()
    for i in range(n):
        sub_node_idx = np.random.randint(num_sub_node_list[i])
        sub_node_base = num_sub_node_sum_list[i]
        sub_node = sub_node_idx + sub_node_base
        sub_node_list.append(sub_node)

    for i in range(len(idx[0])):
        x = idx[0][i]
        y = idx[1][i]

        # x_idx_base = num_sub_node_sum_list[x]
        # y_idx_base = num_sub_node_sum_list[y]
        # x_idx = x_idx_base + np.random.randint(num_sub_node_list[x])
        # y_idx = y_idx_base + np.random.randint(num_sub_node_list[y])
        x_idx = sub_node_list[x]
        y_idx = sub_node_list[y]
        adj[x_idx, y_idx] = 1
        adj[y_idx, x_idx] = 1

    return adj


def generate_graph_sample(lp_graph_adj, low_rand, up_rand, p):
    pos_adj = generate_pos_sample(lp_graph_adj, low_rand, up_rand, p)
    num_edges = int(len(np.nonzero(pos_adj)[0])/2)
    num_nodes = len(pos_adj)
    neg_adj = generate_er_graph_fix_edges(num_nodes, num_edges)
    return pos_adj, neg_adj

def synthesis_generate(lp_graph_adj, num_graph, low_rand, up_rand, p):

    graph_list = list()
    label_list = list()
    for i in range(num_graph):
        pos_adj, neg_adj = generate_graph_sample(lp_graph_adj, low_rand, up_rand, p)
        g_pos_nx = nx.from_numpy_matrix(pos_adj)
        g_pos = dgl.from_networkx(g_pos_nx)
        g_neg_nx = nx.from_numpy_matrix(neg_adj)
        g_neg = dgl.from_networkx(g_neg_nx)
        graph_list.append(g_pos)
        label_list.append(1)
        graph_list.append(g_neg)
        label_list.append(0)
    label_list = torch.tensor(label_list)
    dataset = SynthesisDataset(graph_list, label_list, 2)
    print('Synthesis dataset finish building.')
    return dataset





if __name__ == '__main__':
    # a = [[0, 1, 1, 0],
    #      [1, 0, 0, 1],
    #      [1, 0, 0, 1],
    #      [0, 1, 1, 0]]

    # a = [[0, 1, 0, 0, 0, 1],
    #      [1, 0, 1, 0, 0, 0],
    #      [0, 1, 0, 1, 0, 0],
    #      [0, 0, 1, 0, 1, 0],
    #      [0, 0, 0, 1, 0, 1],
    #      [1, 0, 0, 0, 1, 0],]
    a = [[0, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],]
    pos_adj, neg_adj = generate_graph_sample(a, 6, 10, 0.7)
    g_pos = nx.from_numpy_matrix(pos_adj)
    g_neg = nx.from_numpy_matrix(neg_adj)

    print(pos_adj)
    nx.draw(g_pos)
    plt.show()
    print(neg_adj)
    nx.draw(g_neg)
    plt.show()

    # a = [[0, 1, 0, 0, 0, 1],
    #      [1, 0, 1, 0, 0, 0],
    #      [0, 1, 0, 1, 0, 0],
    #      [0, 0, 1, 0, 1, 0],
    #      [0, 0, 0, 1, 0, 1],
    #      [1, 0, 0, 0, 1, 0],]
    # dataset = synthesis_generate(a, 500, 6, 10, 0.7)
