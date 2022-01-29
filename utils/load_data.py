import dgl
import networkx as nx
import numpy as np
import torch as th
np.random.seed(0)
th.manual_seed(0)
th.backends.cudnn.deterministic = True
th.backends.cudnn.benchmark = False
from graphgallery.datasets import NPZDataset

def load_graphgallery_data(dataset):
    # set `verbose=False` to avoid additional outputs
    data = NPZDataset(dataset, verbose=False)
    graph = data.graph
    nx_g = nx.from_scipy_sparse_matrix(graph.adj_matrix)

    for node_id, node_data in nx_g.nodes(data=True):
        node_data["features"] = graph.feat[node_id].astype(np.float32)
        if dataset in ['blogcatalog', 'flickr']:
            node_data["labels"] = graph.y[node_id].astype(np.long) - 1
        else:
            node_data["labels"] = graph.y[node_id].astype(np.long)


    dgl_graph = dgl.from_networkx(nx_g, node_attrs=['features', 'labels'])
    dgl_graph = dgl.add_self_loop(dgl_graph)
    dgl_graph = dgl.to_simple(dgl_graph, copy_ndata=True)
    dgl_graph = dgl.to_bidirected(dgl_graph, copy_ndata=True)
    
    print(f"Graph has {dgl_graph.number_of_nodes()} nodes, {dgl_graph.number_of_edges()} edges.")
    return dgl_graph, graph.num_classes

def node_sample(g, prop=0.5):
    '''
    sample target/shadow graph (1:1) 
    '''
    node_number = len(g.nodes())
    node_index_list = np.arange(node_number) 
    np.random.seed(0)
    np.random.shuffle(node_index_list)
    split_length = int(node_number * prop)

    train_index = np.sort(node_index_list[:split_length])
    test_index = np.sort(node_index_list[split_length: ])

    return train_index, test_index

def split_target_shadow(g):
    '''
    generate input data for target gnn model
    '''

    target_index, shadow_index = node_sample(g, 0.5)

    target_g = g.subgraph(target_index)
    shadow_g = g.subgraph(shadow_index)

    return target_g, shadow_g

def split_train_test(g):
    '''
    generate input data for target gnn model
    '''

    train_index, test_index = node_sample(g, 0.5)

    train_g = g.subgraph(train_index)
    test_g = g.subgraph(test_index)

    return train_g, test_g