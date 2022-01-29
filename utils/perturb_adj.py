import torch
import numpy as np
import scipy.sparse as sp
import time
import dgl
from tqdm import tqdm
import networkx as nx
# perturb adjacency matrix
# get adjacecy matrix from dgl graph
# 



def get_adj(g):
    return g.adj(scipy_fmt='csr')


def get_noise(noise_type, size, seed, eps=10, delta=1e-5, sensitivity=2):
    np.random.seed(seed)

    if noise_type == 'laplace':
        noise = np.random.laplace(0, sensitivity/eps, size)
    elif noise_type == 'gaussian':
        c = np.sqrt(2*np.log(1.25/delta))
        stddev = c * sensitivity / eps
        noise = np.random.normal(0, stddev, size)
    else:
        raise NotImplementedError('noise {} not implemented!'.format(noise_type))
    if size == 1:
        print(noise)
    return noise

def construct_sparse_mat(indice, N):
        cur_row = -1
        new_indices = []
        new_indptr = []

        for i, j in tqdm(indice):
            if i >= j:
                continue

            while i > cur_row:
                new_indptr.append(len(new_indices))
                cur_row += 1

            new_indices.append(j)

        while N > cur_row:
            new_indptr.append(len(new_indices))
            cur_row += 1

        data = np.ones(len(new_indices), dtype=np.int64)
        indices = np.asarray(new_indices, dtype=np.int64)
        indptr = np.asarray(new_indptr, dtype=np.int64)

        mat = sp.csr_matrix((data, indices, indptr), (N, N))

        return mat + mat.T

def perturb_adj_discrete(args, adj):
        s = 2 / (np.exp(args.epsilon) + 1)
        print(f's = {s:.4f}')
        N = adj.shape[0]

        t = time.time()
        # bernoulli = np.random.binomial(1, s, N * (N-1) // 2)
        # entry = np.where(bernoulli)[0]

        np.random.seed(args.noise_seed)
        bernoulli = np.random.binomial(1, s, (N, N))
        print(f'generating perturbing vector done using {time.time() - t} secs!')
        entry = np.asarray(list(zip(*np.where(bernoulli))))

        dig_1 = np.random.binomial(1, 1/2, len(entry))
        indice_1 = entry[np.where(dig_1 == 1)[0]]
        indice_0 = entry[np.where(dig_1 == 0)[0]]

        add_mat = construct_sparse_mat(indice_1, N)
        minus_mat = construct_sparse_mat(indice_0, N)

        adj_noisy = adj + add_mat - minus_mat

        adj_noisy.data[np.where(adj_noisy.data == -1)[0]] = 0
        adj_noisy.data[np.where(adj_noisy.data == 2)[0]] = 1

        return adj_noisy

def perturb_adj_continuous(args, adj):
        n_nodes = adj.shape[0]
        n_edges = len(adj.data) // 2
        print(f"#Node: {n_nodes}, #Edge: {n_edges}")

        N = n_nodes
        t = time.time()

        A = sp.tril(adj, k=-1)
        print('getting the lower triangle of adj matrix done!')

        eps_1 = args.epsilon * 0.01
        eps_2 = args.epsilon - eps_1
        
        noise = get_noise(noise_type=args.noise_type, size=(N, N), seed=args.noise_seed, eps=eps_2, delta=args.delta, sensitivity=1)
        noise *= np.tri(*noise.shape, k=-1, dtype=np.bool)
        print(f'generating noise done using {time.time() - t} secs!')

        A += noise
        print(f'adding noise to the adj matrix done!')

        t = time.time()
        print(f'eps_2: {eps_2}')
        print(f'eps_1: {eps_1}')
        n_edges_keep = n_edges + int(
            get_noise(noise_type=args.noise_type, size=1, seed=args.noise_seed, 
                    eps=eps_1, delta=args.delta, sensitivity=1)[0])
        print(f'edge number from {n_edges} to {n_edges_keep}')

        t = time.time()
        a_r = A.A.ravel()

        n_splits = 50
        len_h = len(a_r) // n_splits
        ind_list = []
        for i in tqdm(range(n_splits - 1)):
            ind = np.argpartition(a_r[len_h*i:len_h*(i+1)], -n_edges_keep)[-n_edges_keep:]
            ind_list.append(ind + len_h * i)

        ind = np.argpartition(a_r[len_h*(n_splits-1):], -n_edges_keep)[-n_edges_keep:]
        ind_list.append(ind + len_h * (n_splits - 1))

        ind_subset = np.hstack(ind_list)
        a_subset = a_r[ind_subset]
        ind = np.argpartition(a_subset, -n_edges_keep)[-n_edges_keep:]

        row_idx = []
        col_idx = []
        for idx in ind:
            idx = ind_subset[idx]
            row_idx.append(idx // N)
            col_idx.append(idx % N)
            assert(col_idx < row_idx)
        data_idx = np.ones(n_edges_keep, dtype=np.int32)
        print(f'data preparation done using {time.time() - t} secs!')

        mat = sp.csr_matrix((data_idx, (row_idx, col_idx)), shape=(N, N))
        return mat + mat.T

def perturb_adj(args, adj):
    if args.perturb_type == 'discrete':
        return perturb_adj_discrete(args, adj)
    else:
        return perturb_adj_continuous(args, adj)

def get_dp_graph(args, g):
    adj_csr_mx = get_adj(g) 
    features =  {'features': g.ndata['features'], 'labels': g.ndata['labels']}
    adj_perturbed  = perturb_adj(args, adj_csr_mx)
    nx_g = nx.from_scipy_sparse_matrix(adj_perturbed)
    for n in nx_g.nodes():
        for k,v in features.items():
            nx_g.nodes[n][k] = v[n]
    return dgl.from_networkx(nx_g, node_attrs=list(features.keys()))


