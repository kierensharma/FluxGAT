import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from collections import defaultdict
from torch_geometric.utils import dense_to_sparse

def create_k_fold_splits(labels, k=5):
    """
    Generates k-fold splits for training and testing, ignoring NaN labels.

    Parameters:
    labels (numpy.ndarray): Array of labels with possible NaNs.
    k (int): Number of folds for KFold splitting.

    Returns:
    list of tuples: Each tuple contains two arrays, the first for training indices and the second for test indices.
    """
    non_nan_indices = np.where(~np.isnan(labels))[0]
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    splits = [(non_nan_indices[train_idx], non_nan_indices[test_idx]) for train_idx, test_idx in kf.split(non_nan_indices)]
    return splits

def masking(M, features, labels):
    """
    Applies a mask to the nodes, edges, features, and labels based on the MFG adjacency matrix.

    Parameters:
    M (numpy.ndarray): MFG adjacency matrix.
    features (pandas.DataFrame): Node features for all reactions.
    labels (pandas.DataFrame): Node labels for all reactions.

    Returns:
    Tuple: Contains connected nodes, masked edge index, edge weights, masked node features, and masked node labels.
    """
    edge_index, edge_weights = dense_to_sparse(torch.tensor(M))

    genes_one_hot = np.array(features['reactants_one_hot'])
    node_features = np.concatenate((genes_one_hot, genes_one_hot), axis=0)

    reaction_essentiality = np.array(labels['essentiality'])
    node_labels = np.concatenate((reaction_essentiality, reaction_essentiality), axis=0)

    connected_nodes = torch.unique(edge_index)
    node_mapping = {node.item(): i for i, node in enumerate(connected_nodes)}

    masked_edge_index = torch.tensor([[node_mapping[src.item()], node_mapping[dst.item()]] for src, dst in edge_index.t()]).t()
    masked_node_features = np.array([node_features[i] for i in node_mapping])
    masked_node_labels = np.array([node_labels[i] for i in node_mapping])

    return connected_nodes, masked_edge_index, edge_weights, masked_node_features, masked_node_labels

def main():
    try:
        M = pd.read_pickle('../data/MFG.pkl')
        features_df = pd.read_pickle('../data/node_features.pkl')
        labels_df = pd.read_pickle('../data/node_labels.pkl')

        connected_nodes, masked_edge_index, edge_weights, masked_node_features, masked_node_labels = masking(M, features_df, labels_df)
        splits = create_k_fold_splits(masked_node_labels, k=5)

        dict_df = defaultdict(list)
        for fold, (train_indices, test_indices) in enumerate(splits):
            train_mask = torch.zeros(len(connected_nodes), dtype=torch.bool)
            test_mask = torch.zeros(len(connected_nodes), dtype=torch.bool)
            train_mask[train_indices] = True
            test_mask[test_indices] = True

            dict_df['fold'].append(fold + 1)
            dict_df['node_features'].append(masked_node_features)
            dict_df['node_labels'].append(masked_node_labels)
            dict_df['edge_weights'].append(edge_weights)
            dict_df['edge_index'].append(masked_edge_index)
            dict_df['train_mask'].append(train_mask)
            dict_df['test_mask'].append(test_mask)

        df = pd.DataFrame(dict_df)
        df.to_pickle('../data/k_fold_masking.pkl')
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()