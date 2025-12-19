"""
Build adjacency matrix for traffic networks
Supports multiple methods: distance-based, connectivity-based, etc.
"""

import numpy as np
import pandas as pd
import pickle
import argparse
from scipy.spatial.distance import cdist


def get_adjacency_matrix(distance_df, num_nodes, method='gaussian', sigma=0.1, epsilon=0.5):
    """
    Generate adjacency matrix from distance matrix
    
    Args:
        distance_df: DataFrame with distance information
        num_nodes: Number of nodes in the graph
        method: Method to compute adjacency ('gaussian', 'threshold', 'knn')
        sigma: Standard deviation for Gaussian kernel
        epsilon: Threshold for connectivity
    
    Returns:
        Adjacency matrix as numpy array
    """
    if method == 'gaussian':
        distances = distance_df.values.astype(np.float32)
        distances[distances == 0] = np.inf
        adj = np.exp(-np.square(distances / sigma))
        adj[adj < epsilon] = 0
        
    elif method == 'threshold':
        distances = distance_df.values.astype(np.float32)
        adj = np.zeros_like(distances)
        adj[distances < epsilon] = 1
        
    elif method == 'knn':
        k = int(epsilon)
        distances = distance_df.values.astype(np.float32)
        adj = np.zeros_like(distances)
        for i in range(num_nodes):
            idx = np.argsort(distances[i, :])[1:k+1]  # Exclude self
            adj[i, idx] = 1
            
    return adj


def build_adjacency_from_coordinates(coords, num_nodes, method='gaussian', **kwargs):
    """
    Build adjacency matrix from node coordinates
    
    Args:
        coords: Array of shape (num_nodes, 2) with lat/lon coordinates
        num_nodes: Number of nodes
        method: Method to compute adjacency
    
    Returns:
        Adjacency matrix
    """
    distances = cdist(coords, coords, metric='euclidean')
    distance_df = pd.DataFrame(distances)
    
    return get_adjacency_matrix(distance_df, num_nodes, method=method, **kwargs)


def save_adjacency_matrix(adj_matrix, save_path):
    """Save adjacency matrix as pickle file"""
    with open(save_path, 'wb') as f:
        pickle.dump(adj_matrix, f)
    print(f"Adjacency matrix saved to {save_path}")
    print(f"Shape: {adj_matrix.shape}, Density: {np.sum(adj_matrix > 0) / adj_matrix.size:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Build adjacency matrix')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['pems04', 'pems07', 'pems08', 'nyctaxi'],
                       help='Dataset name')
    parser.add_argument('--method', type=str, default='gaussian',
                       choices=['gaussian', 'threshold', 'knn'],
                       help='Method to build adjacency matrix')
    parser.add_argument('--sigma', type=float, default=0.1,
                       help='Sigma for Gaussian kernel')
    parser.add_argument('--epsilon', type=float, default=0.5,
                       help='Threshold or k for adjacency')
    parser.add_argument('--input_file', type=str, 
                       help='Path to input distance/coordinate file')
    parser.add_argument('--output_file', type=str,
                       help='Path to save adjacency matrix')
    
    args = parser.parse_args()
    
    # Set default paths if not provided
    if args.input_file is None:
        args.input_file = f'data/{args.dataset.upper()}/distances.csv'
    if args.output_file is None:
        args.output_file = f'data/{args.dataset.upper()}/adj_mx.pkl'
    
    # Load distance data
    print(f"Loading distance data from {args.input_file}")
    distance_df = pd.read_csv(args.input_file, header=None)
    num_nodes = distance_df.shape[0]
    
    print(f"Building adjacency matrix for {num_nodes} nodes using {args.method} method")
    adj_matrix = get_adjacency_matrix(
        distance_df, 
        num_nodes, 
        method=args.method,
        sigma=args.sigma,
        epsilon=args.epsilon
    )
    
    # Save adjacency matrix
    save_adjacency_matrix(adj_matrix, args.output_file)


if __name__ == '__main__':
    main()
