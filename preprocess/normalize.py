"""
Normalize traffic data using Z-score normalization
Compute statistics on training set and apply to all splits
"""

import numpy as np
import argparse
import os
import pickle


def z_score_normalize(data, mean=None, std=None):
    """
    Z-score normalization: (x - mean) / std
    
    Args:
        data: Input data
        mean: Mean for normalization (computed from data if None)
        std: Std for normalization (computed from data if None)
    
    Returns:
        Normalized data, mean, std
    """
    if mean is None:
        mean = np.mean(data, axis=0)
    if std is None:
        std = np.std(data, axis=0)
    
    # Avoid division by zero
    std = np.where(std == 0, 1, std)
    
    normalized_data = (data - mean) / std
    
    return normalized_data, mean, std


def min_max_normalize(data, min_val=None, max_val=None):
    """
    Min-max normalization: (x - min) / (max - min)
    
    Args:
        data: Input data
        min_val: Minimum value (computed from data if None)
        max_val: Maximum value (computed from data if None)
    
    Returns:
        Normalized data, min_val, max_val
    """
    if min_val is None:
        min_val = np.min(data, axis=0)
    if max_val is None:
        max_val = np.max(data, axis=0)
    
    # Avoid division by zero
    range_val = max_val - min_val
    range_val = np.where(range_val == 0, 1, range_val)
    
    normalized_data = (data - min_val) / range_val
    
    return normalized_data, min_val, max_val


def save_scaler(scaler_dict, save_path):
    """Save normalization parameters"""
    with open(save_path, 'wb') as f:
        pickle.dump(scaler_dict, f)
    print(f"Scaler saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Normalize traffic data')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['pems04', 'pems07', 'pems08', 'nyctaxi'],
                       help='Dataset name')
    parser.add_argument('--data_dir', type=str,
                       help='Directory containing train/val/test splits')
    parser.add_argument('--method', type=str, default='zscore',
                       choices=['zscore', 'minmax'],
                       help='Normalization method')
    parser.add_argument('--output_dir', type=str,
                       help='Directory to save normalized data')
    
    args = parser.parse_args()
    
    # Set default paths
    if args.data_dir is None:
        args.data_dir = f'data/{args.dataset.upper()}'
    if args.output_dir is None:
        args.output_dir = args.data_dir
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load training data to compute statistics
    print(f"Loading training data from {args.data_dir}")
    train_data = np.load(os.path.join(args.data_dir, 'train.npz'))['data']
    
    print(f"Train data shape: {train_data.shape}")
    print(f"Using {args.method} normalization")
    
    # Compute normalization parameters from training data
    if args.method == 'zscore':
        _, mean, std = z_score_normalize(train_data)
        scaler_dict = {'method': 'zscore', 'mean': mean, 'std': std}
        print(f"Mean shape: {mean.shape}, Std shape: {std.shape}")
    elif args.method == 'minmax':
        _, min_val, max_val = min_max_normalize(train_data)
        scaler_dict = {'method': 'minmax', 'min': min_val, 'max': max_val}
        print(f"Min shape: {min_val.shape}, Max shape: {max_val.shape}")
    
    # Apply normalization to all splits
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(args.data_dir, f'{split}.npz')
        if not os.path.exists(split_path):
            print(f"Warning: {split_path} not found, skipping...")
            continue
        
        data = np.load(split_path)['data']
        
        if args.method == 'zscore':
            normalized_data, _, _ = z_score_normalize(data, mean=mean, std=std)
        elif args.method == 'minmax':
            normalized_data, _, _ = min_max_normalize(data, min_val=min_val, max_val=max_val)
        
        # Save normalized data
        output_path = os.path.join(args.output_dir, f'{split}_normalized.npz')
        np.savez_compressed(output_path, data=normalized_data)
        print(f"Saved normalized {split} data to {output_path}: {normalized_data.shape}")
    
    # Save scaler parameters
    scaler_path = os.path.join(args.output_dir, 'scaler.pkl')
    save_scaler(scaler_dict, scaler_path)
    
    print("Normalization completed!")


if __name__ == '__main__':
    main()
