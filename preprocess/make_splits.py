"""
Split dataset into train/validation/test sets
Standard splits: 60% train, 20% validation, 20% test
"""

import numpy as np
import pandas as pd
import argparse
import os
import pickle


def generate_splits(data, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    """
    Split data into train/val/test sets
    
    Args:
        data: Input data array of shape (num_samples, num_nodes, num_features)
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
    
    Returns:
        Dictionary with train/val/test indices
    """
    assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1"
    
    num_samples = data.shape[0]
    num_train = int(num_samples * train_ratio)
    num_val = int(num_samples * val_ratio)
    num_test = num_samples - num_train - num_val
    
    # Sequential split (for time series data)
    train_data = data[:num_train]
    val_data = data[num_train:num_train + num_val]
    test_data = data[num_train + num_val:]
    
    print(f"Train samples: {len(train_data)}")
    print(f"Val samples: {len(val_data)}")
    print(f"Test samples: {len(test_data)}")
    
    return {
        'train': train_data,
        'val': val_data,
        'test': test_data
    }


def save_splits(splits, save_dir):
    """Save train/val/test splits"""
    os.makedirs(save_dir, exist_ok=True)
    
    for split_name, split_data in splits.items():
        save_path = os.path.join(save_dir, f'{split_name}.npz')
        np.savez_compressed(save_path, data=split_data)
        print(f"Saved {split_name} split to {save_path}: {split_data.shape}")


def main():
    parser = argparse.ArgumentParser(description='Generate train/val/test splits')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['pems04', 'pems07', 'pems08', 'nyctaxi'],
                       help='Dataset name')
    parser.add_argument('--input_file', type=str,
                       help='Path to input data file (.npz or .h5)')
    parser.add_argument('--output_dir', type=str,
                       help='Directory to save splits')
    parser.add_argument('--train_ratio', type=float, default=0.6,
                       help='Training set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                       help='Validation set ratio')
    parser.add_argument('--test_ratio', type=float, default=0.2,
                       help='Test set ratio')
    
    args = parser.parse_args()
    
    # Set default paths
    if args.input_file is None:
        args.input_file = f'data/{args.dataset.upper()}/data.npz'
    if args.output_dir is None:
        args.output_dir = f'data/{args.dataset.upper()}'
    
    # Load data
    print(f"Loading data from {args.input_file}")
    if args.input_file.endswith('.npz'):
        data_dict = np.load(args.input_file)
        data = data_dict['data']
    elif args.input_file.endswith('.h5'):
        import h5py
        with h5py.File(args.input_file, 'r') as f:
            data = f['data'][:]
    else:
        raise ValueError("Unsupported file format. Use .npz or .h5")
    
    print(f"Data shape: {data.shape}")
    
    # Generate splits
    splits = generate_splits(
        data,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio
    )
    
    # Save splits
    save_splits(splits, args.output_dir)
    print("Split generation completed!")


if __name__ == '__main__':
    main()
