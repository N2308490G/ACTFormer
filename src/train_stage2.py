"""
Stage 2 Training: Fine-tuning phase
"""

import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os


def train_stage2(config, pretrained_path):
    """
    Stage 2 training logic
    """
    print("Starting Stage 2 training...")
    print(f"Loading pretrained model from: {pretrained_path}")
    # TODO: Implement training logic
    pass


def main():
    parser = argparse.ArgumentParser(description='ACTFormer Stage 2 Training')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--pretrained', type=str, required=True,
                       help='Path to pretrained model from Stage 1')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Train
    train_stage2(config, args.pretrained)


if __name__ == '__main__':
    main()
