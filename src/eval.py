"""
Model Evaluation Script
"""

import argparse
import yaml
import torch
import numpy as np
import json
import os


def calculate_metrics(pred, target):
    """
    Calculate MAE, RMSE, MAPE
    """
    mae = np.mean(np.abs(pred - target))
    rmse = np.sqrt(np.mean((pred - target) ** 2))
    mape = np.mean(np.abs((pred - target) / (target + 1e-5))) * 100
    
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}


def evaluate(config, checkpoint_path, save_results=None):
    """
    Evaluation logic
    """
    print(f"Loading model from: {checkpoint_path}")
    # TODO: Implement evaluation logic
    
    # Dummy results for demonstration
    results = {
        'horizon_3': {'MAE': 0.0, 'RMSE': 0.0, 'MAPE': 0.0},
        'horizon_6': {'MAE': 0.0, 'RMSE': 0.0, 'MAPE': 0.0},
        'horizon_12': {'MAE': 0.0, 'RMSE': 0.0, 'MAPE': 0.0}
    }
    
    print("\nEvaluation Results:")
    for horizon, metrics in results.items():
        print(f"{horizon}: MAE={metrics['MAE']:.4f}, RMSE={metrics['RMSE']:.4f}, MAPE={metrics['MAPE']:.2f}%")
    
    if save_results:
        with open(save_results, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {save_results}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='ACTFormer Evaluation')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--save_results', type=str,
                       help='Path to save evaluation results')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Evaluate
    evaluate(config, args.checkpoint, args.save_results)


if __name__ == '__main__':
    main()
