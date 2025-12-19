#!/bin/bash
# Reproduce main results from paper
# This script trains and evaluates ACTFormer on all datasets

set -e  # Exit on error

echo "========================================="
echo "Reproducing ACTFormer Main Results"
echo "========================================="

# List of datasets
DATASETS=("pems04" "pems07" "pems08" "nyctaxi_drop" "nyctaxi_pick")

# Create results directory
mkdir -p results

for dataset in "${DATASETS[@]}"; do
    echo ""
    echo "========================================="
    echo "Processing dataset: $dataset"
    echo "========================================="
    
    # Stage 1: Pre-training
    echo "Stage 1: Pre-training..."
    python src/train_stage1.py \
        --config configs/${dataset}.yaml \
        --gpu 0 \
        2>&1 | tee logs/${dataset}_stage1.log
    
    # Stage 2: Fine-tuning
    echo "Stage 2: Fine-tuning..."
    python src/train_stage2.py \
        --config configs/${dataset}.yaml \
        --gpu 0 \
        --pretrained checkpoints/${dataset}/stage1_best.pth \
        2>&1 | tee logs/${dataset}_stage2.log
    
    # Evaluation
    echo "Evaluating..."
    python src/eval.py \
        --config configs/${dataset}.yaml \
        --checkpoint checkpoints/${dataset}/best_model.pth \
        --save_results results/${dataset}_results.json \
        2>&1 | tee logs/${dataset}_eval.log
    
    echo "Completed: $dataset"
done

echo ""
echo "========================================="
echo "All experiments completed!"
echo "Results saved in results/"
echo "========================================="
