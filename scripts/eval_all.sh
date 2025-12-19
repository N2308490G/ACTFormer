#!/bin/bash
# Evaluate all trained models on all datasets

set -e

echo "========================================="
echo "Evaluating All Models"
echo "========================================="

DATASETS=("pems04" "pems07" "pems08" "nyctaxi_drop" "nyctaxi_pick")

mkdir -p results

for dataset in "${DATASETS[@]}"; do
    echo ""
    echo "Evaluating: $dataset"
    
    if [ -f "checkpoints/${dataset}/best_model.pth" ]; then
        python src/eval.py \
            --config configs/${dataset}.yaml \
            --checkpoint checkpoints/${dataset}/best_model.pth \
            --save_results results/${dataset}_results.json \
            --verbose
    else
        echo "Warning: Checkpoint not found for ${dataset}"
    fi
done

echo ""
echo "========================================="
echo "Evaluation completed!"
echo "Aggregating results..."
echo "========================================="

# Optional: Aggregate all results into a single summary
python -c "
import json
import os
from glob import glob

results = {}
for f in glob('results/*_results.json'):
    dataset = os.path.basename(f).replace('_results.json', '')
    with open(f, 'r') as fp:
        results[dataset] = json.load(fp)

with open('results/summary.json', 'w') as f:
    json.dump(results, f, indent=2)

print('Summary saved to results/summary.json')
"

echo "Done!"
