# ACTFormer: Adaptive Context-aware Temporal Transformer for Traffic Forecasting

This repository contains the official implementation of **ACTFormer**, a novel deep learning model for spatiotemporal traffic forecasting.

## Overview

ACTFormer leverages adaptive context-aware mechanisms and temporal transformers to capture complex spatiotemporal dependencies in traffic data, achieving state-of-the-art performance on multiple benchmarks.

## Repository Structure

```
ACTFormer/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── configs/                     # Configuration files for different datasets
│   ├── pems04.yaml
│   ├── pems07.yaml
│   ├── pems08.yaml
│   ├── nyctaxi_drop.yaml
│   └── nyctaxi_pick.yaml
├── preprocess/                  # Data preprocessing scripts
│   ├── build_adj.py            # Build adjacency matrix
│   ├── make_splits.py          # Create train/val/test splits
│   └── normalize.py            # Data normalization
├── scripts/                     # Experiment scripts
│   ├── reproduce_main_table.sh # Reproduce main results
│   └── eval_all.sh             # Evaluate all models
├── src/                        # Source code
│   ├── model/                  # Model architecture
│   ├── data/                   # Data loaders
│   ├── train_stage1.py        # Stage 1 training
│   ├── train_stage2.py        # Stage 2 training
│   └── eval.py                # Evaluation script
├── checkpoints/                # Model checkpoints
│   └── README.md              # Download instructions
└── logs/                       # Training logs (optional)
```

## Installation

### Requirements

- Python >= 3.8
- PyTorch >= 1.10.0
- CUDA >= 11.0 (for GPU support)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ACTFormer.git
cd ACTFormer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Datasets

We evaluate ACTFormer on the following datasets:
- **PEMS04**: Traffic speed data from California highways
- **PEMS07**: Traffic speed data from California highways
- **PEMS08**: Traffic speed data from California highways
- **NYC-Taxi**: Taxi demand data (drop-off and pick-up)

### Data Preparation

1. Download datasets from [link_to_data]
2. Place raw data in `data/` directory
3. Run preprocessing:
```bash
python preprocess/build_adj.py --dataset pems04
python preprocess/make_splits.py --dataset pems04
python preprocess/normalize.py --dataset pems04
```

## Training

### Two-Stage Training

**Stage 1: Pre-training**
```bash
python src/train_stage1.py --config configs/pems04.yaml --gpu 0
```

**Stage 2: Fine-tuning**
```bash
python src/train_stage2.py --config configs/pems04.yaml --gpu 0 --pretrained checkpoints/stage1_best.pth
```

## Evaluation

Evaluate a trained model:
```bash
python src/eval.py --config configs/pems04.yaml --checkpoint checkpoints/best_model.pth
```

## Reproducing Results

To reproduce the main results from the paper:
```bash
bash scripts/reproduce_main_table.sh
```

To evaluate all models on all datasets:
```bash
bash scripts/eval_all.sh
```

## Model Checkpoints

Pre-trained model checkpoints are available. See [checkpoints/README.md](checkpoints/README.md) for download instructions and verification.

## Results

### Main Results on PEMS Datasets

| Dataset | Metric | Horizon 3 | Horizon 6 | Horizon 12 |
|---------|--------|-----------|-----------|------------|
| PEMS04  | MAE    | -         | -         | -          |
| PEMS07  | MAE    | -         | -         | -          |
| PEMS08  | MAE    | -         | -         | -          |

*Complete results available in the paper.*

## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{actformer2024,
  title={ACTFormer: Adaptive Context-aware Temporal Transformer for Traffic Forecasting},
  author={Your Name},
  journal={Journal Name},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or issues, please open an issue on GitHub or contact: your.email@example.com

## Acknowledgments

We thank the authors of the baseline methods and dataset providers for making their code and data publicly available.
