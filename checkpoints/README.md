# Model Checkpoints

This directory contains pre-trained model checkpoints for ACTFormer.

## Download Instructions

Due to file size limitations, model checkpoints are hosted externally. Please download them from:

**Google Drive**: [Link to be added]  
**Baidu Pan**: [Link to be added]

## Available Checkpoints

| Dataset | Model | Size | SHA-256 Hash |
|---------|-------|------|--------------|
| PEMS04 | best_model.pth | ~XX MB | `[hash to be added]` |
| PEMS07 | best_model.pth | ~XX MB | `[hash to be added]` |
| PEMS08 | best_model.pth | ~XX MB | `[hash to be added]` |
| NYC-Taxi (Drop) | best_model.pth | ~XX MB | `[hash to be added]` |
| NYC-Taxi (Pick) | best_model.pth | ~XX MB | `[hash to be added]` |

## File Structure

After downloading, organize checkpoints as follows:

```
checkpoints/
├── pems04/
│   ├── stage1_best.pth
│   └── best_model.pth
├── pems07/
│   ├── stage1_best.pth
│   └── best_model.pth
├── pems08/
│   ├── stage1_best.pth
│   └── best_model.pth
├── nyctaxi_drop/
│   ├── stage1_best.pth
│   └── best_model.pth
└── nyctaxi_pick/
    ├── stage1_best.pth
    └── best_model.pth
```

## Verification

To verify the integrity of downloaded files, use:

```bash
sha256sum checkpoints/pems04/best_model.pth
```

Compare the output with the hash listed above.

## Usage

Load a checkpoint in your code:

```python
import torch

checkpoint = torch.load('checkpoints/pems04/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
```

Or use the evaluation script:

```bash
python src/eval.py --config configs/pems04.yaml --checkpoint checkpoints/pems04/best_model.pth
```

## Notes

- `stage1_best.pth`: Best model from Stage 1 (pre-training)
- `best_model.pth`: Best model from Stage 2 (fine-tuning) - used for final evaluation
- All checkpoints are trained with the configurations in `configs/`
- Models are compatible with PyTorch >= 1.10.0
