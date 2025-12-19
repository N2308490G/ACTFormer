# ACTFormer: Adaptive Complexity-Aware Traffic Transformer for Intelligent Flow Prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-ee4c2c.svg)](https://pytorch.org/)

**Paper Status**: Under Review  
**License**: MIT

Official repository for "ACTFormer: Adaptive Complexity-Aware Traffic Transformer for Intelligent Flow Prediction" (Under review).

---

## 📢 Important Notice: Code Release Policy

**The complete implementation will be released after the paper is officially published.**

This repository currently maintains the project structure and documentation. Full source code, pre-trained models, and reproduction scripts will be made available immediately upon publication in IEEE TNNLS.

### Why Code Is Not Yet Available

#### 🚫 University Research Office Mandatory Policy

**Our university's research office has established strict regulations regarding code release for unpublished research:**

> **⚠️ Regulation**: All research code from unpublished papers is **strictly prohibited** from public release until formal publication.

> **📋 Code Release Procedure**:
> 1. Paper must be **officially published** first
> 2. Researcher submits **Code Release Application Form** to Research Office
> 3. Application undergoes institutional review process
> 4. Code can only be released **after approval is granted**

This is a **mandatory institutional requirement** that all researchers must comply with, regardless of personal preferences or academic norms in the field.

#### Additional Considerations

Beyond institutional requirements, several academic integrity factors also support delayed code release:

**1. Academic Integrity and Research Protection**
- Research has not undergone complete peer review
- Premature release risks unauthorized use or misinterpretation
- Protects ongoing peer review integrity

**2. Intellectual Property and Priority Rights**
- Novel algorithms (entropy-based complexity analysis, Gumbel-Softmax adaptive vocabulary selection)
- Ensures proper priority recognition in the field
- Prevents potential intellectual property disputes

**3. Review Process Requirements**
- Peer review may require experimental modifications
- Avoids version discrepancies that affect reproducibility
- Complies with journal pre-publication policies

---

## What Will Be Released

Upon official publication, we will immediately release:

### ✅ Complete Implementation
- Core ACTFormer modules (`complexity_analyzer.py`, `adaptive_vocab.py`, `traffic_encoder.py`)
- Two-stage training framework (`train_stage1.py`, `train_stage2.py`)
- Model architecture with Gumbel-Softmax mechanism
- Traffic-aware contextual encoding modules

### ✅ Pre-trained Models
- Checkpoint files for all 5 datasets (PeMS04/07/08, NYC-Taxi Drop/Pick)
- Stage 1 tokenization models
- Stage 2 forecasting models with optimal vocabulary selection

### ✅ Comprehensive Documentation
- Installation instructions
- Usage tutorials and examples
- API documentation  
- Two-stage training guides

### ✅ Reproducibility Materials
- Complete training configurations for all datasets
- Data preprocessing scripts (`build_adj.py`, `make_splits.py`, `normalize.py`)
- Hyperparameter settings (d_model=64, α=0.4, β=0.3, γ=0.3, τ=1.0)
- Evaluation protocols and metrics computation

---

## Key Results (From Paper)

### Traffic Forecasting Performance

#### Highway Traffic (PeMS Datasets)

| Dataset | Method | MAE | RMSE | MAPE | Improvement |
|---------|--------|-----|------|------|-------------|
| **PeMS04** | STGAFormer (SOTA) | 18.18 | 29.78 | 11.98% | - |
| | PDFormer | 18.32 | 29.97 | 12.10% | - |
| | **ACTFormer** | **16.45** | **27.23** | **10.89%** | **↓9.5%** |
| **PeMS07** | STGAFormer (SOTA) | 19.65 | 32.62 | 8.45% | - |
| | PDFormer | 19.83 | 32.87 | 8.53% | - |
| | **ACTFormer** | **16.78** | **29.12** | **6.95%** | **↓14.6%** |
| **PeMS08** | STGAFormer (SOTA) | 13.06 | 22.43 | 8.87% | - |
| | PDFormer | 13.58 | 23.51 | 9.05% | - |
| | **ACTFormer** | **11.92** | **20.45** | **7.68%** | **↓8.7%** |

#### Urban Mobility (NYC-Taxi)

| Task | Method | MAE | RMSE | MAPE | Improvement |
|------|--------|-----|------|------|-------------|
| **Drop-off** | STGAFormer (SOTA) | 4.76 | 8.89 | 33.45% | - |
| | PDFormer | 4.98 | 9.12 | 34.67% | - |
| | **ACTFormer** | **4.21** | **7.34** | **29.87%** | **↓11.6%** |
| **Pick-up** | STGAFormer (SOTA) | 4.98 | 9.12 | 34.23% | - |
| | PDFormer | 5.21 | 9.45 | 35.12% | - |
| | **ACTFormer** | **4.38** | **7.56** | **29.42%** | **↓12.0%** |

*Improvements calculated against the best baseline (STGAFormer) on each dataset*

### Comprehensive Comparison

ACTFormer outperforms **26 baseline methods** including:
- **Traditional**: LSTM, GRU
- **GNN Methods**: DCRNN, STGCN, MTGNN, MegaCRN, TIIDGCN (15 methods total)
- **Transformers**: GMAN, ASTGNN, PDFormer, STGAFormer (6 methods total)

### Model Efficiency

| Metric | Value |
|--------|-------|
| **Parameters** | 892K |
| **Training Time** | 41.23s/epoch |
| **Vocabulary Sizes** | {128, 512, 1024} tokens |
| **Complexity Correlation** | 0.84 (strong) |

### Ablation Study Impact

| Component Removed | PeMS04 MAE | PeMS08 MAE | Degradation |
|-------------------|------------|------------|-------------|
| Full ACTFormer | 16.45 | 11.89 | - |
| w/o Complexity Analysis | 18.12 | 13.01 | **+10.1%** |
| w/o Gumbel-Softmax | 18.45 | 13.23 | **+12.2%** |
| w/o Adaptive Vocabulary | 17.78 | 12.67 | **+8.1%** |
| w/o Traffic Context | 17.34 | 12.34 | **+5.4%** |

---

## Method Overview

ACTFormer performs **adaptive complexity-aware traffic forecasting** through a two-stage process:

```
Input Traffic → Complexity Analysis → Adaptive Vocab Selection → Traffic Encoding → Forecasting
  X(t)              ↓                     ↓                         ↓              Y(t+H)
                c = α·σ(X) +         Gumbel-Softmax          Multi-scale
                    β·H(X) +         {128,512,1024}          Conditioning
                    γ·F(X)
```

### Three-Stage Process:

1. **Complexity Analysis**: Entropy-based quantification
   - Temporal variance: σ(X)
   - Sample entropy: H(X)  
   - Frequency characteristics: F(X)
   - **Complexity score**: c = 0.4·σ(X) + 0.3·H(X) + 0.3·F(X)

2. **Adaptive Vocabulary Selection**: Differentiable Gumbel-Softmax
   - Low complexity (0.0-0.33) → 128 tokens (34.2% cases, +8.9% MAE gain)
   - Medium complexity (0.33-0.67) → 512 tokens (52.1% cases, +11.2% MAE gain)
   - High complexity (0.67-1.0) → 1024 tokens (13.7% cases, +15.6% MAE gain)

3. **Two-Stage Training**:
   - **Stage 1 (150 epochs)**: Learn complexity-aware tokenization
   - **Stage 2 (300 epochs)**: Enhanced forecasting with early stopping

**Key Innovation**: Per-complexity-level learnable vocabulary sizes adapt model capacity to traffic pattern complexity, achieving strong correlation (0.84) between complexity scores and optimal vocabularies.

---

## Datasets

| Dataset | Nodes | Interval | Timesteps | Period | Type |
|---------|-------|----------|-----------|--------|------|
| **PeMS04** | 307 | 5 min | 16,992 | 01/2018-02/2018 | Highway |
| **PeMS07** | 883 | 5 min | 28,224 | 05/2017-08/2017 | Highway |
| **PeMS08** | 170 | 5 min | 17,856 | 07/2016-08/2016 | Highway |
| **NYC-Taxi** | 266 | 30 min | 4,392 | 01/2014-12/2014 | Urban |

**Data Split**: 60% Train / 20% Validation / 20% Test  
**Normalization**: Z-score  
**Window**: 12 timesteps input → 12 timesteps prediction

---

```bash
# Clone repository
git clone https://github.com/N2308490G/ACTFormer.git
cd ACTFormer

# Install dependencies
pip install -r requirements.txt

# Install ACTFormer
pip install -e .
```

**Requirements**: Python ≥ 3.8, PyTorch ≥ 1.10.0, PyTorch Geometric ≥ 2.0.0

---

## Timeline

- **[2025/12]** Paper submitted (under double-blind review)
- **[2025/12]** Repository structure created
- **[TBD]** Paper acceptance notification
- **[TBD]** Complete code release immediately upon publication

---

## Contact

For inquiries about the paper or code release:

- **Watch this repository** for updates
- **Open an issue** for questions
- Email: [to be added upon publication]

We appreciate your understanding and support in maintaining research integrity. We are committed to open science and will release all materials promptly upon publication.

---

## Related Baseline Methods

This work compares with **26 state-of-the-art baselines**:

### Traditional Methods (3)
- LSTM, GRU, Historical Average

### Graph Neural Networks (15)
- DCRNN, STGCN, ASTGCN, GWN
- MTGNN, AGCRN, STSGCN, STG-NCDE
- DSTAGNN, DGCRN, MSTFGRN, MegaCRN
- AFDGCN, STGAGRTN, TIIDGCN

### Transformer Methods (6)
- STTN, GMAN, TFormer
- ASTGNN, PDFormer, STGAFormer

See paper for comprehensive related work discussion.

---

## Acknowledgments

We thank:
- The authors of baseline methods for making their code publicly available
- PeMS (California DoT) and NYC TLC for providing traffic datasets
- The research community for valuable feedback
- Reviewers for constructive comments

---

**Expected Code Release**: Immediately upon paper acceptance and publication

**License**: MIT License (permissive open-source)

**Last Updated**: December 2025
