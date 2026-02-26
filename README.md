# EchoQual: Self-Supervised Echocardiogram Image Quality Assessment

**Leveraging Foundation Model Representations Without Expert Labels**

This codebase implements a fully self-supervised framework for echocardiogram quality grading
that eliminates the need for human-annotated quality scores. It derives quality signals from
pretrained foundation model embeddings using three complementary proxy strategies.

## Method Overview

Three proxy signals are fused into a composite quality score:

1. **View Classification Confidence** — Softmax entropy from a view classifier as an inverse quality proxy
2. **Embedding Density Scoring** — k-NN distance and Mahalanobis distance to view-specific cluster centroids in the contrastive embedding space
3. **Vision-Language Alignment** — Cosine similarity between video embeddings and canonical text descriptions of well-acquired standard views

These signals require **zero quality labels** for training. Expert annotations are used
**only for validation** to measure rank correlation.

## Dataset

This project uses the [EchoNet-Dynamic](https://echonet.github.io/dynamic/) dataset
(10,030 apical-4-chamber echocardiogram videos). The code is designed to work with a
configurable subset for rapid prototyping.

### Expected data layout

```
data/
└── EchoNet-Dynamic/
    ├── Videos/
    │   ├── 0X1A0A263B22CCD966.avi
    │   ├── ...
    ├── FileList.csv
    └── VolumeTracings.csv
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
# 1. Extract embeddings from EchoNet-Dynamic (subset)
python scripts/extract_embeddings.py \
    --data_dir data/EchoNet-Dynamic \
    --output_dir outputs/embeddings \
    --max_videos 500

# 2. Compute self-supervised quality scores
python scripts/compute_quality_scores.py \
    --embeddings_dir outputs/embeddings \
    --output_dir outputs/scores

# 3. Evaluate against proxy ground truth
python scripts/evaluate.py \
    --scores_path outputs/scores/quality_scores.csv \
    --data_dir data/EchoNet-Dynamic

# Or run the full pipeline:
python scripts/run_pipeline.py \
    --data_dir data/EchoNet-Dynamic \
    --output_dir outputs \
    --max_videos 500
```

## Project Structure

```
echo_quality/
├── README.md
├── requirements.txt
├── configs/
│   └── default.yaml              # All hyperparameters
├── src/
│   ├── __init__.py
│   ├── dataset.py                # EchoNet-Dynamic data loading
│   ├── encoder.py                # Video encoder (R3D / ViViT backbone)
│   ├── ssl_pretraining.py        # Self-supervised contrastive pretraining
│   ├── quality_scorer.py         # Three proxy quality signals + fusion
│   ├── canonical_texts.py        # Canonical view descriptions for VL alignment
│   ├── evaluation.py             # Rank correlation & visualization
│   └── utils.py                  # Shared utilities
├── scripts/
│   ├── extract_embeddings.py     # Step 1: encode videos
│   ├── compute_quality_scores.py # Step 2: compute scores
│   ├── evaluate.py               # Step 3: evaluate
│   └── run_pipeline.py           # End-to-end runner
├── tests/
│   └── test_pipeline.py          # Unit tests with synthetic data
└── docs/
    └── METHODS.md                # Detailed methodology
```

## Citation

If you use this codebase, please cite:

```bibtex
@misc{echoqual2026,
  title={Self-Supervised Echocardiogram Image Quality Assessment},
  year={2026},
  note={Built upon EchoPrime and EchoNet-Dynamic}
}
```

## License

MIT License. See LICENSE for details.

## Acknowledgments

- [EchoPrime](https://github.com/echonet/EchoPrime) — Vukadinovic et al., Nature 2025
- [EchoNet-Dynamic](https://echonet.github.io/dynamic/) — Ouyang et al., Nature 2020
- [EchoCLR](https://arxiv.org/abs/2207.11581) — Holste et al., Commun. Med. 2024
