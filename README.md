# yuhen-model-v2

A molecular toxicity classifier that predicts whether a chemical compound is toxic or non-toxic. It uses cheminformatics fingerprints (mathematical descriptions of molecular structure) as input features to a fully-connected neural network trained with stratified k-fold cross-validation.

This is a clean, reproducible rewrite of the original `yuhen-model` project, consolidating four near-identical codebases into a single maintainable pipeline. It matches the original study exactly for reproducibility purposes.

---

## Table of Contents

1. [What This Does](#what-this-does)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Input Data Format](#input-data-format)
6. [Featurization Methods](#featurization-methods)
7. [Split Methods](#split-methods)
8. [Full CLI Reference](#full-cli-reference)
9. [Output Files](#output-files)
10. [Analysis Notebooks](#analysis-notebooks)
11. [Model Architecture](#model-architecture)
12. [Project Structure](#project-structure)
13. [Advanced Configuration](#advanced-configuration)

---

## What This Does

Given a dataset of molecules labeled as toxic or non-toxic, the pipeline:

1. **Loads** your Excel files (molecules described as SMILES strings or InChI identifiers)
2. **Converts** molecule identifiers into numerical feature vectors (fingerprints or descriptors)
3. **Augments** the training set by generating alternate representations of the same molecule
4. **Trains** a neural network using 5-fold cross-validation and early stopping
5. **Evaluates** the best model on two held-out test sets
6. **Saves** results (metrics, plots, model weights) to an output directory

> **New to cheminformatics?**
> - A **SMILES** string (e.g. `CCO` for ethanol) is a text representation of a molecule's structure.
> - An **InChI** is an alternative standardized identifier for the same purpose.
> - A **fingerprint** encodes which structural features are present in a molecule as a binary vector (0/1 for each feature).

---

## Requirements

| Requirement | Notes |
|---|---|
| Python ≥ 3.11 | Any recent Python 3.11+ works |
| [uv](https://github.com/astral-sh/uv) | Fast Python package manager (replaces pip+venv) |
| GPU (optional) | PyTorch will use CUDA automatically if available; CPU works fine |

---

## Installation

**1. Install `uv`** (if you haven't already):

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**2. Clone and install dependencies:**

```bash
git clone <repo-url>
cd yuhen-model-v2
uv sync
```

This creates a virtual environment and installs all dependencies automatically. You do not need to `pip install` anything manually.

> **Note on DeepChem (PubChem fingerprints):** `deepchem` is a large optional dependency needed only when `--feature-type pubchem` is used. Install it separately if needed:
> ```bash
> uv pip install deepchem
> ```

---

## Quick Start

Place your data files in an `input/` directory, then run:

```bash
uv run python main.py \
    --data-dir    input \
    --feature-type morgan \
    --output-dir  output
```

Training progress is logged to the console. When done, results appear in `output/`.

To try a different fingerprint type:

```bash
uv run python main.py --feature-type macc --output-dir output_macc
```

To use scaffold-based splitting (see [Split Methods](#split-methods)):

```bash
uv run python main.py --split-method scaffold --output-dir output_scaffold
```

---

## Input Data Format

Three Excel files are required, placed in `--data-dir`:

| File | Purpose |
|---|---|
| `Train.xlsx` | Training set (used for cross-validation) |
| `Test.xlsx` | Primary held-out test set (reported as "Test 1") |
| `Test2.xlsx` | Secondary held-out test set (reported as "Test 2") |

Each file must have at least two columns:

| Column | Description |
|---|---|
| `Smile` | SMILES string for the molecule (use `--input-type smile`) |
| `Inchi` | InChI identifier (use `--input-type inchi`) — auto-detected if present |
| `Toxicity` | Binary label: `0` = non-toxic, `1` = toxic |

The first column (row index) is ignored automatically.

> **Tip:** The column name is auto-detected. If both `Smile` and `Inchi` columns are present, `Inchi` takes precedence.

---

## Featurization Methods

Choose one with `--feature-type`:

| Value | Dimensions | Description |
|---|---|---|
| `morgan` | 2 048 | Morgan circular fingerprints (radius 2). Fast, widely used, captures local atom environments. |
| `macc` | 167 | MACCS structural keys. A fixed dictionary of 166 chemical substructure patterns. |
| `combined` | 2 215 | Morgan + MACCS concatenated. Broader coverage than either alone. |
| `mordred` | ~1 600 | Mordred molecular descriptors. Physicochemical properties (logP, MW, ring counts, …). Slower to compute. Uses StandardScaler normalization fit on all splits combined. Augmentation uses stereo-isomers of toxic molecules only. |
| `pubchem` | 881 | PubChem fingerprints via DeepChem. Requires `deepchem` installed and an internet connection (~1–2 s per molecule for the first call). |

**Training augmentation:** To improve model robustness, each training molecule is expanded into multiple equivalent representations:

- `morgan`, `macc`, `combined`, `pubchem`: Each molecule generates `--aug-n-smiles` (default 50) random SMILES strings encoding the same structure.
- `mordred`: Toxic molecules (label = 1) are expanded into all valid stereo-isomers; non-toxic molecules are kept as canonical SMILES. This matches the original study.

Test molecules are always canonicalized (normalized to a single standard form) without augmentation.

---

## Split Methods

Choose with `--split-method`:

### `random` (default)

Uses the pre-split `Train.xlsx` / `Test.xlsx` files as-is. Training set is augmented; test sets are only canonicalized.

### `scaffold`

Combines `Train.xlsx` + `Test.xlsx` into a single pool of molecules, then splits them by **Bemis-Murcko scaffold** — the core ring system shared by structurally similar molecules. This is a stricter evaluation: molecules with scaffolds not seen during training cannot leak into the test set.

```bash
uv run python main.py \
    --split-method scaffold \
    --scaffold-test-frac 0.12 \
    --output-dir output/scaffold
```

Results are written to `<output-dir>/scaffold/`.

> **Why scaffold split?** Random splitting can accidentally put very similar molecules in both train and test, making results look better than they are. Scaffold splitting is a more realistic test of how well the model generalizes to structurally novel compounds.

---

## Full CLI Reference

```
uv run python main.py [options]
```

| Argument | Default | Choices | Description |
|---|---|---|---|
| `--data-dir` | `input` | — | Directory with `Train.xlsx`, `Test.xlsx`, `Test2.xlsx` |
| `--feature-type` | `morgan` | `morgan`, `macc`, `combined`, `mordred`, `pubchem` | Molecular featurization method |
| `--input-type` | `smile` | `smile`, `inchi` | Column type to read from Excel |
| `--output-dir` | `output` | — | Where to write results, plots, and model weights |
| `--seed` | `42` | — | Random seed for full reproducibility |
| `--aug-n-smiles` | `50` | — | Random SMILES per training molecule (not used for mordred) |
| `--n-folds` | `5` | — | Number of stratified cross-validation folds |
| `--n-epochs` | `10000` | — | Maximum training epochs (early stopping usually triggers first) |
| `--batch-size` | `128` | — | Mini-batch size |
| `--lr` | `0.01` | — | Initial learning rate (Adam optimizer) |
| `--split-method` | `random` | `random`, `scaffold` | How to split molecules into train / test |
| `--scaffold-test-frac` | `0.12` | — | Fraction held out for scaffold test set |

---

## Output Files

```
output/
├── model/
│   └── model.pth               # Best model weights (PyTorch state_dict)
├── plots/
│   ├── Fold1-Loss.png          # Training vs. validation loss curve
│   ├── Fold1-Metrics.png       # F1 and MCC across epochs
│   └── ...                     # Repeated for each fold (Fold2, Fold3, …)
└── table/
    ├── cross_validation.xlsx   # Per-fold train/val/test metrics
    ├── Result1.xlsx            # Final metrics on Test set
    └── Result2.xlsx            # Final metrics on Test2 set
```

For scaffold split, everything is nested under `output/scaffold/` and `Result_scaffold_test.xlsx` replaces `Result1.xlsx`.

**Reported metrics** (per file):

| Metric | Description |
|---|---|
| TP / FP / FN / TN | Confusion matrix values |
| F1 Score | Harmonic mean of precision and recall |
| Precision | Fraction of predicted toxic that are truly toxic |
| Recall / Sensitivity | Fraction of true toxic found |
| Specificity | Fraction of true non-toxic correctly identified |
| Accuracy | Overall fraction correct |
| MCC | Matthews Correlation Coefficient (robust to class imbalance) |
| AUROC | Area under the ROC curve |
| Balanced Accuracy | Average of sensitivity and specificity |

---

## Analysis Notebooks

Six interactive notebooks are provided for exploratory analysis and post-training inspection. All are built with [Marimo](https://marimo.io/) — a reactive Python notebook where changing any UI control instantly re-runs dependent cells.

Launch any notebook with:

```bash
uv run marimo edit notebooks/<name>.py
```

### `results_viz.py` — Cross-validation results

Visualizes the training history saved in `output/`:
- Per-fold loss and metric curves
- Bar chart of mean ± SD metrics across folds

```bash
uv run marimo edit notebooks/results_viz.py
```

### `shap_analysis.py` — Feature importance (SHAP)

Loads the saved model and computes [SHAP values](https://shap.readthedocs.io/) to explain which fingerprint bits most influence predictions:
- Loads `model.pth` + test data
- KernelExplainer summary plot
- Interactive controls for model path, data directory, and feature type

```bash
uv run marimo edit notebooks/shap_analysis.py
```

### `pca_analysis.py` — Chemical space visualization

Projects molecular fingerprints into 2D using PCA and t-SNE to visualize how separable toxic and non-toxic molecules are in feature space:
- PCA cumulative explained-variance plot (how many components capture 90 % / 95 % of variance)
- PCA 2D scatter colored by class and by train/test split
- t-SNE 2D scatter with adjustable perplexity and sample size

```bash
uv run marimo edit notebooks/pca_analysis.py
```

### `tanimoto_analysis.py` — Chemical similarity distributions

Computes pairwise Tanimoto similarity within and across toxic/non-toxic classes:
- Violin plots comparing within-class vs. cross-class similarity
- Overlay histogram of all three distributions
- Mann-Whitney U statistical test with p-values
- Supports Morgan, MACCS, Topological Torsion, and PubChem fingerprints

```bash
uv run marimo edit notebooks/tanimoto_analysis.py
```

### `structure_alerts.py` — Structural alert analysis

Screens molecules against known toxicophore substructure patterns (structural alerts):
- Counts how many molecules trigger each alert
- Breaks down alert frequency by toxic vs. non-toxic class

```bash
uv run marimo edit notebooks/structure_alerts.py
```

### `ml_models.py` — Baseline machine learning models

Trains and compares scikit-learn classifiers (Random Forest, Gradient Boosting, SVM, etc.) on the same data as the neural network:
- Useful for a quick baseline or sanity check
- Uses the same featurization pipeline as `main.py`
- Includes RandomizedSearchCV hyperparameter tuning

```bash
uv run marimo edit notebooks/ml_models.py
```

---

## Model Architecture

A fully-connected neural network with six hidden layers, batch normalization, and dropout:

```
Input (fingerprint vector)
  → FC(1024) → BatchNorm → ReLU → Dropout(0.5)
  → FC(512)  → BatchNorm → ReLU → Dropout(0.5)
  → FC(256)  → BatchNorm → ReLU → Dropout(0.5)
  → FC(128)  → BatchNorm → ReLU → Dropout(0.5)
  → FC(64)   → BatchNorm → ReLU → Dropout(0.5)
  → FC(32)   → BatchNorm → ReLU → Dropout(0.5)
  → FC(1) → Sigmoid → probability [0, 1]
```

**Training details:**

| Setting | Value |
|---|---|
| Loss function | Binary cross-entropy + L1 regularization |
| L1 lambda | 0.0005 |
| Optimizer | Adam |
| Weight decay (L2) | 1e-5 |
| Initial learning rate | 0.01 |
| LR scheduler | ReduceLROnPlateau (factor 0.5, patience 3) |
| Early stopping | patience 16 epochs on validation loss |
| Best model selection | Checkpoint at lowest validation loss |

**Cross-validation:** Stratified k-fold (default k=5) on the training set. The fold whose model achieves the best validation F1 score is selected as the final model.

---

## Project Structure

```
yuhen-model-v2/
├── main.py                        # CLI entry point — run this to train
├── pyproject.toml                 # Dependencies and project metadata
├── notebooks/
│   ├── results_viz.py             # Marimo: CV results visualization
│   ├── shap_analysis.py           # Marimo: SHAP feature importance
│   ├── pca_analysis.py            # Marimo: PCA / t-SNE chemical space
│   ├── tanimoto_analysis.py       # Marimo: Tanimoto similarity distributions
│   ├── structure_alerts.py        # Marimo: Structural alert screening
│   └── ml_models.py               # Marimo: Baseline ML model comparison
└── src/
    ├── config.py                  # ExperimentConfig dataclass (all hyperparameters)
    ├── data/
    │   ├── loader.py              # Loads Excel files, auto-detects column names
    │   ├── augment.py             # SMILES augmentation, stereo-isomers, InChI→SMILES
    │   ├── featurize.py           # Computes fingerprints and descriptors
    │   └── scaffold_split.py      # Bemis-Murcko scaffold-based train/test split
    ├── model/
    │   └── network.py             # FCModel neural network definition
    ├── training/
    │   ├── cross_val.py           # K-fold orchestration, fold selection
    │   └── loops.py               # Per-epoch train / validate / test loops
    ├── evaluate/
    │   └── metrics.py             # Metric computation and Excel export
    └── utils/
        └── seed.py                # seed_everything() for reproducibility
```

---

## Advanced Configuration

All hyperparameters map to the `ExperimentConfig` dataclass in `src/config.py`. Parameters not exposed as CLI flags can be edited there directly.

| Parameter | Default | Description |
|---|---|---|
| `morgan_radius` | `2` | Radius for Morgan fingerprint (higher = larger neighbourhoods) |
| `morgan_nbits` | `2048` | Bit-vector length for Morgan fingerprint |
| `weight_decay` | `1e-5` | L2 regularization coefficient |
| `l1_lambda` | `0.0005` | L1 regularization weight |
| `lr_patience` | `3` | Epochs without improvement before LR is halved |
| `early_stop_patience` | `16` | Epochs without improvement before training stops |
