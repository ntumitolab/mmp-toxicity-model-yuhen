#!/usr/bin/env python
"""
yuhen-model-v2  —  entry point

Usage
-----
uv run python main.py \
    --data-dir   model/input \
    --feature-type morgan \
    --input-type   smile \
    --output-dir   output/ \
    --seed         42
"""
import argparse
import logging
import os
import sys

import torch

from src.config import ExperimentConfig
from src.data.augment import augment_smiles, augment_stereo_smiles, canonicalize_smiles, inchi_to_smiles
from src.data.featurize import featurize, featurize_splits
from src.data.loader import load_combined, load_splits
from src.data.scaffold_split import scaffold_split
from src.evaluate.metrics import compute_metrics, save_to_excel
from src.training.cross_val import run_kfold
from src.utils.seed import seed_everything

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="yuhen-model-v2: reproducible molecular toxicity classifier"
    )
    p.add_argument("--data-dir",     default="input",  help="Directory with Train/Test/Test2.xlsx")
    p.add_argument("--feature-type", default="morgan",
                   choices=["morgan", "macc", "combined", "mordred", "pubchem"],
                   help="Fingerprint / descriptor type")
    p.add_argument("--input-type",   default="smile",
                   choices=["smile", "inchi"],
                   help="Molecule identifier column type")
    p.add_argument("--output-dir",   default="output", help="Where to write results")
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--aug-n-smiles", type=int, default=50,
                   help="Random SMILES per training molecule")
    p.add_argument("--n-folds",      type=int, default=5)
    p.add_argument("--n-epochs",     type=int, default=10000)
    p.add_argument("--batch-size",   type=int, default=128)
    p.add_argument("--lr",           type=float, default=0.01)
    p.add_argument("--split-method", default="random",
                   choices=["random", "scaffold"],
                   help="'random' uses pre-split Train/Test files; "
                        "'scaffold' combines Train_inchi+Test then re-splits by Bemis-Murcko scaffold")
    p.add_argument("--scaffold-test-frac", type=float, default=0.12,
                   help="Fraction of molecules for the scaffold-held-out test set "
                        "(default: 0.12, matching the original Train/Test file ratio)")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Scaffold-split helper
# ---------------------------------------------------------------------------

def _run_scaffold(args: argparse.Namespace, cfg) -> None:
    """Full pipeline for --split-method scaffold.

    Combines Train_inchi.xlsx + Test.xlsx into one pool, performs a
    Bemis-Murcko scaffold split, then trains and evaluates exactly like the
    random-split path.  Outputs go to ``<output_dir>/scaffold/``.
    """
    from src.training.loops import test_epoch

    scaffold_output_dir = os.path.join(cfg.output_dir, "scaffold")

    # 1. Load combined pool + external holdout
    logger.info("Loading combined data from %s …", cfg.data_dir)
    X_train_raw, y_train_raw, X_test_raw, y_test_raw, X_test2_raw, y_test2_raw, detected_cols = (
        load_combined(cfg.data_dir)
    )
    logger.info(
        "Raw counts — Train_inchi: %d  Test: %d  Test2: %d",
        len(X_train_raw), len(X_test_raw), len(X_test2_raw),
    )

    # 2. Convert InChI → SMILES for each split as needed
    def _maybe_convert(X, y, col, name):
        if col == "Inchi":
            logger.info("Converting %s InChI → SMILES …", name)
            return inchi_to_smiles(X, y)
        return X, y

    X_train_smi, y_train_smi = _maybe_convert(X_train_raw, y_train_raw, detected_cols["Train"], "Train")
    X_test_smi,  y_test_smi  = _maybe_convert(X_test_raw,  y_test_raw,  detected_cols["Test"],  "Test")
    X_test2_smi, y_test2_smi = _maybe_convert(X_test2_raw, y_test2_raw, detected_cols["Test2"], "Test2")

    # 3. Combine Train + Test into a single pool for scaffold splitting
    logger.info(
        "Combining Train (%d) + Test (%d) into pool of %d molecules …",
        len(X_train_smi), len(X_test_smi), len(X_train_smi) + len(X_test_smi),
    )
    X_pool = X_train_smi + X_test_smi
    y_pool = y_train_smi + y_test_smi

    # 4. Scaffold split
    logger.info(
        "Scaffold split (test_frac=%.2f, seed=%d) …",
        cfg.scaffold_test_frac, cfg.seed,
    )
    X_sc_train, y_sc_train, X_sc_test, y_sc_test = scaffold_split(
        X_pool, y_pool, test_frac=cfg.scaffold_test_frac, seed=cfg.seed,
    )

    # 5. Augment scaffold-train; canonicalize both test sets
    # Mordred uses stereo-isomer augmentation for toxic class only (original behaviour)
    if cfg.feature_type == "mordred":
        logger.info("Augmenting scaffold-train with stereo-isomers (mordred mode) …")
        X_train_aug, y_train_aug = augment_stereo_smiles(X_sc_train, y_sc_train)
    else:
        logger.info("Augmenting scaffold-train SMILES (n_smiles=%d) …", cfg.aug_n_smiles)
        X_train_aug, y_train_aug = augment_smiles(
            X_sc_train, y_sc_train, n_smiles=cfg.aug_n_smiles, seed=cfg.seed,
        )
    X_sc_test_can, y_sc_test_can = canonicalize_smiles(X_sc_test,  y_sc_test)
    X_test2_can,   y_test2_can   = canonicalize_smiles(X_test2_smi, y_test2_smi)

    logger.info(
        "After augmentation — train: %d  scaffold_test: %d  test2: %d",
        len(X_train_aug), len(X_sc_test_can), len(X_test2_can),
    )

    # 6. Featurize
    # Mordred: fit StandardScaler on all splits combined (original behaviour)
    logger.info("Featurizing with '%s' …", cfg.feature_type)
    if cfg.feature_type == "mordred":
        (X_train, y_train_np), (X_sc_test_feat, y_sc_test_np), (X_test2_feat, y_test2_np) = (
            featurize_splits(
                [(X_train_aug, y_train_aug), (X_sc_test_can, y_sc_test_can), (X_test2_can, y_test2_can)],
                "mordred",
            )
        )
    else:
        X_train,   y_train_np   = featurize(
            X_train_aug, y_train_aug, cfg.feature_type,
            morgan_radius=cfg.morgan_radius, morgan_nbits=cfg.morgan_nbits,
        )
        X_sc_test_feat, y_sc_test_np = featurize(
            X_sc_test_can, y_sc_test_can, cfg.feature_type,
            morgan_radius=cfg.morgan_radius, morgan_nbits=cfg.morgan_nbits,
        )
        X_test2_feat, y_test2_np = featurize(
            X_test2_can, y_test2_can, cfg.feature_type,
            morgan_radius=cfg.morgan_radius, morgan_nbits=cfg.morgan_nbits,
        )

    logger.info(
        "Feature matrix shapes — train: %s  scaffold_test: %s  test2: %s",
        X_train.shape, X_sc_test_feat.shape, X_test2_feat.shape,
    )

    # 7. Cross-validation (scaffold-test acts as test1; Test2 as test2)
    logger.info("Starting %d-fold cross-validation (scaffold mode) …", cfg.n_folds)
    best_model, sc_test_loader, test_loader2 = run_kfold(
        X_train,        y_train_np,
        X_sc_test_feat, y_sc_test_np,
        X_test2_feat,   y_test2_np,
        cfg, scaffold_output_dir,
    )

    # 8. Save best model
    model_path = os.path.join(scaffold_output_dir, "model", "model.pth")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(best_model.state_dict(), model_path)
    logger.info("Model saved to %s", model_path)

    # 9. Final evaluation
    device = next(best_model.parameters()).device
    te1 = test_epoch(best_model, sc_test_loader, cfg.l1_lambda, device)
    te2 = test_epoch(best_model, test_loader2,   cfg.l1_lambda, device)

    metrics1 = compute_metrics(te1.y_true, te1.y_pred)
    metrics2 = compute_metrics(te2.y_true, te2.y_pred)

    table_dir = os.path.join(scaffold_output_dir, "table")
    save_to_excel(metrics1, os.path.join(table_dir, "Result_scaffold_test.xlsx"))
    save_to_excel(metrics2, os.path.join(table_dir, "Result2.xlsx"))

    logger.info("Scaffold-test  F1=%.4f  MCC=%.4f", metrics1["F1 Score"], metrics1["MCC"])
    logger.info("Test2          F1=%.4f  MCC=%.4f", metrics2["F1 Score"], metrics2["MCC"])
    logger.info("Done (scaffold mode).")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    args = _parse_args(argv)

    # Build config from CLI
    cfg = ExperimentConfig(
        feature_type        = args.feature_type,
        aug_n_smiles        = args.aug_n_smiles,
        n_folds             = args.n_folds,
        n_epochs            = args.n_epochs,
        batch_size          = args.batch_size,
        lr                  = args.lr,
        seed                = args.seed,
        output_dir          = args.output_dir,
        data_dir            = args.data_dir,
        split_method        = args.split_method,
        scaffold_test_frac  = args.scaffold_test_frac,
    )

    seed_everything(cfg.seed)
    logger.info("Seed set to %d", cfg.seed)
    logger.info("Config: %s", cfg)

    # -----------------------------------------------------------------------
    # Scaffold-split mode
    # -----------------------------------------------------------------------
    if cfg.split_method == "scaffold":
        _run_scaffold(args, cfg)
        return

    # -----------------------------------------------------------------------
    # Random-split mode (original behaviour)
    # -----------------------------------------------------------------------

    # Determine input column name
    input_col = "Smile" if args.input_type == "smile" else "Inchi"

    # 1. Load raw splits
    logger.info("Loading data from %s …", cfg.data_dir)
    X_train_raw, y_train, X_test_raw, y_test, X_test2_raw, y_test2, detected_cols = load_splits(
        cfg.data_dir, input_col=input_col
    )

    logger.info(
        "Raw counts — train: %d  test: %d  test2: %d",
        len(X_train_raw), len(X_test_raw), len(X_test2_raw),
    )

    # 2. Convert InChI → SMILES per split, based on what column was actually loaded
    def _maybe_convert(X, y, col, name):
        if col == "Inchi":
            logger.info("Converting %s InChI → SMILES …", name)
            return inchi_to_smiles(X, y)
        return X, y

    X_train_raw, y_train = _maybe_convert(X_train_raw, y_train, detected_cols["Train"], "Train")
    X_test_raw,  y_test  = _maybe_convert(X_test_raw,  y_test,  detected_cols["Test"],  "Test")
    X_test2_raw, y_test2 = _maybe_convert(X_test2_raw, y_test2, detected_cols["Test2"], "Test2")

    logger.info(
        "After InChI conversion — train: %d  test: %d  test2: %d",
        len(X_train_raw), len(X_test_raw), len(X_test2_raw),
    )

    # 3. Augment training set; canonicalize test sets
    # Mordred uses stereo-isomer augmentation for toxic class only (original behaviour)
    if cfg.feature_type == "mordred":
        logger.info("Augmenting training set with stereo-isomers (mordred mode) …")
        X_train_aug, y_train_aug = augment_stereo_smiles(X_train_raw, y_train)
    else:
        logger.info("Augmenting training SMILES (n_smiles=%d) …", cfg.aug_n_smiles)
        X_train_aug, y_train_aug = augment_smiles(
            X_train_raw, y_train, n_smiles=cfg.aug_n_smiles, seed=cfg.seed
        )
    X_test_can,  y_test_can  = canonicalize_smiles(X_test_raw,  y_test)
    X_test2_can, y_test2_can = canonicalize_smiles(X_test2_raw, y_test2)

    logger.info(
        "After augmentation — train: %d  test: %d  test2: %d",
        len(X_train_aug), len(X_test_can), len(X_test2_can),
    )

    # 4. Featurize
    # Mordred: fit StandardScaler on all splits combined (original behaviour)
    logger.info("Featurizing with '%s' …", cfg.feature_type)
    if cfg.feature_type == "mordred":
        (X_train, y_train_np), (X_test, y_test_np), (X_test2, y_test2_np) = featurize_splits(
            [(X_train_aug, y_train_aug), (X_test_can, y_test_can), (X_test2_can, y_test2_can)],
            "mordred",
        )
    else:
        X_train, y_train_np = featurize(
            X_train_aug, y_train_aug, cfg.feature_type,
            morgan_radius=cfg.morgan_radius, morgan_nbits=cfg.morgan_nbits,
        )
        X_test,  y_test_np  = featurize(
            X_test_can,  y_test_can,  cfg.feature_type,
            morgan_radius=cfg.morgan_radius, morgan_nbits=cfg.morgan_nbits,
        )
        X_test2, y_test2_np = featurize(
            X_test2_can, y_test2_can, cfg.feature_type,
            morgan_radius=cfg.morgan_radius, morgan_nbits=cfg.morgan_nbits,
        )

    logger.info(
        "Feature matrix shapes — train: %s  test: %s  test2: %s",
        X_train.shape, X_test.shape, X_test2.shape,
    )

    # 4. Cross-validation
    logger.info("Starting %d-fold cross-validation …", cfg.n_folds)
    best_model, test_loader, test_loader2 = run_kfold(
        X_train, y_train_np,
        X_test,  y_test_np,
        X_test2, y_test2_np,
        cfg, cfg.output_dir,
    )

    # 5. Save best model
    model_path = os.path.join(cfg.output_dir, "model", "model.pth")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(best_model.state_dict(), model_path)
    logger.info("Model saved to %s", model_path)

    # 6. Final evaluation
    from src.training.loops import test_epoch
    device = next(best_model.parameters()).device
    te1 = test_epoch(best_model, test_loader,  cfg.l1_lambda, device)
    te2 = test_epoch(best_model, test_loader2, cfg.l1_lambda, device)

    metrics1 = compute_metrics(te1.y_true, te1.y_pred)
    metrics2 = compute_metrics(te2.y_true, te2.y_pred)

    table_dir = os.path.join(cfg.output_dir, "table")
    save_to_excel(metrics1, os.path.join(table_dir, "Result1.xlsx"))
    save_to_excel(metrics2, os.path.join(table_dir, "Result2.xlsx"))

    logger.info("Test 1  F1=%.4f  MCC=%.4f", metrics1["F1 Score"], metrics1["MCC"])
    logger.info("Test 2  F1=%.4f  MCC=%.4f", metrics2["F1 Score"], metrics2["MCC"])
    logger.info("Done.")


if __name__ == "__main__":
    main()
