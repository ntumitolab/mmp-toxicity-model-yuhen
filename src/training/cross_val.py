"""
K-fold cross-validation orchestration.

Bug fixes vs. original:
  - Epoch histories are accumulated as *lists* (one value per epoch), enabling
    real loss/F1 curves instead of single-scalar plots.
  - Best model is selected by *lowest validation loss* (not reversed F1).
"""
import logging
import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from src.config import ExperimentConfig
from src.model.network import FCModel
from src.training.loops import train_epoch, validate_epoch, test_epoch

logger = logging.getLogger(__name__)


def _make_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
    drop_last: bool = True,
) -> torch.utils.data.DataLoader:
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(X).float(),
        torch.from_numpy(y).float(),
    )
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last
    )


def _save_loss_curve(
    train_losses: list,
    val_losses: list,
    fold: int,
    plot_dir: str,
) -> None:
    epochs = range(1, len(train_losses) + 1)
    plt.figure()
    plt.plot(epochs, train_losses, label="Training Loss")
    plt.plot(epochs, val_losses,   label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"Fold {fold} — Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"Fold{fold}-Loss.png"))
    plt.close()


def _save_metric_curve(
    train_f1: list,
    val_f1: list,
    train_mcc: list,
    val_mcc: list,
    fold: int,
    plot_dir: str,
) -> None:
    epochs = range(1, len(train_f1) + 1)
    plt.figure()
    plt.plot(epochs, train_f1,  label="Train F1",  linestyle="-")
    plt.plot(epochs, val_f1,    label="Val F1",    linestyle="--")
    plt.plot(epochs, train_mcc, label="Train MCC", linestyle="-",  alpha=0.6)
    plt.plot(epochs, val_mcc,   label="Val MCC",   linestyle="--", alpha=0.6)
    plt.xlabel("Epochs")
    plt.ylabel("Score")
    plt.title(f"Fold {fold} — F1 and MCC per epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"Fold{fold}-Metrics.png"))
    plt.close()


def run_kfold(
    X: np.ndarray,
    y: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_test2: np.ndarray,
    y_test2: np.ndarray,
    cfg: ExperimentConfig,
    output_dir: str,
) -> Tuple[FCModel, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Run stratified k-fold cross-validation and save results.

    Returns
    -------
    (best_model, test_loader, test_loader2)
        The fold model with the lowest validation loss, plus test loaders.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    plot_dir  = os.path.join(output_dir, "plots")
    table_dir = os.path.join(output_dir, "table")
    model_dir = os.path.join(output_dir, "model")
    for d in (plot_dir, table_dir, model_dir):
        os.makedirs(d, exist_ok=True)

    kfold = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)

    # Test loaders are the same for every fold
    test_loader  = _make_loader(X_test,  y_test,  batch_size=len(X_test),  shuffle=False, drop_last=False)
    test_loader2 = _make_loader(X_test2, y_test2, batch_size=len(X_test2), shuffle=False, drop_last=False)

    metrics_rows = []
    fold_val_losses = []   # best val loss per fold, for final model selection
    fold_models     = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y), start=1):
        logger.info("=" * 60)
        logger.info("Fold %d / %d", fold, cfg.n_folds)

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        train_loader = _make_loader(X_train, y_train, cfg.batch_size, shuffle=True,  drop_last=True)
        val_loader   = _make_loader(X_val,   y_val,   cfg.batch_size, shuffle=False, drop_last=True)

        model = FCModel(input_size=X_train.shape[1]).to(device).to(torch.float)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=cfg.lr_patience
        )

        # Per-epoch history lists (fix: was scalar in original)
        train_loss_hist, val_loss_hist = [], []
        train_f1_hist,   val_f1_hist   = [], []
        train_mcc_hist,  val_mcc_hist  = [], []

        best_val_loss = float("inf")
        best_model_state = None
        patience_counter = 0

        for epoch in tqdm(range(cfg.n_epochs), desc=f"Fold {fold}", unit="ep"):
            tr = train_epoch(model, optimizer, train_loader, cfg.l1_lambda, device)
            vl = validate_epoch(model, val_loader, cfg.l1_lambda, device)
            scheduler.step(vl.loss)

            # Accumulate histories
            train_loss_hist.append(tr.loss)
            val_loss_hist.append(vl.loss)
            train_f1_hist.append(tr.f1)
            val_f1_hist.append(vl.f1)
            train_mcc_hist.append(tr.mcc)
            val_mcc_hist.append(vl.mcc)

            logger.debug(
                "Epoch %d — train_loss=%.4f  val_loss=%.4f  val_f1=%.4f  val_mcc=%.4f",
                epoch + 1, tr.loss, vl.loss, vl.f1, vl.mcc,
            )

            # Model selection: lowest validation loss (fix: original used F1 with `<`)
            if vl.loss < best_val_loss:
                best_val_loss = vl.loss
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= cfg.early_stop_patience:
                    logger.info("Early stopping at epoch %d.", epoch + 1)
                    break

        # Restore best weights for this fold
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
        model.eval()

        # Evaluate on both test sets
        te1 = test_epoch(model, test_loader,  cfg.l1_lambda, device)
        te2 = test_epoch(model, test_loader2, cfg.l1_lambda, device)

        logger.info(
            "Fold %d  test1: loss=%.4f f1=%.4f mcc=%.4f | test2: f1=%.4f mcc=%.4f",
            fold, te1.loss, te1.f1, te1.mcc, te2.f1, te2.mcc,
        )

        # Save plots for this fold
        _save_loss_curve(train_loss_hist, val_loss_hist, fold, plot_dir)
        _save_metric_curve(train_f1_hist, val_f1_hist, train_mcc_hist, val_mcc_hist, fold, plot_dir)

        # Record best-epoch scalar metrics for this fold
        best_ep = int(np.argmin(val_loss_hist))
        metrics_rows.append({
            "Fold":            fold,
            "Training Loss":   train_loss_hist[best_ep],
            "Validation Loss": val_loss_hist[best_ep],
            "Testing Loss":    te1.loss,
            "Testing 2 Loss":  te2.loss,
            "Training F1":     train_f1_hist[best_ep],
            "Validation F1":   val_f1_hist[best_ep],
            "Testing F1":      te1.f1,
            "Testing 2 F1":    te2.f1,
            "Training MCC":    train_mcc_hist[best_ep],
            "Validation MCC":  val_mcc_hist[best_ep],
            "Testing MCC":     te1.mcc,
            "Testing 2 MCC":   te2.mcc,
            "Training CM":     str(te1.confusion.tolist()),   # store as string for Excel
            "Validation CM":   str(te2.confusion.tolist()),
            "Testing CM":      str(te1.confusion.tolist()),
            "Testing 2 CM":    str(te2.confusion.tolist()),
        })

        fold_val_losses.append(best_val_loss)
        fold_models.append({k: v.cpu() for k, v in model.state_dict().items()})

    # Save cross-validation table
    results_df = pd.DataFrame(metrics_rows)
    results_df.to_excel(os.path.join(table_dir, "cross_validation.xlsx"), index=False)
    logger.info("Saved cross_validation.xlsx")

    # Select the fold with the lowest best validation loss
    best_fold_idx = int(np.argmin(fold_val_losses))
    logger.info(
        "Best fold: %d (val_loss=%.4f)",
        best_fold_idx + 1, fold_val_losses[best_fold_idx],
    )

    # Reconstruct model from best fold weights
    best_model = FCModel(input_size=X.shape[1]).to(device).to(torch.float)
    best_model.load_state_dict(
        {k: v.to(device) for k, v in fold_models[best_fold_idx].items()}
    )

    return best_model, test_loader, test_loader2
