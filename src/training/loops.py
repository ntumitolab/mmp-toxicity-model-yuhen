"""
Per-epoch training, validation and test loops.

All three share the same L1-regularised BCE loss helper; each returns
an ``EpochResult`` dataclass so callers never unpack raw tuples.
"""
from dataclasses import dataclass, field
from typing import List

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, matthews_corrcoef, confusion_matrix

from src.model.network import FCModel


@dataclass
class EpochResult:
    loss: float
    f1: float
    mcc: float
    confusion: np.ndarray      # shape (2, 2)
    y_true: List[int]
    y_pred: List[int]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _bce_loss() -> nn.BCELoss:
    return nn.BCELoss()


def _l1_norm(model: FCModel) -> torch.Tensor:
    return sum(p.abs().sum() for p in model.parameters())


def _to_predictions(outputs: torch.Tensor) -> np.ndarray:
    return torch.round(outputs).squeeze().cpu().detach().numpy().flatten()


# ---------------------------------------------------------------------------
# Public loops
# ---------------------------------------------------------------------------

def train_epoch(
    model: FCModel,
    optimizer: torch.optim.Optimizer,
    loader: torch.utils.data.DataLoader,
    l1_lambda: float,
    device: torch.device,
) -> EpochResult:
    model.train()
    criterion = _bce_loss()
    running_loss = 0.0
    y_true_all: List[int] = []
    y_pred_all: List[int] = []

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.float()) + l1_lambda * _l1_norm(model)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = _to_predictions(outputs)
        y_true_all.extend(labels.cpu().numpy().flatten().tolist())
        y_pred_all.extend(preds.tolist())

    loss_avg = running_loss / len(loader)
    f1  = f1_score(y_true_all, y_pred_all, zero_division=0)
    mcc = matthews_corrcoef(y_true_all, y_pred_all)
    cm  = confusion_matrix(y_true_all, y_pred_all)
    return EpochResult(loss=loss_avg, f1=f1, mcc=mcc, confusion=cm,
                       y_true=y_true_all, y_pred=y_pred_all)


def validate_epoch(
    model: FCModel,
    loader: torch.utils.data.DataLoader,
    l1_lambda: float,
    device: torch.device,
) -> EpochResult:
    model.eval()
    criterion = _bce_loss()
    running_loss = 0.0
    y_true_all: List[int] = []
    y_pred_all: List[int] = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(inputs)
            loss = criterion(outputs, labels.float()) + l1_lambda * _l1_norm(model)
            running_loss += loss.item()

            preds = _to_predictions(outputs)
            y_true_all.extend(labels.cpu().numpy().flatten().tolist())
            y_pred_all.extend(preds.tolist())

    loss_avg = running_loss / len(loader)
    f1  = f1_score(y_true_all, y_pred_all, zero_division=0)
    mcc = matthews_corrcoef(y_true_all, y_pred_all)
    cm  = confusion_matrix(y_true_all, y_pred_all)
    return EpochResult(loss=loss_avg, f1=f1, mcc=mcc, confusion=cm,
                       y_true=y_true_all, y_pred=y_pred_all)


def test_epoch(
    model: FCModel,
    loader: torch.utils.data.DataLoader,
    l1_lambda: float,
    device: torch.device,
) -> EpochResult:
    model.eval()
    criterion = _bce_loss()
    running_loss = 0.0
    y_true_all: List[int] = []
    y_pred_all: List[int] = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(inputs)
            loss = criterion(outputs, labels.float()) + l1_lambda * _l1_norm(model)
            running_loss += loss.item()

            preds = _to_predictions(outputs)
            y_true_all.extend(labels.cpu().numpy().flatten().tolist())
            y_pred_all.extend(preds.tolist())

    loss_avg = running_loss / len(loader)
    f1  = f1_score(y_true_all, y_pred_all, zero_division=0)
    mcc = matthews_corrcoef(y_true_all, y_pred_all)
    cm  = confusion_matrix(y_true_all, y_pred_all)
    return EpochResult(loss=loss_avg, f1=f1, mcc=mcc, confusion=cm,
                       y_true=y_true_all, y_pred=y_pred_all)
