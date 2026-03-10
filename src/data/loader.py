import logging
import os
from typing import Dict, List, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

_ALT = {"Smile": "Inchi", "Inchi": "Smile"}


def _detect_col(df: pd.DataFrame, preferred: str, filename: str) -> str:
    """Return *preferred* if present, otherwise fall back to the alternate column."""
    if preferred in df.columns:
        return preferred
    alt = _ALT.get(preferred)
    if alt and alt in df.columns:
        logger.warning(
            "%s: column '%s' not found; using '%s' instead",
            filename, preferred, alt,
        )
        return alt
    raise ValueError(
        f"Column '{preferred}' not found in {filename}. "
        f"Available columns: {list(df.columns)}"
    )


def load_combined(
    data_dir: str,
    label_col: str = "Toxicity",
) -> Tuple[List[str], List[int], List[str], List[int], List[str], List[int], Dict[str, str]]:
    """Load Train_inchi.xlsx + Test.xlsx (combined pool) and Test2.xlsx (external holdout).

    Both Train_inchi.xlsx and Test.xlsx are loaded for scaffold splitting.
    Test2.xlsx is returned separately as the fixed external holdout.

    Returns
    -------
    X_train_raw, y_train, X_test_raw, y_test, X_test2_raw, y_test2, detected_cols
        Raw molecule strings (InChI or SMILES) and integer labels.
        ``detected_cols`` maps split name to the column actually read from each file.

    Raises
    ------
    FileNotFoundError
        If any required Excel file is absent.
    ValueError
        If any DataFrame is empty or required columns are missing.
    """
    files = {
        "Train": os.path.join(data_dir, "Train_inchi.xlsx"),
        "Test":  os.path.join(data_dir, "Test.xlsx"),
        "Test2": os.path.join(data_dir, "Test2.xlsx"),
    }
    for name, path in files.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} file not found: {path}")

    dfs = {name: pd.read_excel(path, index_col=0) for name, path in files.items()}

    detected_cols: Dict[str, str] = {}
    for name, df in dfs.items():
        if df.empty:
            raise ValueError(f"{name} DataFrame is empty.")
        detected_cols[name] = _detect_col(df, "Inchi", f"{name}.xlsx")
        if label_col not in df.columns:
            raise ValueError(
                f"Column '{label_col}' not found in {name}. "
                f"Available columns: {list(df.columns)}"
            )

    def _extract(df: pd.DataFrame, col: str) -> Tuple[List[str], List[int]]:
        X = df[col].tolist()
        y = df[label_col].astype(int).tolist()
        return X, y

    X_train_raw, y_train = _extract(dfs["Train"], detected_cols["Train"])
    X_test_raw,  y_test  = _extract(dfs["Test"],  detected_cols["Test"])
    X_test2_raw, y_test2 = _extract(dfs["Test2"], detected_cols["Test2"])

    return X_train_raw, y_train, X_test_raw, y_test, X_test2_raw, y_test2, detected_cols


def load_splits(
    data_dir: str,
    input_col: str = "Smile",
    label_col: str = "Toxicity",
) -> Tuple[List[str], List[int], List[str], List[int], List[str], List[int], Dict[str, str]]:
    """
    Load Train_smile.xlsx or Train_inchi.xlsx (based on *input_col*) plus
    Test/Test2.xlsx from *data_dir* and return six lists plus a dict of the
    column name actually used in each split.

    The column is detected per-file: if the preferred column is absent the
    alternate (Smile ↔ Inchi) is used automatically with a warning.

    Returns
    -------
    X_train, y_train, X_test, y_test, X_test2, y_test2
        Raw string identifiers (SMILES or InChI) and integer labels.
    detected_cols : dict
        ``{"Train": "Smile", "Test": "Inchi", ...}`` — the column name
        actually read from each file.

    Raises
    ------
    FileNotFoundError
        If any of the three required Excel files is absent.
    ValueError
        If any DataFrame is empty or required columns are missing.
    """
    train_file = "Train_smile.xlsx" if input_col == "Smile" else "Train_inchi.xlsx"
    test_input_col = "Inchi"

    files = {
        "Train": os.path.join(data_dir, train_file),
        "Test":  os.path.join(data_dir, "Test.xlsx"),
        "Test2": os.path.join(data_dir, "Test2.xlsx"),
    }
    for name, path in files.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} file not found: {path}")

    dfs = {name: pd.read_excel(path, index_col=0) for name, path in files.items()}

    detected_cols: Dict[str, str] = {}
    for name, df in dfs.items():
        if df.empty:
            raise ValueError(f"{name} DataFrame is empty.")
        
        if "test" in name.lower():
            detected_cols[name] = _detect_col(df, test_input_col, f"{name}.xlsx") 
        else:
            detected_cols[name] = _detect_col(df, input_col, f"{name}.xlsx")
        if label_col not in df.columns:
            raise ValueError(
                f"Column '{label_col}' not found in {name}.xlsx. "
                f"Available columns: {list(df.columns)}"
            )

    def _extract(df: pd.DataFrame, col: str) -> Tuple[List[str], List[int]]:
        X = df[col].tolist()
        y = df[label_col].astype(int).tolist()
        return X, y

    X_train, y_train = _extract(dfs["Train"], detected_cols["Train"])
    X_test,  y_test  = _extract(dfs["Test"],  detected_cols["Test"])
    X_test2, y_test2 = _extract(dfs["Test2"], detected_cols["Test2"])

    return X_train, y_train, X_test, y_test, X_test2, y_test2, detected_cols
