"""
Featurization: SMILES strings → float32 numpy arrays.

Dispatch via ``featurize(smiles_list, y_list, feature_type)``.
For Mordred, use ``featurize_splits`` to fit the StandardScaler on all
splits combined (matching the original implementation).
"""
import logging
from typing import List, Tuple

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, MACCSkeys
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _morgan_fp(smiles_list: List[str], y_list: List[int],
               radius: int = 2, nbits: int = 2048) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for smi, label in zip(smiles_list, y_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
        arr = np.zeros((nbits,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        X.append(arr)
        y.append(label)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def _maccs_fp(smiles_list: List[str], y_list: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for smi, label in zip(smiles_list, y_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = MACCSkeys.GenMACCSKeys(mol)
        arr = np.zeros((167,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        X.append(arr)
        y.append(label)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def _combined_fp(smiles_list: List[str], y_list: List[int],
                 radius: int = 2, nbits: int = 2048) -> Tuple[np.ndarray, np.ndarray]:
    """Morgan (2048) concatenated with MACCS (167) → 2215-dim vector."""
    X, y = [], []
    for smi, label in zip(smiles_list, y_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp_m = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
        arr_m = np.zeros((nbits,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp_m, arr_m)
        fp_k = MACCSkeys.GenMACCSKeys(mol)
        arr_k = np.zeros((167,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp_k, arr_k)
        X.append(np.concatenate([arr_m, arr_k]))
        y.append(label)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def _mordred_fp(smiles_list: List[str], y_list: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    """All Mordred descriptors → keep numeric columns → StandardScaler.

    Matches original: ignore_3D=False, select_dtypes(include=[np.number]).
    Non-numeric columns (failed descriptors, e.g. 3D on 2D molecules) are
    dropped.  StandardScaler is fit on the provided molecules only.

    For multi-split featurization with a shared scaler (train+test+test2
    normalised together, as in the original), use featurize_splits().
    """
    import pandas as pd
    from sklearn.preprocessing import StandardScaler

    try:
        from mordred import Calculator, descriptors as mordred_descriptors
    except ImportError as e:
        raise ImportError(
            "mordred is required for feature_type='mordred'. "
            "Install it with: pip install mordred"
        ) from e

    calc = Calculator(mordred_descriptors, ignore_3D=False)

    mols, valid_y = [], []
    for smi, label in zip(smiles_list, y_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            logger.warning("Invalid SMILES skipped in mordred featurizer: %s", smi)
            continue
        mols.append(mol)
        valid_y.append(label)

    if not mols:
        raise ValueError("No valid molecules for Mordred featurization.")

    df = calc.pandas(mols)
    # Keep only numeric columns; non-numeric dtype indicates descriptor failure
    # (3D descriptors on 2D molecules return error objects → object dtype)
    df = df.select_dtypes(include=[np.number])

    X = df.values.astype(np.float32)
    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)
    return X, np.array(valid_y, dtype=np.int64)


def _mordred_fp_splits(
    splits: List[Tuple[List[str], List[int]]],
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Mordred featurization for multiple splits with a shared StandardScaler.

    Fits the scaler on ALL molecules combined (matching the original
    implementation where train, test and test2 were normalised together).
    Returns a list of (X, y) tuples in the same order as *splits*.
    """
    import pandas as pd
    from sklearn.preprocessing import StandardScaler

    try:
        from mordred import Calculator, descriptors as mordred_descriptors
    except ImportError as e:
        raise ImportError(
            "mordred is required for feature_type='mordred'. "
            "Install it with: pip install mordred"
        ) from e

    calc = Calculator(mordred_descriptors, ignore_3D=False)

    all_mols: List = []
    all_y: List[int] = []
    split_row_indices: List[List[int]] = []

    for smiles_list, y_list in splits:
        rows: List[int] = []
        for smi, label in zip(smiles_list, y_list):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                logger.warning("Invalid SMILES skipped in mordred featurizer: %s", smi)
                continue
            rows.append(len(all_mols))
            all_mols.append(mol)
            all_y.append(label)
        split_row_indices.append(rows)

    if not all_mols:
        raise ValueError("No valid molecules for Mordred featurization.")

    df = calc.pandas(all_mols)
    df = df.select_dtypes(include=[np.number])

    X_all = df.values.astype(np.float32)
    scaler = StandardScaler()
    X_all = scaler.fit_transform(X_all).astype(np.float32)
    y_all = np.array(all_y, dtype=np.int64)

    return [
        (X_all[rows], y_all[rows])
        for rows in split_row_indices
    ]


def _pubchem_fp(smiles_list: List[str], y_list: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    """PubChem 881-bit fingerprints via DeepChem.

    Matches the original implementation which used deepchem.feat.PubChemFingerprint.
    Requires network access (PubChem REST API); computation is slow (~1-2 s/mol).
    """
    try:
        from deepchem.feat import PubChemFingerprint
    except ImportError as e:
        raise ImportError(
            "deepchem is required for feature_type='pubchem'. "
            "Install with: pip install deepchem"
        ) from e

    featurizer = PubChemFingerprint()
    X, y = [], []
    for smi, label in zip(smiles_list, y_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        canonical_smi = Chem.MolToSmiles(mol)
        try:
            features = featurizer(canonical_smi)[0]
            if features is not None and len(features) == 881:
                X.append(features)
                y.append(label)
        except Exception as exc:
            logger.warning("PubChem fingerprint failed for %s: %s", canonical_smi, exc)

    if not X:
        raise ValueError("No valid PubChem fingerprints computed.")

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def featurize(
    smiles_list: List[str],
    y_list: List[int],
    feature_type: str = "morgan",
    morgan_radius: int = 2,
    morgan_nbits: int = 2048,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert SMILES strings to a numeric feature matrix.

    Parameters
    ----------
    smiles_list:
        List of SMILES strings (already canonicalized / augmented).
    y_list:
        Corresponding integer labels.
    feature_type:
        One of ``"morgan"``, ``"macc"``, ``"combined"``, ``"mordred"``,
        ``"pubchem"``.
    morgan_radius, morgan_nbits:
        Parameters for Morgan fingerprints.

    Returns
    -------
    (X, y)
        ``X`` is float32 ndarray of shape (n_valid, n_features).
        ``y`` is int64 ndarray of shape (n_valid,).
        Only valid molecules are included.

    Notes
    -----
    For ``"mordred"``, the StandardScaler is fit on *smiles_list* only.
    To reproduce the original multi-split normalisation (scaler fit on
    train+test+test2 combined), use :func:`featurize_splits` instead.
    """
    feature_type = feature_type.lower()
    if feature_type == "morgan":
        return _morgan_fp(smiles_list, y_list, radius=morgan_radius, nbits=morgan_nbits)
    elif feature_type == "macc":
        return _maccs_fp(smiles_list, y_list)
    elif feature_type == "combined":
        return _combined_fp(smiles_list, y_list, radius=morgan_radius, nbits=morgan_nbits)
    elif feature_type == "mordred":
        return _mordred_fp(smiles_list, y_list)
    elif feature_type == "pubchem":
        return _pubchem_fp(smiles_list, y_list)
    else:
        raise ValueError(
            f"Unknown feature_type '{feature_type}'. "
            "Choose from: morgan, macc, combined, mordred, pubchem."
        )


def featurize_splits(
    splits: List[Tuple[List[str], List[int]]],
    feature_type: str,
    morgan_radius: int = 2,
    morgan_nbits: int = 2048,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Featurize multiple data splits, sharing preprocessing state where needed.

    For ``"mordred"``, the StandardScaler is fit on all molecules from all
    splits combined (reproducing the original implementation).
    For all other feature types, each split is featurized independently.

    Parameters
    ----------
    splits:
        List of ``(smiles_list, y_list)`` tuples — one per split.
    feature_type:
        Same choices as :func:`featurize`.

    Returns
    -------
    List of ``(X, y)`` tuples, one per input split.
    """
    feature_type = feature_type.lower()
    if feature_type == "mordred":
        return _mordred_fp_splits(splits)
    return [
        featurize(smi, y, feature_type, morgan_radius=morgan_radius, morgan_nbits=morgan_nbits)
        for smi, y in splits
    ]
