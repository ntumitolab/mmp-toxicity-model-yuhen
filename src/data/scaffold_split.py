"""Bemis-Murcko scaffold split utilities."""
import logging
from collections import defaultdict
from typing import List, Tuple

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

logger = logging.getLogger(__name__)


def compute_scaffold(smi: str) -> str:
    """Return canonical Bemis-Murcko scaffold SMILES for *smi*.

    Molecules with no ring system (acyclic) return an empty string and are
    grouped together as a single "no-scaffold" cluster.
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return ""
    try:
        core = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(core)
    except Exception:
        return ""


def scaffold_split(
    smiles: List[str],
    labels: List[int],
    test_frac: float = 0.12,
    seed: int = 42,  # kept for API compatibility; not used (split is deterministic)
) -> Tuple[List[str], List[int], List[str], List[int]]:
    """Split *smiles*/*labels* into train and test by Bemis-Murcko scaffold.

    Molecules sharing the same scaffold are always assigned to the same
    partition, preventing scaffold leakage between train and test.

    Assignment strategy (matches DeepChem / MoleculeNet convention):
    scaffold groups are sorted largest → smallest, then assigned to **train**
    until the train quota (``1 - test_frac``) is filled; the remaining
    smaller / rarer scaffold groups go to test.  This keeps large, common
    scaffold families in training and reserves novel scaffolds for evaluation.
    The split is fully deterministic (no randomness).

    Parameters
    ----------
    smiles:
        List of canonical SMILES strings.
    labels:
        Corresponding integer class labels.
    test_frac:
        Target fraction of molecules to place in the test set.
        Default 0.12 matches the original Train/Test file ratio.
    seed:
        Ignored — kept for API compatibility only.

    Returns
    -------
    X_train, y_train, X_test, y_test
    """
    assert len(smiles) == len(labels), "smiles and labels must have equal length"

    # Group molecule indices by scaffold
    scaffold_to_indices: dict = defaultdict(list)
    for i, smi in enumerate(smiles):
        sc = compute_scaffold(smi)
        scaffold_to_indices[sc].append(i)

    n_no_scaffold = len(scaffold_to_indices.get("", []))
    logger.info(
        "Scaffold split: %d unique scaffolds (%d acyclic molecules grouped together)",
        len(scaffold_to_indices), n_no_scaffold,
    )

    # Sort groups largest → smallest; fill train first, remainder → test.
    # Large scaffold families (acyclic, benzene, …) stay in train;
    # rarer / novel scaffolds form the test set.
    groups = sorted(scaffold_to_indices.values(), key=len, reverse=True)

    n_train_target = int(len(smiles) * (1 - test_frac))
    train_idx: List[int] = []
    test_idx: List[int] = []

    for group in groups:
        if len(train_idx) < n_train_target:
            train_idx.extend(group)
        else:
            test_idx.extend(group)

    logger.info(
        "Scaffold split result — train: %d (%.1f%%)  test: %d (%.1f%%)",
        len(train_idx), 100 * len(train_idx) / len(smiles),
        len(test_idx),  100 * len(test_idx)  / len(smiles),
    )

    X_train = [smiles[i] for i in train_idx]
    y_train = [labels[i] for i in train_idx]
    X_test  = [smiles[i] for i in test_idx]
    y_test  = [labels[i] for i in test_idx]

    return X_train, y_train, X_test, y_test
