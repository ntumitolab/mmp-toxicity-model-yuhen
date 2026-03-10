import logging
import random
from typing import List, Tuple

from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

logger = logging.getLogger(__name__)


def augment_smiles(
    smiles_list: List[str],
    y_list: List[int],
    n_smiles: int = 50,
    seed: int = 42,
) -> Tuple[List[str], List[int]]:
    """
    Augment a SMILES training set via random SMILES enumeration.

    For each valid molecule, up to *n_smiles* unique random SMILES are
    generated and paired with the original label.  Invalid entries are
    logged and skipped.

    Parameters
    ----------
    smiles_list:
        Raw SMILES strings (training split only).
    y_list:
        Corresponding integer labels.
    n_smiles:
        Target number of random SMILES per molecule (after deduplication
        the actual count may be lower).
    seed:
        Random seed passed to Python's ``random`` module for
        reproducibility.

    Returns
    -------
    (X_aug, y_aug)
        Augmented lists of SMILES strings and labels.
    """
    random.seed(seed)
    X_aug: List[str] = []
    y_aug: List[int] = []
    n_skipped = 0

    for smiles, label in zip(smiles_list, y_list):
        if not isinstance(smiles, str):
            logger.warning("Skipping non-string entry: %r (type %s)", smiles, type(smiles))
            n_skipped += 1
            continue

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning("Invalid SMILES (could not parse): %s", smiles)
            n_skipped += 1
            continue

        variants = {Chem.MolToSmiles(mol, doRandom=True) for _ in range(n_smiles)}
        X_aug.extend(variants)
        y_aug.extend([label] * len(variants))

    if n_skipped:
        logger.warning("augment_smiles: skipped %d invalid entries.", n_skipped)

    return X_aug, y_aug


def inchi_to_smiles(
    inchi_list: List[str],
    y_list: List[int],
) -> Tuple[List[str], List[int]]:
    """
    Convert InChI strings to canonical SMILES, dropping invalid entries.

    Used before augmentation or featurization when the input column is InChI.
    """
    X_out: List[str] = []
    y_out: List[int] = []
    n_skipped = 0

    for inchi, label in zip(inchi_list, y_list):
        if not isinstance(inchi, str):
            logger.warning("Skipping non-string InChI: %r", inchi)
            n_skipped += 1
            continue
        mol = Chem.MolFromInchi(inchi)
        if mol is None:
            logger.warning("Invalid InChI (could not parse): %s", inchi)
            n_skipped += 1
            continue
        X_out.append(Chem.MolToSmiles(mol))
        y_out.append(label)

    if n_skipped:
        logger.warning("inchi_to_smiles: skipped %d invalid entries.", n_skipped)

    return X_out, y_out


def augment_stereo_smiles(
    smiles_list: List[str],
    y_list: List[int],
) -> Tuple[List[str], List[int]]:
    """
    Augment training data with stereo-isomers for the TOXIC class only.

    Non-toxic molecules are kept as-is (canonical SMILES).
    Matches the original model_mordred augmentation strategy.

    Parameters
    ----------
    smiles_list:
        Raw SMILES strings (training split only).
    y_list:
        Corresponding integer labels.

    Returns
    -------
    (X_aug, y_aug)
        Augmented lists of SMILES strings and labels.
    """
    from rdkit.Chem.EnumerateStereoisomers import (
        EnumerateStereoisomers,
        StereoEnumerationOptions,
    )

    X_aug: List[str] = []
    y_aug: List[int] = []
    n_skipped = 0

    for smiles, label in zip(smiles_list, y_list):
        if not isinstance(smiles, str):
            logger.warning("Skipping non-string entry: %r", smiles)
            n_skipped += 1
            continue
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning("Invalid SMILES skipped: %s", smiles)
            n_skipped += 1
            continue

        if label == 1:
            opts = StereoEnumerationOptions(unique=True)
            isomers = list(EnumerateStereoisomers(mol, options=opts))
            isomer_smiles = list({Chem.MolToSmiles(iso) for iso in isomers})
            X_aug.extend(isomer_smiles)
            y_aug.extend([label] * len(isomer_smiles))
        else:
            X_aug.append(Chem.MolToSmiles(mol))
            y_aug.append(label)

    if n_skipped:
        logger.warning("augment_stereo_smiles: skipped %d invalid entries.", n_skipped)

    return X_aug, y_aug


def canonicalize_smiles(
    smiles_list: List[str],
    y_list: List[int],
) -> Tuple[List[str], List[int]]:
    """
    Convert raw SMILES to canonical form, dropping invalid entries.

    Used for test splits (no augmentation needed, just standardisation).
    """
    X_out: List[str] = []
    y_out: List[int] = []
    n_skipped = 0

    for smiles, label in zip(smiles_list, y_list):
        if not isinstance(smiles, str):
            logger.warning("Skipping non-string entry: %r", smiles)
            n_skipped += 1
            continue
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning("Invalid SMILES: %s", smiles)
            n_skipped += 1
            continue
        X_out.append(Chem.MolToSmiles(mol))
        y_out.append(label)

    if n_skipped:
        logger.warning("canonicalize_smiles: skipped %d invalid entries.", n_skipped)

    return X_out, y_out
