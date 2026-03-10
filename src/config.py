from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ExperimentConfig:
    feature_type: Literal["morgan", "macc", "combined", "mordred", "pubchem"] = "morgan"
    split_method: Literal["random", "scaffold"] = "random"
    scaffold_test_frac: float = 0.12   # matches original Train/Test file ratio
    aug_n_smiles: int = 50          # random SMILES per molecule for train augmentation
    morgan_radius: int = 2
    morgan_nbits: int = 2048
    n_folds: int = 5
    batch_size: int = 128
    n_epochs: int = 10000
    lr: float = 0.01
    weight_decay: float = 1e-5
    l1_lambda: float = 0.0005
    lr_patience: int = 3            # ReduceLROnPlateau patience
    early_stop_patience: int = 16
    seed: int = 42
    output_dir: str = "output"
    data_dir: str = "input"
