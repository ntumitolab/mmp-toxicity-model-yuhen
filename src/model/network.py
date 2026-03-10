import torch
import torch.nn as nn


class FCModel(nn.Module):
    """
    Fully-connected classifier: input → 1024 → 512 → 256 → 128 → 64 → 32 → 1 (sigmoid).

    Architecture mirrors the original yuhen-model but removes the dead
    ``self.softmax`` attribute that was never used in ``forward()``.
    """

    def __init__(self, input_size: int, output_size: int = 1) -> None:
        super().__init__()
        self.fc1  = nn.Linear(input_size, 1024)
        self.bn1  = nn.BatchNorm1d(1024)
        self.fc2  = nn.Linear(1024, 512)
        self.bn2  = nn.BatchNorm1d(512)
        self.fc3  = nn.Linear(512, 256)
        self.bn3  = nn.BatchNorm1d(256)
        self.fc4  = nn.Linear(256, 128)
        self.bn4  = nn.BatchNorm1d(128)
        self.fc5  = nn.Linear(128, 64)
        self.bn5  = nn.BatchNorm1d(64)
        self.fc6  = nn.Linear(64, 32)
        self.bn6  = nn.BatchNorm1d(32)
        self.fc7  = nn.Linear(32, output_size)
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for fc, bn in [
            (self.fc1, self.bn1),
            (self.fc2, self.bn2),
            (self.fc3, self.bn3),
            (self.fc4, self.bn4),
            (self.fc5, self.bn5),
            (self.fc6, self.bn6),
        ]:
            x = self.dropout(self.relu(bn(fc(x))))
        return torch.sigmoid(self.fc7(x))

    def __repr__(self) -> str:
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return (
            f"FCModel("
            f"input={self.fc1.in_features}, "
            f"layers=1024→512→256→128→64→32→1, "
            f"params={n_params:,})"
        )
