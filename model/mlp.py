import torch
import torch.nn as nn

import torch
import torch.nn as nn


class ResidualMLPBlock(nn.Module):

    def __init__(self, in_dim: int, hidden_dim: int, dropout: float):
        super().__init__()

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)

        self.fc2 = nn.Linear(hidden_dim, in_dim)
        self.bn2 = nn.BatchNorm1d(in_dim)

        self.proj = None
        # here output is in_dim, so skip dims match (no projection)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(x)
        h = self.bn1(h)
        h = self.act(h)
        h = self.drop(h)

        h = self.fc2(h)
        h = self.bn2(h)

        out = x + h
        out = self.act(out)
        return out


class MLPRegressor(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dims,
        dropout,):
        super().__init__()

        d1, d2, d3, d4 = hidden_dims

        layers = []

        layers += [
            nn.Linear(input_dim, d1),
            nn.BatchNorm1d(d1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        ]

        layers += [
            nn.Linear(d1, d2),
            nn.BatchNorm1d(d2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(d2, d3),
            nn.BatchNorm1d(d3),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(d3, d4),
            nn.BatchNorm1d(d4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        ]

        self.backbone = nn.Sequential(*layers)
        self.residual = ResidualMLPBlock(d4, hidden_dim=max(64, d4), dropout=dropout) 

        # Head
        self.head = nn.Linear(d4, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x) :
        x = self.backbone(x)
        x = self.residual(x)
        out = self.head(x).squeeze(-1)  # [B]
        return out
