import torch.nn as nn

from data_prepare import HIDDEN, DROPOUT


class TitanicNet(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, HIDDEN),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN, 1),
        )
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, x):
        return self.net(x).squeeze(-1)