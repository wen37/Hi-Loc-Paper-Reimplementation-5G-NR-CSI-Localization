import torch
import torch.nn as nn


class FeatureAttention(nn.Module):
    def __init__(self, feature_dim, reduction=8):
        super().__init__()
        hidden_dim = max(1, feature_dim // reduction)
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (B, T, F)
        weights = self.attention(x)
        return x * weights, weights
