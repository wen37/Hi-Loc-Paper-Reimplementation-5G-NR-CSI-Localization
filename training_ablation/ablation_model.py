import torch
import torch.nn as nn

from models.bilstm_block import BiLSTMBlock
from models.cnn_block import CNNBlock


class ToggleBaselineModel(nn.Module):
    def __init__(
        self,
        feature_dim,
        cnn_out_channels,
        lstm_hidden_size,
        output_dim=2,
        lstm_num_layers=2,
        dropout_rate=0.3,
        fc_hidden_dim=128,
        use_input_norm=True,
        use_dropout=True,
    ):
        super().__init__()

        self.input_norm = nn.LayerNorm(feature_dim) if use_input_norm else nn.Identity()
        input_dropout_rate = dropout_rate if use_dropout else 0.0
        self.input_dropout = nn.Dropout(input_dropout_rate)
        self.cnn = CNNBlock(in_channels=feature_dim, out_channels=cnn_out_channels)
        self.lstm = BiLSTMBlock(
            input_size=cnn_out_channels,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            dropout=input_dropout_rate,
        )
        self.temporal_dropout = nn.Dropout(input_dropout_rate)
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_size * 2, fc_hidden_dim),
            nn.ReLU(),
            nn.Dropout(input_dropout_rate),
            nn.Linear(fc_hidden_dim, output_dim),
        )

    def forward(self, x):
        x = self.input_norm(x)
        x = self.input_dropout(x)
        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = x.transpose(1, 2)
        x = self.lstm(x)
        x = self.temporal_dropout(x)
        x = x[:, -1, :]
        return self.fc(x)
