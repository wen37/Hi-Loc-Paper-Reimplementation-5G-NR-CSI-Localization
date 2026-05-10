import torch
import torch.nn as nn
from .cnn_block import CNNBlock
from .bilstm_block import BiLSTMBlock
from .feature_attention import FeatureAttention

class BaselineModel(nn.Module):
    def __init__(
        self,
        feature_dim,
        cnn_out_channels,
        lstm_hidden_size,
        output_dim=2,
        lstm_num_layers=2,
        dropout_rate=0.3,
        fc_hidden_dim=128,
        use_feature_attention=False,
        attention_reduction=8,
    ):
        super(BaselineModel, self).__init__()

        self.input_norm = nn.LayerNorm(feature_dim)
        self.input_dropout = nn.Dropout(dropout_rate)
        self.cnn = CNNBlock(in_channels=feature_dim, out_channels=cnn_out_channels)
        self.use_feature_attention = use_feature_attention
        if use_feature_attention:
            self.feature_attention = FeatureAttention(
                feature_dim=cnn_out_channels,
                reduction=attention_reduction,
            )

        self.lstm = BiLSTMBlock(
            input_size=cnn_out_channels,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            dropout=dropout_rate,
        )

        self.temporal_dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_size * 2, fc_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fc_hidden_dim, output_dim)
        )
        
    def forward(self, x):
        # x: (B, T, F)

        # Normalize each time step before spatial-temporal modeling.
        x = self.input_norm(x)
        x = self.input_dropout(x)

        # 1. CNN (B, F, T)
        x = x.transpose(1, 2) # (B, F, T)
        x = self.cnn(x) # (B, C, T)
        
        # 2. BiLSTM (B, T, C)
        x = x.transpose(1, 2) # (B, T, C)
        if self.use_feature_attention:
            x, _ = self.feature_attention(x)
        x = self.lstm(x) # (B, T, 2*H)
        x = self.temporal_dropout(x)
        
        # 3. Take the last time step for regression
        x = x[:, -1, :] # (B, 2*H)
        
        # 4. FC
        out = self.fc(x) # (B, output_dim)
        
        return out
