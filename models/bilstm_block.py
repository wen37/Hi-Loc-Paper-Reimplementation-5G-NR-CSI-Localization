import torch
import torch.nn as nn

class BiLSTMBlock(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.0):
        super(BiLSTMBlock, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        
    def forward(self, x):
        # x: (B, T, input_size)
        out, _ = self.lstm(x)
        return out # (B, T, hidden_size * 2)
