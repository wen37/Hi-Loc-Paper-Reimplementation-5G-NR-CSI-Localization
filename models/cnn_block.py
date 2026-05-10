import torch
import torch.nn as nn

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(CNNBlock, self).__init__()
        # If input is (B, F, T), and we want to keep T dimension
        # Conv1d slides over the last dimension (T)
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
        
    def forward(self, x):
        # x: (B, F, T)
        return self.conv(x) # (B, out_channels, T)
