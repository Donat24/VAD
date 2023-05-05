import torch
from torch import nn
from torch.nn import functional as F
import lightning.pytorch as pl

from .lightning_base import SimpleLightningBase

class CNN(SimpleLightningBase):
    def __init__(self, channels = 32) -> None:
        
        super().__init__()

            
        self.conv1 = nn.Conv1d(in_channels = 1, out_channels = channels, kernel_size = 160, stride = 4)
        self.bn1   = nn.BatchNorm1d(channels)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(in_channels = channels, out_channels = channels, kernel_size = 4,  stride = 1)
        self.bn2   = nn.BatchNorm1d(channels)
        self.pool2 = nn.MaxPool1d(2)
        self.conv3 = nn.Conv1d(in_channels = channels, out_channels = channels, kernel_size = 2,  stride = 1)
        self.bn3   = nn.BatchNorm1d(channels)
        self.pool3 = nn.MaxPool1d(2)
        self.fc1   = nn.Linear(channels,1)
    
    def forward(self, x):
        
        #reshape
        x = x.unsqueeze(1)

        #forward
        out = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        out = self.pool2(torch.relu(self.bn2(self.conv2(out))))
        out = self.pool3(torch.relu(self.bn3(self.conv3(out))))
        
        out = torch.avg_pool1d(out,out.size(-1)).squeeze()

        out = self.fc1(out)
        #out = torch.sigmoid(out)
        return out.squeeze()