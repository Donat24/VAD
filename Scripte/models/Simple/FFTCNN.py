import torch
from torch import nn
from typing import Any

from util.audio_processing import *
from .SimpleLightningBase import SimpleLightningBase
from .DeepCNN import Block, DeepCNN

class FFTCNN(SimpleLightningBase):
    def __init__(self, fft_window = torch.hann_window(512), first_kernel_size = 16, kernel_size = 16, mid_channels=32, last_channels=32, n_blocks = 1, dense_features = 32) -> None:

        #Super
        super().__init__()

        #FFT
        self.fft = FFT( window = fft_window, window_trainable = False , low_treshold = -60)

        #First Layer
        self.first_cnn_layer = Block( in_channels = 1, out_channels = mid_channels, kernel_size = first_kernel_size, stride = 1, bn = False)

        #Blocks
        self.block_list = nn.ModuleList()
        for i in range(n_blocks):
            self.block_list.append( Block (
                in_channels  = mid_channels,
                out_channels = mid_channels,
                kernel_size  = kernel_size
            ))
        
        #Last CNN Layer
        self.last_cnn_layer = self.last_cnn_layer = Block(in_channels = mid_channels, out_channels = last_channels, kernel_size = kernel_size, stride = 1)

        #Dense
        self.fc1 = nn.Linear(last_channels, dense_features)
        self.bn1 = nn.BatchNorm1d(last_channels)
        self.fc2 = nn.Linear(dense_features, 1)
    
    def forward(self, x):

        #Out
        out = x

        #FFT
        out = self.fft(out)

        #Reshape
        out = out.unsqueeze(1)

        #First Layer
        out = self.first_cnn_layer(out)
        
        #Blocks
        for block in self.block_list:
            out = block(out)
        
        #Last CNN Layer
        out = self.last_cnn_layer(out)

        #Flatten
        out = torch.avg_pool1d(out, out.size(-1))
        out = out.squeeze()

        #Dense
        out = self.fc1(out)
        out = torch.relu(out)
        out = self.bn1(out)
        
        out = self.fc2(out)
        #out = torch.sigmoid(out)
        out = out.squeeze()
        
        return out
