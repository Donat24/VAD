import torch
import torch.nn as nn
from nnAudio import features
from .SimpleLightningBase import SimpleLightningBase
from util.util import *

class Block2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bn = True, activation_fn = torch.relu, padding = "same") -> None:
        
        super().__init__()

        self.conv          = nn.Conv2d (in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, padding = padding)
        self.activation_fn = activation_fn                if activation_fn else None
        self.bn            = nn.BatchNorm2d(out_channels) if bn            else None

    def forward(self, x):

        out = x
        
        #Conv
        out = self.conv(out)

        #Actiation
        if self.activation_fn is not None:
            out = self.activation_fn(out)

        #Batchnorm
        if self.bn is not None:
            out = self.bn(out)

        return out

class STFTCNN(SimpleLightningBase):
    def __init__(self, stft_n_fft = 512, stft_hop_length = 512, stft_sr = None, first_kernel_size = 16, kernel_size = 16, mid_channels=32, last_channels=32, n_blocks = 1, dense_features = 32) -> None:
        
        super().__init__()

        #STFT
        self.stft = features.STFT(
            n_fft         = stft_n_fft,
            hop_length    = stft_hop_length,
            sr            = stft_sr,
            output_format ="Magnitude"
        )

        #STFT Window
        self.stft_window_sum = self.stft.window_mask.sum()

        #CONV
        self.first_cnn_layer = Block2D(in_channels = 1, out_channels = mid_channels, kernel_size = first_kernel_size)

        #Blocks
        self.block_list = nn.ModuleList()
        for i in range(n_blocks):
            self.block_list.append( Block2D (
                in_channels  = mid_channels,
                out_channels = mid_channels,
                kernel_size  = kernel_size
            ))

        #Last CNN Layer
        self.last_cnn_layer = self.last_cnn_layer = Block2D(in_channels = mid_channels, out_channels = last_channels, kernel_size = kernel_size)

        #Dense
        self.fc1 = nn.Linear(last_channels, dense_features)
        self.bn1 = nn.BatchNorm1d(last_channels)
        self.fc2 = nn.Linear(dense_features, 1)

    
    def forward(self, x):

        #Out
        out = x

        #FFT
        out = self.stft(out)

        #Rescale
        out = amp_to_db(rescale_fft_magnitude_with_window(tensor = out, window_sum = self.stft_window_sum), low_treshold = -100)
        out = (out + 100) / 100

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
        out = torch.nn.functional.avg_pool2d(out,out.shape[-2:])
        out = out.squeeze(dim=(-2,-1))

        #Dense
        out = self.fc1(out)
        out = torch.relu(out)
        out = self.bn1(out)
        
        out = self.fc2(out)
        #out = torch.sigmoid(out)
        out = out.squeeze()

        return out