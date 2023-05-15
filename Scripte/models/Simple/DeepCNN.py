import torch
import librosa
from torch import nn
from torch.nn import functional as F
import lightning.pytorch as pl

from .SimpleLightningBase import SimpleLightningBase
from .net1d import MyConv1dPadSame, MyMaxPool1dPadSame, Swish

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, bn = True, activation_fn = torch.relu) -> None:
        
        super().__init__()

        self.conv          = MyConv1dPadSame(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride)
        self.activation_fn = activation_fn                if activation_fn else None
        self.bn            = nn.BatchNorm1d(out_channels) if bn            else None

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

class ConvConvBlockWithSkip(nn.Module):
    def __init__(self, in_channels, out_channels, ratio, kernel_size, bn = True) -> None:
        super().__init__()
        
        #Anzahl der Channels in der Mitte
        _mid_channels = in_channels * ratio

        self.block_1 = Block( in_channels = in_channels,   out_channels = _mid_channels, kernel_size = kernel_size, bn = True  )
        self.block_2 = Block( in_channels = _mid_channels, out_channels = out_channels,  kernel_size = kernel_size, bn = False )
        self.bn      = nn.BatchNorm1d(out_channels) if bn else None
    
    def forward(self, x):

        out      = x
        identity = x

        #Conv Blocks
        out = self.block_1(out)
        out = self.block_2(out)

        #Skip
        out = out + identity
        
        #Batchnorm
        if self.bn:
            out = self.bn(out)

        #Return
        return out


class DeepCNN(SimpleLightningBase):
    def __init__(self, first_kernel_size = 64, kernel_size = 16, mid_channels = 32, last_channels = 32, n_blocks = 4) -> None:
        
        super().__init__()

        #First Layer
        self.first_cnn_layer = Block(in_channels = 1, out_channels = mid_channels, kernel_size = first_kernel_size, stride = 1)
        
        #Blocks
        self.block_list = nn.ModuleList()
        for i in range(n_blocks):
            self.block_list.append(ConvConvBlockWithSkip(
                in_channels  = mid_channels,
                out_channels = mid_channels,
                ratio        = 2,
                kernel_size  = kernel_size
            ))
        
        #Last CNN Layer
        self.last_cnn_layer = Block(in_channels = mid_channels, out_channels = last_channels, kernel_size = kernel_size, stride = 1)

        #Dense
        self.bn  = nn.BatchNorm1d(last_channels)
        self.fc1 = nn.Linear(last_channels,1)


       
    
    def forward(self, x):
        
        #Out
        out = x

        #reshape
        out = out.unsqueeze(-2)

        #First Layer
        out = self.first_cnn_layer(out)
        
        #Blocks
        for block in self.block_list:
            out = block(out)
        
        #Last Layer
        out = self.last_cnn_layer(out)
        
        #Flatten
        out = torch.avg_pool1d(out, out.size(-1))
        out = out.squeeze()

        #Dense
        out = self.fc1(out)
        #out = torch.sigmoid(out)

        return out.squeeze()