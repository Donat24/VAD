import torch
import librosa
from torch import nn
from torch.nn import functional as F
import lightning.pytorch as pl

from .lightning_base import LightningBase
from .net1d import MyConv1dPadSame, MyMaxPool1dPadSame, Swish

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, bn = True, activation_fn = torch.relu) -> None:
        
        super().__init__()

        self.bn            = nn.BatchNorm1d(in_channels)
        self.conv          = MyConv1dPadSame(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride)
        self.activation_fn = activation_fn
    
    def forward(self, x):

        out = x

        #Batchnorm
        if self.bn:
            out = self.bn(out)
        
        out = self.conv(out)

        #Actiation
        if self.activation_fn:
            out = self.activation_fn(out)

        return out

class ConvPoolConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ratio, kernel_size) -> None:
        super().__init__()
        
        #Anzahl der Channels in der Mitte
        _mid_channels = in_channels * ratio

        self.block_1 = Block( in_channels = in_channels,   out_channels = _mid_channels, kernel_size = kernel_size, bn=True )
        #self.pool    = MyMaxPool1dPadSame( kernel_size = kernel_size )
        self.block_2 = Block( in_channels = _mid_channels, out_channels = out_channels,  kernel_size = kernel_size, bn=True )
    
    def forward(self, x):

        out      = x
        identity = x

        out = self.block_1(out)
        #out = self.pool(out)
        out = self.block_2(out)

        return out + identity


class DeepCNN(LightningBase):
    def __init__(self, first_kernel_size = 64, kernel_size = 16, mid_channels = 32, last_channels = 32, n_blocks = 4) -> None:
        
        super().__init__()

        #First Layer
        self.first_cnn_layer = Block(in_channels = 1, out_channels = mid_channels, kernel_size = first_kernel_size, stride = 1, bn=False)
        
        #Blocks
        self.block_list = nn.ModuleList()
        for i in range(n_blocks):
            self.block_list.append(ConvPoolConvBlock(
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
        out = out.unsqueeze(1)

        #CNN
        out = self.first_cnn_layer(out)
        
        for block in self.block_list:
            out = block(out)
        
        out = self.last_cnn_layer(out)
        
        #Flatten
        out = torch.avg_pool1d(out, out.size(-1))

        #Dense
        out = self.bn(out)
        out = out.squeeze()
        out = self.fc1(out)
        out = torch.sigmoid(out)

        return out.squeeze()