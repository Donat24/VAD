import torch
from torch import nn
from typing import Any

from util.audio_processing import *
from .lightning_base import LightningBase
from .DeepCNN import Block, DeepCNN

class STFTCNN(DeepCNN):
    def __init__(self, kernel_size=16, mid_channels=32, last_channels=32, n_blocks=1) -> None:

        #Super
        super().__init__(kernel_size, mid_channels, last_channels, n_blocks)

        #STFT
        self.stft = STFT( window = torch.hann_window(512), window_trainable = True , low_treshold=-60)

    
    def forward(self, x):

        #Out
        out = x

        #STFT
        out = self.stft(out)

        #CNN
        out = super().forward(out)

        if out.max() > 1 or out.min() < 0:
            print("WTF")

        return out
