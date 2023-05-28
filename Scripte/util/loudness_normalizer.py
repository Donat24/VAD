import torch
from torch import nn
import torchaudio
import librosa

from .util import *

class FeedForwardLoudnessControll(nn.Module):
    def __init__(self, sample_rate, hop_length, block_length_in_seconds = 0.4, target_loudness = -23, target_loudness_treshold = 0.5, increase_per_second = 10, decrease_per_second = 10, meter_func = None) -> None:
        super().__init__()

        #Params
        self.sample_rate = sample_rate
        self.hop_length  = hop_length
        self.block_length_in_seconds = block_length_in_seconds
        self.block_length_in_samples = librosa.time_to_samples(times=self.block_length_in_seconds, sr=self.sample_rate)

        #Ziel Lautstärk
        self.target_loudness          = target_loudness
        self.target_loudness_treshold = target_loudness_treshold

        #increase und decrase
        self.increase_per_second = increase_per_second
        self.decrease_per_second = decrease_per_second
        self.increase_per_block  = self.hop_length / self.sample_rate * increase_per_second
        self.decrease_per_block  = self.hop_length / self.sample_rate * decrease_per_second

        #Meter
        if meter_func is None:
            self._meter = torchaudio.transforms.Loudness(sample_rate=self.sample_rate)
            meter_func = lambda x: self._meter(x.unsqueeze(0))
         
        self.meter_func = meter_func

        #Params
        self.clip = True
    
    @torch.no_grad()
    def forward(self, x):

        #Gain
        gain = 0
        gain_next = 0

        #Clont Tensor
        out = x.clone()

        #Iterriert über alle Blöcke
        for start_idx in range(0, x.size(0), self.hop_length):
            
            #Errechnet Block
            end_idx = start_idx + self.block_length_in_samples
            block = out[start_idx : end_idx]

            #Checkt auf vollen Block
            if end_idx <=  x.size(0):

                #Calc Error
                block_loudness = self.meter_func(block)
                error = block_loudness - self.target_loudness

                #Calc Gain
                if error + gain < - self.target_loudness_treshold:
                    gain_next = gain + self.increase_per_block
                
                elif error + gain > self.target_loudness_treshold:
                    gain_next = gain - self.decrease_per_block
                
                else:
                    gain_next = gain
                
                #linear Gain
                gain_map = torch.linspace(start=gain, end=gain_next, steps=self.hop_length)

                #Apply Gain
                block[ - self.hop_length : ] *= db_to_amp(gain_map)

                #Setzt Gain für nächsten Block
                gain = gain_next
            
            #Kein voller Block / Ende
            else:
                
                #Wendet Gain auf letzten Teil an
                block[self.block_length_in_samples - self.hop_length : ] *= db_to_amp(gain)
                
        #Clip
        if self.clip:
            out[ out > 1]  = 1
            out[ out < -1] = -1

        return out