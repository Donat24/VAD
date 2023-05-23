import torch
from torch import nn
import torchaudio

from .util import *

#Gain Util
class Gain(nn.Module):
    def __init__(self ) -> None:
        super().__init__()
    
    @torch.no_grad()
    def forward(self, x, gain = 0):
        
        x = x * db_to_amp(gain)
        
        #HardClipper
        x[x > 1]  = 1
        x[x < -1] = -1
        
        return x


#Modul welches Waveform random gained
class RandomGain(Gain):
    def __init__(self, low = -20, high = 6) -> None:
        super().__init__()
        
        #Params
        self.low = low
        self.high = high
    
    @torch.no_grad()
    def forward(self, x):
        return super().forward(x, gain = torch.randint(self.low, self.high, size=(1,)))

#Puffert
class AudioBuffer(nn.Module):
    def __init__(self, size, shift) -> None:
        
        #Init
        super().__init__()

        #Parameter
        self.size  = size
        self.shift = -shift

        #Puffer
        self.buffer = None
    
    #Setzt Buffer zurück
    def reset(self):
        self.buffer = None

    def forward(self, x):

        #Erzeugt neuen Buffer
        if self.buffer is None:
            _size     = list(x.size())
            _size[-1] = self.size
            self.buffer = torch.zeros(size = tuple(_size), dtype = x.dtype)
        
        #Shifftet Buffer
        self.buffer = torch.roll(input = self.buffer, shifts = self.shift)

        #Neue Daten
        self.buffer[..., - x.size(-1) : ] = x

        #Return
        return self.buffer

#Modul welches FFT macht
class FFT(nn.Module):
    def __init__(self, window, low_treshold = -60) -> None:
        
        #Init
        super().__init__()

        #Window
        self.register_buffer("window", window)
        
        #Maximaler Negativ-Wert
        low_treshold = torch.tensor(low_treshold, dtype=torch.float)
        self.register_buffer("low_treshold", low_treshold)

    def forward(self, x, normalize=True):

        #Errechnet Spec
        magnitude_spec         = torch.abs(torch.fft.rfft(self.window * x))                            # Multiplikation mit 2 da negtiver Teil des Spektrums fehlt
        magnitude_scaled_spec  = rescale_fft_magnitude_with_window(magnitude_spec, window=self.window) # Skaliert durch Summe von window
        db_spec                = amp_to_db(magnitude_scaled_spec, self.low_treshold)                   # Rechnet in dbFS um
        spec = db_spec

        #Normalisiert
        if normalize:
            return (spec + torch.abs(self.low_treshold)) / torch.abs(self.low_treshold)
        
        #nicht normalisiert
        return spec

#Modul für Compressor
class Compressor(nn.Module):
    def __init__(self, treshold = -20, ratio = 4, input_gain = 0, out_gain = 0, trainable=True) -> None:
        
        #Init
        super().__init__()

        #Converting Params
        treshold   = db_to_amp(torch.tensor(treshold))
        ratio      = torch.tensor(1 / ratio)
        input_gain = db_to_amp(torch.tensor(input_gain))
        out_gain   = db_to_amp(torch.tensor(out_gain))
    
        #Treshold
        treshold = nn.Parameter(treshold, requires_grad = trainable)
        self.register_parameter("treshold", treshold)

        #Ratio
        ratio = nn.Parameter(ratio, requires_grad = trainable)
        self.register_parameter("ratio", ratio)
        
        #Input Gain
        input_gain = nn.Parameter(input_gain, requires_grad = trainable)
        self.register_parameter("input_gain", input_gain)

        #Out Gain
        out_gain = nn.Parameter(out_gain, requires_grad = trainable)
        self.register_parameter("out_gain", out_gain)
    
    def forward(self, x):

        #Input Gain
        x = x * self.input_gain

        #Berechnet Mask
        treshold_db    = amp_to_db (self.treshold)
        x_db           = amp_to_db (x.abs())
        mask           = x_db > treshold_db
        
        #Wendet Compression an
        affected_rows_in_db  = x_db[mask]
        affected_rows_out_db = treshold_db + (affected_rows_in_db - treshold_db) * self.ratio
        gain_reduction       = affected_rows_out_db - affected_rows_in_db
        factor               = db_to_amp(gain_reduction)
        x[mask]              *= factor
        
        #Out Gain
        x = x * self.out_gain

        #Returnt x
        return x

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