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