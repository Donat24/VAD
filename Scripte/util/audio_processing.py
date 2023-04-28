import torch
from torch import nn

from .util import *

#Modul welches Waveform random gained
class RandomGain(nn.Module):
    def __init__(self, low = -20, high = 6) -> None:
        super().__init__()
        
        #Params
        self.low = low
        self.high = high
    
    @torch.no_grad()
    def forward(self, x):
        
        x = x * db_to_amp(torch.randint(self.low, self.high, size=(1,)))
        
        #HardClipper
        x[x < -1] = -1
        x[x > 1]  = 1
        
        return x

#Modul welches STFT macht
class STFT(nn.Module):
    def __init__(self, window, window_trainable = False, low_treshold = -60) -> None:
        
        #Init
        super().__init__()

        #Parameter
        if window_trainable:

            window = nn.Parameter(window, requires_grad = window_trainable)
            self.register_parameter("window", window)
        
        #Buffer
        else:
            self.register_buffer("window", window)
        
        #Maximaler Negativ-Wert
        self.register_buffer("low_treshold", low_treshold)

    def forward(self, x, normalize=True):

        #Errechnet Spec
        magnitude_spec         = torch.abs(torch.fft.rfft(self.window * x)).mul_(2) # Multiplikation mit 2 da negtiver Teil des Spektrums fehlt
        magnitude_scaled_spec  = magnitude_spec.div_(torch.sum(self.window))        # Skaliert durch Summe von window
        db_spec                = torch.log10_(magnitude_scaled_spec).mul_(20)       # Rechnet in dbFS um
        db_spec[db_spec < self.low_treshold] = self.low_treshold                    # Minimum -> Treshhold
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