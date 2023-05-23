from torch import nn
from util.audio_processing import *

#Train
class AudioProcessingTrain(nn.Module):
    
    def __init__(self) -> None:
        
        #Init
        super().__init__()
        
        #Fx
        self.random_gain = RandomGain()
    
    def forward(self, x, sr = None, info = None):
        out = x
        out = self.random_gain(x)
        return out

#Test
class AudioProcessingTest(nn.Module):
    
    def __init__(self) -> None:
        
        #Init
        super().__init__()
        
        #Fx
        self.gain = Gain()
    
    def forward(self, x, sr = None, info = None):
        out = x
        out = self.gain(x, gain = info["gain"])
        return out