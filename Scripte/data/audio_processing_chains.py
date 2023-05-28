from torch import nn
from util.audio_processing import *

class AudioProcessing(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x, sr = None, info = None):
        raise Exception("not implented")

#Train
class AudioProcessingTrain(AudioProcessing):
    
    def __init__(self) -> None:
        
        #Init
        super().__init__()
        
        #Fx
        self.random_gain = RandomGain()
    
    def forward(self, x, sr = None, info = None):
        
        out = x
        
        #Random Gain
        out = self.random_gain(x)
        
        return out

#Test
class AudioProcessingTest(AudioProcessing):
    
    def __init__(self) -> None:
        
        #Init
        super().__init__()
        
        #Fx
        self.gain = Gain()
    
    def forward(self, x, sr = None, info = None):
        
        out = x
        
        #Random Gain
        out = self.gain(x, gain = info["gain"])
        
        return out