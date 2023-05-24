from torch import nn
from util.audio_processing import *

class AudioProcessing(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x, sr = None, info = None):
        raise Exception("not implented")

#Train
class AudioProcessingTrain(AudioProcessing):
    
    def __init__(self, normalizer = None) -> None:
        
        #Init
        super().__init__()
        
        #Fx
        self.random_gain = RandomGain()
        self.normalizer = normalizer
    
    def forward(self, x, sr = None, info = None):
        
        out = x
        
        #Random Gain
        out = self.random_gain(x)
        
        #Normalizer Forward
        if self.normalizer is not None:
            out = self.normalizer(out)
        
        return out

#Test
class AudioProcessingTest(AudioProcessing):
    
    def __init__(self, normalizer = None) -> None:
        
        #Init
        super().__init__()
        
        #Fx
        self.gain = Gain()
        self.normalizer = normalizer
    
    def forward(self, x, sr = None, info = None):
        
        out = x
        
        #Random Gain
        out = self.gain(x, gain = info["gain"])

        #Normalizer Forward
        if self.normalizer is not None:
            out = self.normalizer(out)
        
        return out