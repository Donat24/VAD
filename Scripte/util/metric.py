import torch
import torchaudio
from torch import nn
from torch.nn import functional as F

class BinaryAccuracy(nn.Module):
    def __init__(self, treshold = 0) -> None:
        #Init
        super().__init__()
        
        #Params
        self.treshold = treshold
    
    def forward(self, pred, y):
        
        #Flatten
        pred = pred.flatten()
        y    = y.flatten()

        #Size
        if pred.size(0) != y.size(0):
            raise Exception("TENSOR-SHAPES DOESNT MATCH")

        #shiftet Treshold auf 0
        if self.treshold:
            pred = pred - self.treshold

        #Heavyside sorgt für 0 oder 1
        pred = torch.heaviside(pred,values=torch.tensor(1))
        
        #Return
        return (pred == y).sum() / y.size(0)