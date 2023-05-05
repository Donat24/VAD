import torch
from torch import nn
from torch.nn import functional as F
import lightning.pytorch as pl

from .lightning_base import SimpleLightningBase

class MLP(SimpleLightningBase):
    def __init__(self, input_size) -> None:
        
        super().__init__()
        
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512,64)
        self.fc3 = nn.Linear(64,64)
        self.fc4 = nn.Linear(64,64)
        self.fc5 = nn.Linear(64,1)
    
    def forward(self, x):
        
        #forward
        out = torch.relu(self.fc1(x))
        out = torch.relu(self.fc2(out))
        out = torch.relu(self.fc3(out))
        out = torch.relu(self.fc4(out))
        out = torch.sigmoid(self.fc5(out))

        return out.view(out.shape[0])