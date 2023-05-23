import torch
from torch import nn
from torch.nn import functional as F
import torchmetrics
import lightning.pytorch as pl
from lion_pytorch import Lion

from data.data import SAMPLE_LENGTH, HOP_LENGTH
from util.util import *
import util.metric as metric

#Für einfache Netzte ohne Puffer
class SimpleLightningBase(pl.LightningModule):
    def __init__(self) -> None:        
        
        super().__init__()        
        
        #Metriken
        self.loss_fn         = F.binary_cross_entropy_with_logits
        self.accuracy        = metric.BinaryAccuracy()
    
    #Shaped Tensor
    def shape_data(self, x, y):
        
        #BATCH TIMESERIES SAMPLE
        if len(x.shape) > 2:
            return x.flatten(start_dim=0, end_dim=1), y.flatten()
        
        #BATCH SAMPLE
        else:
            return x, y

    def training_step(self, batch, batch_idx):
        
        x, y    = batch
        x, y    = self.shape_data(x, y)

        #Forward
        output  = self(x)
        loss    = self.loss_fn(output, y)
        
        self.log("train_loss", loss)
        return loss
        
    #Erzeugt Tensor der wie Y aussieht
    def forward_whole_file(self,x):
        return y_to_full_length(self(x), SAMPLE_LENGTH, HOP_LENGTH)

    def test_step(self, batch, batch_idx = None):
        
        with torch.no_grad():
            
            x, y    = batch
            x, y    = self.shape_data(x, y)
            
            #Forward
            output = self.forward_whole_file(x)
            output = output[..., : y.size(-1)]
            
            #Metrics
            loss    = self.loss_fn(output, y)
            acc     = self.accuracy(output, y)

            #log
            if self._trainer is not None:
                self.log("test_loss", loss)
                self.log("test_acc",  acc )

            return { "loss" : loss, "acc" : acc }
    
    def validation_step(self, batch, batch_idx = None):
        
        with torch.no_grad():
            
            x, y,   = batch
            x, y    = self.shape_data(x, y)
            
            #Forward
            output  = self(x)

            #Metrics
            loss    = self.loss_fn(output, y)
            acc     = self.accuracy(output, y)

            self.log("val_loss", loss)
            self.log("val_acc",  acc )
            
            return { "loss" : loss, "acc" : acc }

    def configure_optimizers(self):        
        
        optimizer = Lion(self.parameters(), lr=1e-4, weight_decay=1e-2)
        return optimizer