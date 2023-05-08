import torch
from torch import nn
from torch.nn import functional as F
import torchmetrics
import lightning.pytorch as pl
from lion_pytorch import Lion

#Für einfache Netzte ohne Puffer
class SimpleLightningBase(pl.LightningModule):
    def __init__(self) -> None:        
        
        super().__init__()        
        
        #Metriken
        self.loss_fn  = F.binary_cross_entropy_with_logits
        self.accuracy = torchmetrics.classification.BinaryAccuracy(threshold = 0)
    
    #Shaped Tensor der Form BATCH TIMESERIES SAMPLE zu N SAMPLE Für X und Y
    def shape_data(self, x, y):
        return x.flatten(start_dim=0, end_dim=1), y.flatten()

    def training_step(self, batch, batch_idx):
        
        x, y    = batch
        x, y    = self.shape_data(x, y)
        output  = self(x)
        loss    = self.loss_fn(output, y)
        self.log("train_loss", loss)
        return loss

    
    def test_step(self, batch, batch_idx):
        
        with torch.no_grad():
            
            x, y    = batch
            x, y    = self.shape_data(x, y)
            output  = self(x)
            loss    = self.loss_fn(output, y)
            acc     = self.accuracy(output, y)
            self.log("test_loss", loss)
            self.log("test_acc",  acc )
            return loss
    
    def validation_step(self, batch, batch_idx):
        
        with torch.no_grad():
            
            x, y,   = batch
            x, y    = self.shape_data(x, y)
            output  = self(x)
            loss    = self.loss_fn(output, y)
            acc     = self.accuracy(output, y)
            self.log("val_loss", loss)
            self.log("val_acc",  acc )
            return loss

    def configure_optimizers(self):        
        
        optimizer = Lion(self.parameters(), lr=1e-4, weight_decay=1e-2)
        return optimizer