import torch
from torch import nn
from torch.nn import functional as F
import torchmetrics
import lightning.pytorch as pl
from lion_pytorch import Lion

#Mit Timeseries
class TimeseriesLightningBase(pl.LightningModule):
    def __init__(self) -> None:
        
        super().__init__()

        #Metriken
        self.loss_fn  = F.binary_cross_entropy_with_logits
        self.accuracy = torchmetrics.classification.BinaryAccuracy()

    def training_step(self, batch, batch_idx):

        #Resetet Model
        self.reset()

        #X, Y, Pred
        x, y   = batch
        output = torch.zeros_like(y)
        
        #Iterriert über File
        for ts, curr_x in enumerate( x.swapaxes(0,1) ):
            output[:,ts] = self(curr_x)
        
        loss = self.loss_fn(output, y)
        self.log("train_loss", loss)
        return loss

    
    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            
            #Resetet Model
            self.reset()

            #X, Y, Pred
            x, y   = batch
            output = torch.zeros_like(y)
            
            #Iterriert über File
            for ts, curr_x in enumerate( x.swapaxes(0,1) ):
                output[:,ts] = self(curr_x)
            
            loss    = self.loss_fn(output, y)
            acc     = self.accuracy(output, y)
            self.log("test_loss", loss)
            self.log("test_acc",  acc )
            return loss
    
    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            
            #Resetet Model
            self.reset()

            #X, Y, Pred
            x, y   = batch
            output = torch.zeros_like(y)
            
            #Iterriert über File
            for ts, curr_x in enumerate( x.swapaxes(0,1) ):
                output[:,ts] = self(curr_x)
            
            loss    = self.loss_fn(output, y)
            acc     = self.accuracy(output, y)
            self.log("val_loss", loss)
            self.log("val_acc",  acc )
            return loss

    def configure_optimizers(self):
        
        optimizer = Lion(self.parameters(), lr=1e-4,weight_decay=1e-2)
        return optimizer