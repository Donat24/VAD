import torch
from torch import nn
from torch.nn import functional as F
import torchmetrics
import lightning.pytorch as pl

from util.exception import TrainException

#Für einfache Netzte ohne Puffer
class SimpleLightningBase(pl.LightningModule):
    def __init__(self) -> None:
        
        super().__init__()

        #Metriken
        self.loss_fn  = nn.BCELoss()
        self.accuracy = torchmetrics.classification.BinaryAccuracy()
    
    #Shaped Tensor der Form BATCH TIMESERIES SAMPLE zu N SAMPLE Für X und Y
    def shape_data(self, x, y):
        return x.flatten(start_dim=0, end_dim=1), y.flatten()

    def training_step(self, batch, batch_idx):
        #try:
        x, y    = batch
        x, y    = self.shape_data(x, y)
        output  = self(x)
        loss    = self.loss_fn(output, y)
        self.log("train_loss", loss)
        return loss
        
        #Für Debugging
        #except Exception as e:
        #    raise TrainException(model = self, batch = batch) from e

    
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
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

#Mit Timeseries
class TimeseriesLightningBase(pl.LightningModule):
    def __init__(self) -> None:
        
        super().__init__()

        #Metriken
        self.loss_fn  = nn.BCELoss()
        self.accuracy = torchmetrics.classification.BinaryAccuracy()

    def training_step(self, batch, batch_idx):
        #try:

        #Resetet Model
        self.reset()

        #X, Y, Pred
        x, y   = batch
        output = torch.zeros_like(y)
        
        #Iterriert über File
        for ts, curr_x in enumerate(x.swapaxes(0,1)):
            output[:,ts] = self(curr_x)
        
        loss = self.loss_fn(output, y)
        self.log("train_loss", loss)
        return loss
        
        #Für Debugging
        #except Exception as e:
        #    raise TrainException(model = self, batch = batch) from e

    
    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            
            #Resetet Model
            self.reset()

            #X, Y, Pred
            x, y   = batch
            output = torch.zeros_like(y)
            
            #Iterriert über File
            for ts, curr_x in enumerate(x.swapaxes(0,1)):
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
            for ts, curr_x in enumerate(x.swapaxes(0,1)):
                output[:,ts] = self(curr_x)
            
            loss    = self.loss_fn(output, y)
            acc     = self.accuracy(output, y)
            self.log("val_loss", loss)
            self.log("val_acc",  acc )
            return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer