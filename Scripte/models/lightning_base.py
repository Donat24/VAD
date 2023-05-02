import torch
from torch import nn
from torch.nn import functional as F
import torchmetrics
import lightning.pytorch as pl

#Mixin
class LightningBase(pl.LightningModule):
    def __init__(self) -> None:
        
        super().__init__()

        #Metriken
        self.loss_fn  = nn.BCELoss()
        self.accuracy = torchmetrics.classification.BinaryAccuracy()
    
    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        output  = self(x)
        loss    = self.loss_fn(output, y)
        self.log("train_loss", loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            x, y, _ = batch
            output  = self(x)
            loss    = self.loss_fn(output, y)
            acc     = self.accuracy(output, y)
            self.log("test_loss", loss)
            self.log("test_acc",  acc )
            return loss
    
    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            x, y, _ = batch
            output  = self(x)
            loss    = self.loss_fn(output, y)
            acc     = self.accuracy(output, y)
            self.log("val_loss", loss)
            self.log("val_acc",  acc )
            return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer