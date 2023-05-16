import torch
from torch import nn
from torch.nn import functional as F
import torchmetrics
import lightning.pytorch as pl
from lion_pytorch import Lion

from data.data import HOP_LENGTH


#Für einfache Netzte ohne Puffer
class SimpleLightningBase(pl.LightningModule):
    def __init__(self) -> None:        
        
        super().__init__()        
        
        #Metriken
        self.loss_fn         = F.binary_cross_entropy_with_logits
        self.accuracy        = torchmetrics.classification.BinaryAccuracy()

        #Für Test per Batch
        self.clear_test_result()
    
    #Shaped Tensor der Form BATCH TIMESERIES SAMPLE zu N SAMPLE Für X und Y
    def shape_data(self, x, y):
        return x.flatten(start_dim=0, end_dim=1), y.flatten()

    def training_step(self, batch, batch_idx):
        
        x, y    = batch
        x, y    = self.shape_data(x, y)

        #Forward
        output  = self(x)
        loss    = self.loss_fn(output, y)
        
        self.log("train_loss", loss)
        return loss

    #Für Test
    def clear_test_result(self):
        self.test_results = []
    
    #Returned Result
    def get_test_result(self):
        return self.test_results
    
    #Erzeugt Tensor der wie Y aussieht
    def forward_whole_file(self,x):
        output  = self(x)

        #Transform Shape
        transformed = output.unsqueeze(-1).repeat( 1, x.size(-1) )
        last        = transformed[-1]
        all_other   = transformed[ : -1][..., : HOP_LENGTH].flatten()
        output      = torch.concat([all_other, last])

        #Return
        return output

    def test_step(self, batch, batch_idx):
        
        with torch.no_grad():
            
            x, y    = batch
            x, y    = self.shape_data(x, y)
            
            #Forward
            output  = self.forward_whole_file(x)
            
            #Accuracy
            acc = self.accuracy( torch.sigmoid(output), y)
            self.log("test_acc",  acc)
            self.test_results.append({"batch_idx" : batch_idx, "acc" : acc.item()})

            return acc
    
    def validation_step(self, batch, batch_idx):
        
        with torch.no_grad():
            
            x, y,   = batch
            x, y    = self.shape_data(x, y)
            
            output  = self(x)
            loss    = self.loss_fn(output, y)
            acc     = self.accuracy(torch.sigmoid(output), y)

            self.log("val_loss", loss)
            self.log("val_acc",  acc )
            return loss

    def configure_optimizers(self):        
        
        optimizer = Lion(self.parameters(), lr=1e-4, weight_decay=1e-2)
        return optimizer