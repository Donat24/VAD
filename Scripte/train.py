import argparse
import gc
import torchmetrics
import tqdm
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger

from data.data import *
import models

#Torch Matmul
torch.set_float32_matmul_precision('high')

def train_model(model, max_epochs=1, max_steps = -1,limit_val_batches=1.0, accelerator = "auto", name_params = None):

    #Clean
    torch.cuda.empty_cache()
    gc.collect()

    #Model Name
    name = str( type(model).__name__ )
    if name_params:
        name =  name + "_" + "_".join(f"{key}-{value}" for key, value in name_params.items())

    trainer = pl.Trainer(

        #für Debugging
        accelerator = accelerator,

        #Training
        max_epochs = max_epochs,
        max_steps  = max_steps,

        #Logging
        logger=TensorBoardLogger("lightning_logs", name=name ),
        log_every_n_steps   = 100,
        val_check_interval  = 1000,
        limit_val_batches   = limit_val_batches,
        precision           = "16-mixed",
        gradient_clip_val   = 0.7,

        #Checkpoints
        every_n_train_steps = 5000,
        every_n_epochs      = 1,
        save_on_train_epoch_end = True
        
    )

    trainer.fit(model=model, train_dataloaders=dataloader_train, val_dataloaders=dataloader_val)

def test_model(model):

    #Eval
    model.eval()

    #Test Result
    model.clear_test_result()

    #pytorch Lighning
    trainer = pl.Trainer()
    trainer.test(model, dataloader_test)

    #Test Result
    result = model.get_test_result()
    result.sort( key=lambda item: item["acc"])

    #Return
    return result

def main():

    #CMD Params
    parser = argparse.ArgumentParser( prog="Simple Model Trainer" )
    parser.add_argument("model",     type=str)
    parser.add_argument("max_steps", type=int, nargs='+', default=800)
    args = parser.parse_args()

    #Selects Model
    model = getattr(models, args.model)

    #Other Params
    max_steps = args.max_steps

    train_model(model = model, max_steps= max_steps)

if __name__ == "__main__":
    main()
