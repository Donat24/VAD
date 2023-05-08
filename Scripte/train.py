import argparse
import gc
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger

from data.data import *
import models

def train_model(model, max_epochs=1, max_steps = -1,limit_val_batches=1.0, accelerator = "auto"):

    #Clean
    torch.cuda.empty_cache()
    gc.collect()

    trainer = pl.Trainer(

        #für Debugging
        accelerator = accelerator,

        #Training
        max_epochs = max_epochs,
        max_steps  = max_steps,

        #Logging
        logger=TensorBoardLogger("lightning_logs", name=type(model).__name__ ),
        log_every_n_steps  = 100,
        val_check_interval = 1000,
        limit_val_batches  = limit_val_batches,
        precision          = "16-mixed",
        gradient_clip_val  = 0.7
        
    )
    trainer.fit(model=model, train_dataloaders=dataloader_train, val_dataloaders=dataloader_test)

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
