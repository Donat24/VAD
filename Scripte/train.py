import argparse
import gc
import torchmetrics
import tqdm
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from fvcore.nn import FlopCountAnalysis
from tqdm import tqdm

import models
from data.data import *
from util.util import *

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
        callbacks=[
            ModelCheckpoint(
                save_top_k              = 10,
                save_last               = True,
                every_n_train_steps     = 5000,
                #every_n_epochs          = 1,
                save_on_train_epoch_end = True,
            )
        ]
        
    )

    trainer.fit(model=model, train_dataloaders=dataloader_train, val_dataloaders=dataloader_val)

def test_model(model, dataset = dataset_test):

    #Eval
    if model.training:
        model.eval()

    #Result
    result = []

    #Iter Dataset
    for idx, batch in tqdm(enumerate(dataset)):

        #cuda
        if "cuda" in str(model.device):
            batch = tuple([tensor.to(device = model.device) for tensor in batch])
        
        #Train-Step
        batch_result = model.test_step(batch)
        
        #Tensor -> item
        for key, value in batch_result.items():
            if torch.is_tensor(value):
                batch_result[key] = value.item()

        #IDx
        if "idx" not in batch_result:
            batch_result["idx"] = idx
        
        #Append
        result.append(batch_result)

    #Sort
    result.sort( key=lambda item: item["acc"])

    #Return
    return result

def get_model_flops(model, input = None, duration=10, sample_length = SAMPLE_LENGTH, hop_length = HOP_LENGTH, context_length = CONTEXT_LENGTH):
    
    #Erstellt Frame aus 0en
    if input is None:

        #duration to Samples
        duration = librosa.time_to_samples(times=duration, sr=SAMPLE_RATE)
        input = torch.zeros( size = (duration,))
    
    else:
        #Checkt Shape
        if len(input.shape) > 1:
            raise Exception("BAD TENSOR SHAPE")
    
    #Überfluss damit das letzte Frame ausgewertet wird
    overflow = (input.size(-1) - sample_length) % hop_length
    input = torch.concat( [ input, torch.zeros( size=(overflow,) ) ] )
    
    #Context Länge
    input = torch.concat( [ torch.zeros( size=(context_length,) ), input ] )

    #Samples
    input = get_samples(input, sample_length= sample_length + context_length, hop_length = hop_length)
    
    #Kalkuliert Flops
    flops = FlopCountAnalysis(model, input)
    return flops

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
