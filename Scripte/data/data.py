import ast
import librosa
import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

try:
    
    #Für das Script
    from ..util.datasets import *
    from ..util.audio_processing import *

except ImportError as e:
    
    #Für Hauptverzeichniss
    from util.datasets import *
    from util.audio_processing import *

#Konstante
SAMPLE_RATE    = 16000
FIXED_LENGTH   = librosa.time_to_samples(times=7, sr=SAMPLE_RATE) # Trainingsdatensätze bekommen fixe Länge
SAMPLE_LENGTH  = 512
HOP_LENGTH     = 256
TRUTH_TRESHOLD = 64


#Fixt relative Pfade
__current_dir = "/mnt/data/source_jonas/Samples/"#os.path.dirname(os.path.abspath(__file__))

#Lädt CSVs
train_csv = pd.read_csv(os.path.join(__current_dir, "train.csv"))
test_csv  = pd.read_csv(os.path.join(__current_dir, "test.csv"))

#Fixt Spalte mit AST
train_csv["voice"] = train_csv["voice"].apply(ast.literal_eval)
test_csv["voice"]  = test_csv["voice"].apply(ast.literal_eval)

#Erstellt Y-Tensor
def get_y(tensor, sr ,info):
    out = torch.zeros_like(tensor)
    out[info["voice"][0] : info["voice"][1] + 1] = 1
    return out

#FileDataset
filedataset_train              = TarDataset(os.path.join(__current_dir,"train.tar"), data=train_csv, target_samplerate=SAMPLE_RATE)
filedataset_test               = TarDataset(os.path.join(__current_dir,"test.tar"), data=train_csv, target_samplerate=SAMPLE_RATE)
filedataset_train_fixed_length = TarDataset(os.path.join(__current_dir,"train.tar"), data=train_csv, target_samplerate=SAMPLE_RATE, fixed_length = FIXED_LENGTH)
filedataset_test               = TarDataset(os.path.join(__current_dir,"test.tar"),  data=test_csv,  target_samplerate=SAMPLE_RATE)

#AudioProcessing
audio_processing_chain = nn.Sequential(
    RandomGain() #Macht Audio willkürlich lauter oder Leiser
)

#SpeakDataset
speakdataset_train_unchunked              = SpeakDataset(filedataset_train,              audio_processing_chain = None, get_y = get_y)
speakdataset_test_unchunked               = SpeakDataset(filedataset_test,               audio_processing_chain = None, get_y = get_y)
speakdataset_train_unchunked_fixed_length = SpeakDataset(filedataset_train_fixed_length, audio_processing_chain = None, get_y = get_y)
speakdataset_test_unchunked_fixed_length  = SpeakDataset(filedataset_test_fixed_length,  audio_processing_chain = None, get_y = get_y)

#ChunkedDataset
dataset_train = ChunkedDataset(speakdataset_train_unchunked_fixed_length, SAMPLE_LENGTH, HOP_LENGTH, y_truth_treshold = TRUTH_TRESHOLD)
dataset_test  = ChunkedDataset(speakdataset_test_unchunked_fixed_length,  SAMPLE_LENGTH, HOP_LENGTH, y_truth_treshold = TRUTH_TRESHOLD)

#Dataloader für Training
dataloader_train = DataLoader(dataset_train, batch_size=4, shuffle=True, collate_fn=costume_collate_fn, pin_memory=False, num_workers=4)
dataloader_test  = DataLoader(dataset_test,  batch_size=8, shuffle=False, collate_fn=costume_collate_fn, pin_memory=False, num_workers=4)