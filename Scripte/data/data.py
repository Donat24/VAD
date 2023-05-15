import ast
import librosa
import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
    
from .paths_config import *
from util.datasets import *
from util.audio_processing import *

#Konstante
SAMPLE_RATE    = 16000
SAMPLE_LENGTH  = 512
HOP_LENGTH     = 256
TRUTH_TRESHOLD = 64
FIXED_LENGTH   = librosa.time_to_samples(times=7, sr=SAMPLE_RATE) # Trainingsdatensätze bekommen fixe Länge

#Lädt CSVs
train_csv = pd.read_csv(TRAIN_CSV_PATH)
test_csv  = pd.read_csv(TEST_CSV_PATH)

#Fixt Spalte mit AST
train_csv["voice"] = train_csv["voice"].apply(ast.literal_eval)
test_csv["voice"]  = test_csv["voice"].apply(ast.literal_eval)

#Erstellt Y-Tensor
def get_y(tensor, sr ,info):
    out = torch.zeros_like(tensor)
    out[info["voice"][0] : info["voice"][1] + 1] = 1
    return out

#FileDataset
filedataset_train              = TarDataset(TRAIN_TAR_PATH, data=train_csv, target_samplerate=SAMPLE_RATE)
filedataset_test               = TarDataset(TEST_TAR_PATH,  data=test_csv,  target_samplerate=SAMPLE_RATE)
#filedataset_train_fixed_length = TarDataset(TRAIN_TAR_PATH, data=train_csv, target_samplerate=SAMPLE_RATE, fixed_length = FIXED_LENGTH)
#filedataset_test_fixed_length  = TarDataset(TEST_TAR_PATH,  data=test_csv,  target_samplerate=SAMPLE_RATE)

#AudioProcessing
audio_processing_chain = nn.Sequential(
    RandomGain() #Macht Audio willkürlich lauter oder Leiser
)

#SpeakDataset
speakdataset_train_unchunked              = SpeakDataset(filedataset_train,              audio_processing_chain = audio_processing_chain, get_y = get_y)
speakdataset_test_unchunked               = SpeakDataset(filedataset_test,               audio_processing_chain = None,                   get_y = get_y)
#speakdataset_train_unchunked_fixed_length = SpeakDataset(filedataset_train_fixed_length, audio_processing_chain = audio_processing_chain, get_y = get_y)
#speakdataset_test_unchunked_fixed_length  = SpeakDataset(filedataset_test_fixed_length,  audio_processing_chain = audio_processing_chain, get_y = get_y)

#ChunkedDataset
dataset_train = ChunkedDataset(speakdataset_train_unchunked, SAMPLE_LENGTH, HOP_LENGTH, y_truth_treshold = TRUTH_TRESHOLD)
dataset_val   = ChunkedDataset(speakdataset_train_unchunked, SAMPLE_LENGTH, HOP_LENGTH, y_truth_treshold = TRUTH_TRESHOLD)
dataset_test  = ChunkedDataset(speakdataset_test_unchunked,  SAMPLE_LENGTH, HOP_LENGTH, chunk_y = False)

#Costume Collate
def costume_collate_fn(batch):

    x_list = []
    y_list = []

    #Iter
    for x, y in batch:
        x_list.append(x)
        y_list.append(y)

    #Padding
    x = torch.nn.utils.rnn.pad_sequence(sequences = x_list, batch_first=True, padding_value=0)

    #Stack Y
    y = torch.nn.utils.rnn.pad_sequence(sequences = y_list, batch_first=True, padding_value=0)

    return x, y

#Dataloader für Training
dataloader_train = DataLoader( dataset_train, batch_size=4, shuffle=True,  collate_fn=costume_collate_fn, pin_memory=False, num_workers=0 )
dataloader_val   = DataLoader( dataset_val,   batch_size=4, shuffle=False, collate_fn=costume_collate_fn, pin_memory=False, num_workers=0 )
dataloader_test  = DataLoader( dataset_test,  batch_size=1, shuffle=False, collate_fn=costume_collate_fn, pin_memory=False, num_workers=0 )