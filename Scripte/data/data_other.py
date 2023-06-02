import ast
import pandas as pd

from .paths_config import *
from util.datasets import *
from util.audio_processing import *
from .data import SAMPLE_RATE, SAMPLE_LENGTH, HOP_LENGTH, CONTEXT_LENGTH, TRUTH_TRESHOLD

#Liest CSV
data_ava = pd.read_csv(AVA_CSV_PATH)
data_ava["parts"] = data_ava.parts.apply(ast.literal_eval)

#Erstellt Y-Tensor
def get_y_ava(tensor, sr ,info):
    out = torch.zeros_like(tensor)
    
    #Iterriert Parts
    for label, start, end in info.parts:
        
        if not label == "NO_SPEECH":
            start = librosa.time_to_samples(times=start, sr=sr)
            end   = librosa.time_to_samples(times=end,sr=sr)
            out[start:end] = 1
    
    return out

#Filedataset
filedataset_ava  = LocalFileDataset(root_dir=AVA_DIR_PATH, data=data_ava)
speakdataset_ava = SpeakDataset(filedataset_ava, audio_processing_chain=None, get_y = get_y_ava)