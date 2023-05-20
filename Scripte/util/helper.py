from data.data import *
from .datasets import *

from IPython.display import Audio
from IPython.display import clear_output, display

def get_audio(dataset, idxs):
    
    #lädt UnchunkedDataset
    if isinstance(dataset, ChunkedDataset):
        dataset = dataset.dataset

    #Cast
    if isinstance(idxs, int):
        idxs = [idxs]
    #Iter
    for idx in idxs:
        x,y = dataset[idx]
        display(Audio(data=x, rate=dataset.dataset.target_samplerate))