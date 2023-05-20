from data.data import *
from IPython.display import Audio
from IPython.display import clear_output, display

def get_audio(idxs):
    
    #Cast
    if isinstance(idxs, int):
        idxs = [idxs]
    #Iter
    for idx in idxs:
        x,y = speakdataset_train_unchunked[idx]
        display(Audio(data=x, rate=speakdataset_train_unchunked.dataset.target_samplerate))