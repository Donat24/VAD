import torch

#Eigene Sachen
from data.data import *
from util.util import *
from util.datasets import *

#Model
_model = None

#Dataset
SAMPLE_LENGTH = 512
DATASET = ChunkedDataset(speakdataset_test_unchunked,
    sample_length  = SAMPLE_LENGTH,
    hop_length     = SAMPLE_LENGTH,
    context_length = 0,
    chunk_y = False,
    fill_y_to_sample_length = False
)

#Lädt Model
def load_model():
    
    #Global
    global _model

    #Lädt Model
    model, util = torch.hub.load(
        repo_or_dir  = 'snakers4/silero-vad',
        model        = 'silero_vad',
        #force_reload = True,
        trust_repo   = True
    )

    #EVAL
    model = model.eval()

    #Setzt Model
    _model = model

#Returned Model
def get_model():

    #Global
    global _model
    
    #Falls Model schon geladen wurde
    if _model is not None:
        return _model
    
    #Lädt Model
    load_model()

    #Returned Model
    return _model

#Vohersagen
def predict(x, sample_rate = DATASET.dataset.dataset.target_samplerate):

    #No Grad
    with torch.no_grad():

        #Erhält Model
        model = get_model()
        
        #Reset
        model.reset_states()

        #Out
        pred = []

        #Iterriert über Chunks
        for chunk in x:
            pred.append ( model(chunk, sample_rate).squeeze() )
        
        #Verlängert Pred auf Länge des Tensors
        return torch.stack(pred).unsqueeze(1).repeat( [1, x.size(-1)] ).flatten()