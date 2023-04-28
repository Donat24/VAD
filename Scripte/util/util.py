import librosa
import torch
from torch.nn import functional as F

#AMP to DB und umgekehrt
def amp_to_db(tensor):
    return tensor.log10() * 20

#DB to Amp
def db_to_amp(tensor):
    return 10.0**(0.5 * tensor/10)

#Berechnet RMS
def rms(tensor):
    tensor = tensor.square()
    tensor = tensor.mean(dim=-1)
    tensor = tensor.sqrt()
    return tensor

#Berechnet DB
def db(tensor):
    return amp_to_db(rms(tensor))

#Funtkion zum Normalisieren der Waveform
def normalize_waveform_to_peak(waveform, peak = -0.1,):
    scale = librosa.db_to_amplitude(peak) / waveform.abs().max()
    return waveform * scale

#Zerelgt Tensor in einzelne Samples
@torch.no_grad()
def get_samples(waveform, sample_length, hop_length):
    
    #Sichert das der Übergebense Tensor die Form[Datenpunkt]
    if len(waveform.shape) != 1:
        raise Exception("BAD TENSOR SHAPE")
    
    #Erzeugt Samples
    return waveform.unfold(0, size = sample_length, step = hop_length)

#Wandelt einzelne Samples wieder in eine Waveform zurück
def reverse_unfold(tensor, sample_length, hop_length):
    
    #Sichert das der Übergebense Tensor die Form[Datenpunkt]
    if len(tensor.shape) != 2:
        raise Exception("BAD TENSOR SHAPE")
    
    #liest variable aus
    rows                 = tensor.size(0)
    cols                 = tensor.size(1)
    tensor_sample_length = (rows - 1) * hop_length + sample_length

    #Erzeugt padding zwischen den Einträgen, flattet den Tensor und sorgt so für einen Versatz
    pad      = tensor_sample_length - cols + hop_length
    reversed = F.pad(tensor, (0, pad), "constant", value=0)
    reversed = reversed.flatten()[:rows * tensor_sample_length].reshape(rows, tensor_sample_length)
    reversed = reversed.sum(0)

    #Summe von reversed muss durch Anzahl der Elemente geteilt werden
    num_entries = torch.ones_like(tensor,dtype=torch.float)
    num_entries = F.pad(num_entries, (0, pad), "constant", value=0)
    num_entries = num_entries.flatten()[:rows * tensor_sample_length].reshape(rows, tensor_sample_length)
    num_entries = num_entries.sum(0)

    #Erzeugt Samples
    return torch.div(reversed, num_entries)

#Liefert zusammenhängende Parts zurück
#TODO: PERFORMANTER MACHEN (ist für Plots aber egal)
def get_parts(tensor, treshhold = 0):

    #Für Return
    parts = []

    #Für 
    start = 0
    end   = 0
    searching = False
    
    for idx, value in enumerate(tensor > treshhold):
        
        #Sucht nach neuem Part
        if value:
            if not searching:
                start     = idx
                end       = idx
                searching = True
            else:
                end = idx
        
        else:
            #Erstellt neuen Part
            if searching:
                searching = False
                parts.append((start,end))
    
    #Falls Part bis zum Ende geht
    if searching:
        parts.append((start, end))
    
    return parts