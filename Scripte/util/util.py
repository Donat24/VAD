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
def reverse_unfold(tensor, hop_length):
    
    #Sichert das der Übergebense Tensor die Form[Datenpunkt]
    if len(tensor.shape) != 2:
        raise Exception("BAD TENSOR SHAPE")
    
    #Letztes Frame wird ganz verwendet, alle Anderen nur bis zur Hop-Length 
    last_frame       = tensor[-1]
    all_other_frames = tensor[ : -1][..., : hop_length].flatten()

    #Verbindet Frames
    return torch.concat([last_frame, all_other_frames])

def y_to_full_length(tensor, sample_length, hop_length):
    
    #Sichert das der Übergebense Tensor die Form[Datenpunkt]
    if len(tensor.shape) != 1:
        raise Exception("BAD TENSOR SHAPE")
    
    #Skaliert Y auf Länge
    tensor = tensor.unsqueeze(-1).repeat( 1, sample_length )
    
    #Returned
    return reverse_unfold(tensor, hop_length)

#Liefert zusammenhängende Parts zurück
#TODO: PERFORMANTER MACHEN (ist für Plots aber egal)
def get_parts(tensor, treshold = 0):

    #Für Return
    parts = []

    #Für 
    start = 0
    end   = 0
    searching = False
    
    for idx, value in enumerate(tensor > treshold):
        
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

#Liefert zusammenhängende Parts mit Buffer am Ende
def get_parts_with_buffer(tensor, treshold = 0, decay = librosa.time_to_samples(times = 0.3, sr = 16000)):
    
    #Falls decay wird neuer Tensor erzeugt
    if decay:

        #out
        out = torch.zeros_like(tensor)
        
        #iterriert parts
        for start, end in get_parts(tensor, treshold):
            out[start: end + decay] = True
    
    #Sonst einfach input verwenden
    else:
        out = tensor
    
    return get_parts(out, treshold)