import os
import pyloudnorm as pyln
import librosa
import torch
from torch import nn
from torch.utils.data import Dataset
import pandas as pd

SAMPLE_LENGTH = 512
HOP_LENGTH    = 256

#erzeugt 30 ms Samples aus der geladenen Datei
@torch.no_grad()
def get_samples(waveform, sample_length = SAMPLE_LENGTH, hop_length = HOP_LENGTH):
    
    #Sichert das der Übergebense Tensor die Form[Datenpunkt]
    if len(waveform.shape) != 1:
        raise Exception("BAD TENSOR SHAPE")
    
    #Erzeugt Samples
    return waveform.unfold(0, size = sample_length, step = hop_length)

def get_y(tensor,sr,info):
    out = torch.zeros_like(tensor)
    out[info["voice"][0] : info["voice"][1] + 1] = 1
    return out

#AMP to DB und umgekehrt
def amp_to_db(tensor):
    return tensor.log10() * 20

def db_to_amp(tensor):
    return 10.0**(0.5 * tensor/10)

#Berechnet RMS
def rms(tensor):
    tensor = tensor.square()
    tensor = tensor.mean(dim=-1)
    tensor = tensor.sqrt()
    return tensor

def db(tensor):
    return amp_to_db(rms(tensor))

class RandomGain(nn.Module):
    def __init__(self, low = -20, high = 6) -> None:
        super().__init__()
        
        #Params
        self.low = low
        self.high = high
    
    @torch.no_grad()
    def forward(self, x):
        
        x = x * db_to_amp(torch.randint(self.low, self.high, size=(1,)))
        
        #HardClipper
        x[x < -1] = -1
        x[x > 1]  = 1
        
        return x

class LocalFileDataset(Dataset):
    def __init__(self, root_dir, csv_file = None, data = None, target_samplerate=16000) -> None:
        super().__init__()
        
        #Verzeichnissordner
        self.root_dir    = root_dir

        #Speichert Data direkt
        if data is not None:
            
            if "filename" not in data.columns:
                raise Exception ("Bad Data Format")
            
            self.data = data

        #Lädt Csv
        elif csv_file is not None:
            
            data = pd.read_csv(csv_file)

            if "filename" not in data.columns:
                raise Exception ("Bad Data Format")
            
            self.data = data

        #Erstellt DataFrame aus Datein im Ordner
        else:
            files = [entry.name for entry in os.scandir(self.root_dir)]
            df    = pd.DataFrame({"filename" : files})
            df    = df[df.filename.str.contains("wav|mp3|ogg|flac")]
            
            self.data = df
        
        #Für Transforms
        self.target_samplerate = target_samplerate
    
    def __len__(self):
            return len(self.data)

    def __getitem__(self, n):

        #lädt Zeile
        row = self.data.iloc[n]

        #lädt Datei
        file_path = os.path.join(self.root_dir, row.filename)
        waveform, sample_rate = librosa.load(file_path, sr=self.target_samplerate, mono=True, dtype="float64")

        #Tensor
        waveform = torch.from_numpy(waveform).to(torch.float32)

        #Return
        return waveform, self.target_samplerate, row
    
    def check_files(self, rms_treshhold = 0.001):
        try:
            for waveform, _, row in self:
                if rms(waveform) < rms_treshhold:
                    raise Exception(f"{row.filename} has rms < {rms_treshhold}")
        
        except Exception as e:
            print(e)


class SpeakDataset(Dataset):
    def __init__(self, dataset) -> None:
        super().__init__()
        self.dataset = dataset
        self.random_gain = RandomGain()
    
    def __len__(self):
            return len(self.dataset)

    @torch.no_grad()
    def __getitem__(self, idx):
        
        #Lädt x
        x, sr, info = self.dataset[idx]

        #Erzeugt y
        y = get_y(x,sr,info)

        #Random Gain
        x = self.random_gain(x)
        
        return x, y

#Chunker
class ChunkedDataset(Dataset):
    def __init__(self, dataset, sample_length = SAMPLE_LENGTH, hop_length = HOP_LENGTH, chunk_y = True) -> None:
        super().__init__()

        #unchunked Dataset
        self.dataset = dataset

        #Chunker Parameter
        self.sample_length = sample_length
        self.hop_length    = hop_length

        #Chunk Y
        self.chunk_y = chunk_y
    
    def __len__(self):
            return len(self.dataset)

    @torch.no_grad()
    def __getitem__(self, idx):
        
        #Lädt x, y für 
        x, y = self.dataset[idx]

        #Erzeugt X
        x = get_samples(x, self.sample_length, self.hop_length)

        #Erzeugt y
        if self.chunk_y:
            y = get_samples(y, self.sample_length, self.hop_length)
            y = y.sum(dim=-1).gt_(0)

        return x, y, idx