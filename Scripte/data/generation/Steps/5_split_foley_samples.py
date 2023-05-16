import os
import warnings
import soundfile as sf
import pandas as pd
import numpy as np
import librosa
from tqdm import tqdm
import shutil

#FÜR SPLIT
LABELS = ["TRAIN", "TEST"]
P      = [0.7, 0.3]

#PATH
FROM = "./NOISES RAW/FOLEYS"
TO   = "./NOISES SPLITTET/FOLEYS"

#Export
MIN_LENGTH        = 1 #s
CHUNK_SIZE        = 5 #s
TARGET_SAMPLERATE = 16000

#zerteilt DataFrame
def chunker(item, chunk_size=1000):
    item_length = len(item)
    for idx_lower in range(0,item_length,chunk_size):
        
        idx_higher = idx_lower + chunk_size
        if idx_higher > item_length:
            idx_higher = item_length
        
        yield item[idx_lower:idx_higher]

#Lädt Datei
def load_audio_file(path, samplerate=TARGET_SAMPLERATE):
    try:
        return librosa.load(path, sr=samplerate, mono=True, dtype="float64")[0]
    
    except Exception as e:
        print("Probleme bei Import von Datei: ", path)
        return None

#Exportiert Datei
def export_audio_file(path, waveform, samplerate=TARGET_SAMPLERATE):
    try:
        
        with warnings.catch_warnings() as e:
            sf.write(
                file       = path,
                data       = waveform,
                samplerate = samplerate,
                subtype    = "PCM_24"
            )

            if e is not None:
                raise e
            
    #Catch Exception
    except Exception as e:
        print("Probleme bei Export von Datei: ", path)
        os.remove(path)

#Lädt alle Sampless
samples = []

for subdir, dirs, files in os.walk(FROM):
    for file in files:
        filepath = os.path.join(subdir, file)
        #Fügt neue Zeile an
        samples.append({
            "filename"                : ".".join(file.split(".")[:-1]),
            "fileending"              : file.split(".")[-1].lower(),
            "filepath"                : filepath,
        })

#zu DataFrame
samples = pd.DataFrame(samples)

#Nur SoundFiles
samples = samples[samples["fileending"].str.contains("wav|mp3|ogg|flac")]
print("Anzahl Samples: ",len(samples))

#ERZEUGT PATH
for label in LABELS:
    path = os.path.join(TO, label)
    
    #Löscht Pfad
    if os.path.exists(path):
        shutil. rmtree(path)
    
    #Erzeugt Dir neu
    os.mkdir(path)

#Arbeitet DF ab in Chunks
__chunk_size = 50
for chunk in tqdm(chunker(samples, chunk_size = __chunk_size)):

    #Kopiert part
    part = chunk.copy()
    
    #Samples
    part["waveform"]     = part.filepath.apply(lambda filepath: load_audio_file(path = filepath))
    part                 = part[ part["waveform"].apply(lambda waveform: waveform is not None) ].copy()

    #Lädt AudioDatei
    part["audio_laenge"] = part.apply(lambda row: librosa.get_duration(y=row["waveform"], sr=TARGET_SAMPLERATE), axis=1)
    part                 = part[ part["audio_laenge"] > MIN_LENGTH ].copy()

    #Teilt
    part["waveform"]     = part.waveform.apply(lambda waveform: [ part for part in chunker(waveform, librosa.time_to_samples(CHUNK_SIZE, sr=TARGET_SAMPLERATE)) ])
    
    #Exloding
    part                 = part.explode("waveform")
    part["audio_laenge"] = part.apply(lambda row: librosa.get_duration(y=row["waveform"], sr=TARGET_SAMPLERATE), axis=1)
    part                 = part[ part["audio_laenge"] > MIN_LENGTH ].copy()

    #RMS
    part["rms"]          = part.waveform.apply(lambda waveform: np.sqrt(np.mean(np.square(waveform))))
    part                 = part[part.rms > 0.05].copy()

    #Nummeriert Teile
    part["part"]         = part.groupby(by="filename").cumcount()
    part["last_part"]    = part.filename.apply(lambda filename: part["filename"].value_counts()[filename]- 1)
    
    #Dateiname
    part["filename"]     = part.apply(lambda row: row["filename"].replace(".","_") + "_part_" + str(row["part"]) + "_" + str(row.name) + ".wav", axis=1)
    
    #Split
    part["LABELS"]       = np.random.choice(LABELS,size=len(part), p=P)
    part["outPath"]      = part["LABELS"].apply(lambda label: os.path.join(TO, label))
    
    #Export
    part.apply(lambda row: export_audio_file(path = os.path.join(row["outPath"], row["filename"]), waveform = row["waveform"]), axis=1)