import os
import pandas as pd
from pytube import YouTube

#PARAMS
path = r"/mnt/data/source_jonas/VAD/Scripte/data/samples/GENERIERUNG/AVA"

#Erstellt Pfad
if not os.path.exists(path):
    os.mkdir(path)

#Lädt CSV
df = pd.read_csv("./ava_speech_labels_v1.csv", header=None)

for id in df[0].unique():
    try:
        yt = YouTube(url=f"https://www.youtube.com/watch?v={id}")
        stream = yt.streams.filter(only_audio=True).filter(abr="128kbps")[0]
        fieending = stream.mime_type.split("/")[-1]
        filename = f"{id}.{fieending}"

        #Skip
        if os.path.exists(os.path.join(path, filename)):
            continue
        
        #Download
        stream.download(output_path = path, filename = filename)
    
    except:
        print(f"Fehler bei ID: '{id}'")