import os
import pandas as pd
import librosa
import subprocess

#PARAMS
path  = r"/mnt/data/source_jonas/VAD/Scripte/data/samples/GENERIERUNG/AVA"

for file in os.listdir(path):
    in_file  = os.path.join(path, file)
    out_file = os.path.join(path, file.split(".")[0] + ".wav")
    
    subprocess.run([f"ffmpeg -ss 900 -to 1800 -i {in_file} -ac 2 -f wav {out_file}"],shell=True)