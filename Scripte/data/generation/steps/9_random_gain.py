import os
import numpy as np
import pandas as pd

#PARAMETER
CSV_TEST = r"/mnt/data/source_jonas/VAD/Scripte/data/samples/DATA/test.csv"
CSV_VAL  = r"/mnt/data/source_jonas/VAD/Scripte/data/samples/DATA/val.csv"

#Für beide Datasets
for path in [CSV_TEST, CSV_VAL]:
    #Lädt Dataset
    df = pd.read_csv(path)
    
    #Random Gain
    df["gain"] = np.round( np.random.uniform( low = -20, high = 6, size = len(df) ), decimals = 1)
    
    #Export
    df.to_csv(path, index = False)