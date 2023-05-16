import os
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from huggingsound import SpeechRecognitionModel
import librosa

#Parameter
EXTRACTED_DIRECTORY = "./EXTRACTED"

#SAMPLE SIZE
MAX_DURATION = 10
MIN_DURATION = 2

#Models
model_language_mapping = {
    "english"    : "jonatasgrosman/wav2vec2-large-xlsr-53-english",
    "german"     : "jonatasgrosman/wav2vec2-large-xlsr-53-german",
    "dutch"      : "jonatasgrosman/wav2vec2-large-xlsr-53-dutch",
    "french"     : "jonatasgrosman/wav2vec2-large-xlsr-53-french",
    "italian"    : "jonatasgrosman/wav2vec2-large-xlsr-53-italian",
    "spanish"    : "jonatasgrosman/wav2vec2-large-xlsr-53-spanish",
    "portuguese" : "jonatasgrosman/wav2vec2-large-xlsr-53-portuguese",
    "greek"      : "jonatasgrosman/wav2vec2-large-xlsr-53-greek",
    "hungarian"  : "jonatasgrosman/wav2vec2-large-xlsr-53-hungarian",
    "russian"    : "jonatasgrosman/wav2vec2-large-xlsr-53-russian",
    "polish"     : "jonatasgrosman/wav2vec2-large-xlsr-53-polish",
    "chinese"    : "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn",
    "japanese"   : "jonatasgrosman/wav2vec2-large-xlsr-53-japanese",
    "finnish"    : "jonatasgrosman/wav2vec2-large-xlsr-53-finnish",
    "arabic"     : "jonatasgrosman/wav2vec2-large-xlsr-53-arabic",
    "persian"    : "jonatasgrosman/wav2vec2-large-xlsr-53-persian"
}

#Erzeugt Model
def get_model(language):
    return SpeechRecognitionModel(model_language_mapping[language])

#Liefert Wieviele Einträge jeder Sprache verwendet werden sollen
def get_max_entries(language):
    mapping = {
        "german"  : 20000,
        "english" : 10000,
        "other"   : 2000,
    }
    return mapping.get(language, mapping.get("other"))

#Lädt Länge der Datei
def get_file_duration(path):
    try:
        return librosa.get_duration(filename=path)
    except Exception as e:
        return np.NAN

for language in os.listdir(EXTRACTED_DIRECTORY):
    
    #Main Dir
    language_dir = os.path.join(EXTRACTED_DIRECTORY, language)

    #Lädt Tabelle und errechnet Länge
    df = pd.read_table(os.path.join(language_dir, "train.tsv"))
    df["path"]     = df.path.apply(lambda path: path if path.endswith(".mp3") else path + ".mp3")
    df["duration"] = df.path.apply(lambda path: get_file_duration(os.path.join(language_dir, "clips", path)))

    #Wählt Samples mit der entsprechenden Größe aus
    df = df[(df.duration >= MIN_DURATION) & (df.duration <= MAX_DURATION)]

    #Wählt Elemente aus
    if len(df) > get_max_entries(language):
        df = shuffle(df)
        df = df.iloc[ : get_max_entries(language) ]

    #Wählt nur relevante Felder aus
    df = df[["path","duration"]]

    #Kopiert für Out
    result = df.copy()

    #Transscription
    model         = get_model(language)
    transcription = model.transcribe( result.path.apply(lambda path: os.path.join(language_dir, "clips", path)).to_list() )
    result["wav2vec_result"] = transcription
    
    #Export
    result.to_csv(os.path.join(language_dir, "transcription.csv"), index=False)