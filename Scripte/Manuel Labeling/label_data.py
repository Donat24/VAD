import os
import torch
import streamlit as st
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import mpld3
from mpld3 import plugins
import streamlit.components.v1 as components

#GLOABLS
SAMPLE_RATE = 16000

#Berechnet RMS
def rms(tensor):
    tensor = tensor.square()
    tensor = tensor.mean(dim=-1)
    tensor = tensor.sqrt()
    return tensor

def db(tensor):
    return rms(tensor).log10() * 20

#Lädt Sample-Namen
@st.cache_data
def load_filenames(path):

    #Für Return
    samples = []
    
    #Iterriert Samples
    for subdir, dirs, files in os.walk(path):
        for file in files:

            #Dateiendung
            fileending = file.split(".")[-1].lower()

            #Chekt ob es sich bei der Dateiendung um Audio-File handelt
            if any([ allowed_filetypes in fileending for allowed_filetypes in ["wav","mp3","ogg","flac"] ]):

                #Fügt neue Zeile an
                samples.append( file )

    return samples

#Lädt Gelabelte Daten
def load_or_create_df(df_name,path):
    
    #Lädt CSV
    if os.path.exists(df_name):

        st.session_state.df_existis = True
        
        return pd.read_csv(df_name)
    
    st.session_state.df_existis = False

    #Erzeugt neues DF
    return pd.DataFrame(

        #Cols    
        columns = ["filename", "start", "end"],
        
        #Data
        data    = {
            "filename" : load_filenames(path)
        }
    )

#Lädt Audiodaten
def load_waveform(directory,filename):
    waveform, sr = librosa.load(os.path.join(directory, filename), sr=SAMPLE_RATE, mono=True, dtype="float64")
    #waveform = torch.from_numpy(waveform).to(torch.float32)
    return waveform

#Lädt Dataset
def change_database():
    
    if st.session_state.database == "train":
        st.session_state.df_name = r"train.csv"
        st.session_state.path    = r"D:\Masterarbeit\SAMPLES PROCESSED\VOICE\TRAIN"
    
    elif st.session_state.database == "test":
        st.session_state.df_name = r"test.csv"
        st.session_state.path    = r"D:\Masterarbeit\SAMPLES PROCESSED\VOICE\TEST"
    
    else:
        raise Exception("Something went realy wrong")

    st.session_state.samples = load_or_create_df(df_name=st.session_state.df_name, path=st.session_state.path)
    load_sample_by_idx(0)

#Um IDX zu Ändern
def load_sample_by_idx(idx):
    
    #Ändert idx
    st.session_state.curr_sample_idx = idx
    
    #Ändert filename
    st.session_state.filename = st.session_state.samples.iloc[st.session_state.curr_sample_idx].filename

    #Ändert Waveform
    st.session_state.waveform   = load_waveform(st.session_state.path, st.session_state.filename)
    
    #Träckt Änderungen
    st.session_state.edited = False

    #Unclipp
    st.session_state.unclipp = (0., librosa.get_duration(y=st.session_state.waveform, sr=SAMPLE_RATE))

    #Non-Silent
    if pd.isnull(st.session_state.samples.iloc[st.session_state.curr_sample_idx].start) or pd.isnull(st.session_state.samples.iloc[st.session_state.curr_sample_idx].end):
        st.session_state.non_silent = (0., librosa.get_duration(y=st.session_state.waveform, sr=SAMPLE_RATE))
        ui_autogenerate_start_end()
    else:
        st.session_state.non_silent = (float(st.session_state.samples.iloc[st.session_state.curr_sample_idx].start), float(st.session_state.samples.iloc[st.session_state.curr_sample_idx].end))
    
#Löscht Zeile
def ui_remove_sample():
    
    print(f"Löscht Sample {st.session_state.curr_sample_idx} aus DF")

    #Dropt Zeile
    st.session_state.samples.drop(index=st.session_state.samples.iloc[st.session_state.curr_sample_idx].name, inplace=True)
    
    #Lädt neues Sample
    new_idx = st.session_state.curr_sample_idx % len(st.session_state.samples)
    load_sample_by_idx(new_idx)

#UI Callback für nächsten und vorherigen Button
def ui_next_sample(increment):

    #Speichert Werte für aktuelles Sample
    if st.session_state.edited:
        st.session_state.samples.at[st.session_state.curr_sample_idx, "start"] = st.session_state.non_silent[0]
        st.session_state.samples.at[st.session_state.curr_sample_idx, "end"]   = st.session_state.non_silent[1]

    #Lädt neues Sample
    new_idx = st.session_state.curr_sample_idx + increment
    new_idx %= len(st.session_state.samples)
    load_sample_by_idx(new_idx)

#UI Callback für Speichern
def ui_save_dataframe():
    st.session_state.samples.to_csv(st.session_state.df_name, index=False)

def ui_change_sample_start(time_in_seconds = 0):
    start, end = st.session_state.non_silent
    start += time_in_seconds
    start = max(start, 0)
    st.session_state.non_silent = (start,end)
    st.session_state.edited = True

def ui_change_sample_end(time_in_seconds = 0):
    start, end = st.session_state.non_silent
    end += time_in_seconds
    end = min(end,librosa.get_duration(y=st.session_state.waveform, sr=SAMPLE_RATE))
    st.session_state.non_silent = (start,end)
    st.session_state.edited = True

def ui_change_sample_start_end():
    start, end = st.session_state.non_silent
    st.session_state.non_silent = (round(start,2), round(end,2))
    st.session_state.edited = True

#Liefert Audio-Schnippsel mit Sprache
def get_non_silent_audio():
    start = librosa.time_to_samples(st.session_state.non_silent[0],sr=SAMPLE_RATE)
    end   = librosa.time_to_samples(st.session_state.non_silent[1],sr=SAMPLE_RATE)
    return st.session_state.waveform[start : end]

#Liefert Audio-Schnippsel mit Sprache
def get_silent_audio():
    start = librosa.time_to_samples(st.session_state.non_silent[0],sr=SAMPLE_RATE)
    end   = librosa.time_to_samples(st.session_state.non_silent[1],sr=SAMPLE_RATE)
    return np.concatenate([st.session_state.waveform[:start], st.session_state.waveform[end:]])

#Automatisch generiert Start und Ende
def ui_autogenerate_start_end():
    
    #Nur Falls autmatic_detection aktiviert ist
    if not st.session_state.autmatic_detection:
        return
    
    #Kalkulation
    waveform = torch.from_numpy(st.session_state.waveform).to(torch.float32)
    parts    = waveform.unfold(
        dimension = 0,
        size      = librosa.time_to_samples(0.03,sr=SAMPLE_RATE),
        step      = librosa.time_to_samples(0.01,sr=SAMPLE_RATE)
    )
    parts_db = db(parts)
    parts_db.gt_(parts_db.max() - st.session_state.autmatic_detection_treshhold)
    
    part_starts = torch.flatten(parts_db.nonzero()) * 0.01

    part_starts = part_starts[part_starts >= st.session_state.unclipp[0]]
    part_starts = part_starts[part_starts <= st.session_state.unclipp[1]]

    start   = part_starts.min().item()
    end     = part_starts.max().item() + 0.03

    st.session_state.non_silent = (start, end)

    st.session_state.edited = True

#Startup
if "database" not in st.session_state:
    st.session_state.database = "train"
    st.session_state.autmatic_detection = True
    st.session_state.autmatic_detection_treshhold = 20
    change_database()

st.sidebar.header(f"Datensatz")
st.sidebar.radio("Auswahl",options=["train", "test"], key="database", on_change=change_database, label_visibility ="collapsed")
if st.session_state.df_existis:
    st.sidebar.write(f"Datensatz exisitert")
st.sidebar.button("Speichern",         on_click=ui_save_dataframe)
st.sidebar.markdown("""---""")
st.sidebar.number_input("Treshhold",min_value=5,max_value=40,key="autmatic_detection_treshhold",on_change=ui_autogenerate_start_end)
st.sidebar.checkbox("Automatische Erkennung beim Laden", key="autmatic_detection")
st.sidebar.button("Automatische Kalkulation", on_click=ui_autogenerate_start_end)
st.sidebar.button("Sample Löschen",    on_click=ui_remove_sample)
st.sidebar.markdown("""---""")
st.sidebar.header(f"Sample '{st.session_state.filename}'")
st.sidebar.text(f"Sample {st.session_state.curr_sample_idx + 1} von {len(st.session_state.samples)}")
st.sidebar.progress((st.session_state.curr_sample_idx + 1) / len(st.session_state.samples), text="Fortschritt")
st.sidebar.button("Nächstes Sample",   on_click=ui_next_sample, kwargs=dict(increment=1))
st.sidebar.button("Vorheriges Sample", on_click=ui_next_sample, kwargs=dict(increment=-1))
st.sidebar.number_input(
    label = "Wechsel Sample Idx",
    min_value = 0,
    max_value = len(st.session_state.samples),
    value     = st.session_state.curr_sample_idx,
    key       = "goto_sample_idx",
    on_change= lambda: load_sample_by_idx(st.session_state.goto_sample_idx)
)


#UI
st.title("Label Data")
st.header(f"Sample '{st.session_state.filename}'")

st.slider(
    "Unclipp",
    min_value = 0.,
    max_value = librosa.get_duration(y=st.session_state.waveform, sr=SAMPLE_RATE),
    value = st.session_state.unclipp,
    key="unclipp",
    on_change=ui_change_sample_start_end
)

#Grafik
_y_celling = round(np.max(np.absolute(st.session_state.waveform)),2) + .02

fig = plt.figure()
plt.plot(st.session_state.waveform)
plt.ylim((-_y_celling, _y_celling))

plt.fill_between(
    x=(
        0,
        librosa.time_to_samples(times=st.session_state.unclipp[0], sr=SAMPLE_RATE)
    ),
    y1=-_y_celling,
    y2=_y_celling,
    color="red",
    alpha=0.4
)

plt.fill_between(
    x=(
        librosa.time_to_samples(times=st.session_state.unclipp[1], sr=SAMPLE_RATE),
        len(st.session_state.waveform)
    ),
    y1=-_y_celling,
    y2=_y_celling,
    color="red",
    alpha=0.4
)

plt.fill_between(
    x=(
        librosa.time_to_samples(times=st.session_state.non_silent[0], sr=SAMPLE_RATE),
        librosa.time_to_samples(times=st.session_state.non_silent[1], sr=SAMPLE_RATE)
    ),
    y1=-_y_celling,
    y2=_y_celling,
    color="green",
    alpha=0.1
)

st.pyplot(fig)

st.slider(
    "Sprache",
    label_visibility ="collapsed",
    min_value = 0.,
    max_value = librosa.get_duration(y=st.session_state.waveform, sr=SAMPLE_RATE),
    value = st.session_state.non_silent,
    key="non_silent",
    on_change=ui_change_sample_start_end
)

#Buttons
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.text("Start")
    st.button("+0.01 S",key="start+0.01", on_click=ui_change_sample_start, kwargs=dict(time_in_seconds=0.01))
    st.button("+0.05 S",key="start+0.05", on_click=ui_change_sample_start, kwargs=dict(time_in_seconds=0.05))
    st.button("+0.1  S",key="start+0.1" , on_click=ui_change_sample_start, kwargs=dict(time_in_seconds=0.10))
with col2:
    st.text("Start")
    st.button("-0.01 S",key="start-0.01", on_click=ui_change_sample_start, kwargs=dict(time_in_seconds=-0.01))
    st.button("-0.05 S",key="start-0.05", on_click=ui_change_sample_start, kwargs=dict(time_in_seconds=-0.05))
    st.button("-0.1  S",key="start-0.1" , on_click=ui_change_sample_start, kwargs=dict(time_in_seconds=-0.10))
with col3:
    st.text("Ende")
    st.button("+0.01 S",key="ende+0.01", on_click=ui_change_sample_end, kwargs=dict(time_in_seconds=0.01))
    st.button("+0.05 S",key="ende+0.05", on_click=ui_change_sample_end, kwargs=dict(time_in_seconds=0.05))
    st.button("+0.1  S",key="ende+0.1" , on_click=ui_change_sample_end, kwargs=dict(time_in_seconds=0.10))
with col4:
    st.text("Ende")
    st.button("-0.01 S",key="ende-0.01", on_click=ui_change_sample_end, kwargs=dict(time_in_seconds=-0.01))
    st.button("-0.05 S",key="ende-0.05", on_click=ui_change_sample_end, kwargs=dict(time_in_seconds=-0.05))
    st.button("-0.1  S",key="ende-0.1" , on_click=ui_change_sample_end, kwargs=dict(time_in_seconds=-0.10))

st.text("Sprache")
st.audio(get_non_silent_audio(), sample_rate=SAMPLE_RATE)
st.text("Noise")
st.audio(get_silent_audio(), sample_rate=SAMPLE_RATE)


st.header("Alle Samples")
st.dataframe(st.session_state.samples)