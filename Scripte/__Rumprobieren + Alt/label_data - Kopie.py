import os
import streamlit as st
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import mpld3
from mpld3 import plugins
import streamlit.components.v1 as components

#GLOABLS
DF_NAME     = r"result.csv"
PATH        = r"D:\Masterarbeit\SAMPLES PROCESSED\VOICE\TEST"
SAMPLE_RATE = 16000

#Lädt Sample-Namen
@st.cache_data
def load_filenames():

    #Für Return
    samples = []
    
    #Iterriert Samples
    for subdir, dirs, files in os.walk(PATH):
        for file in files:

            #Dateiendung
            fileending = file.split(".")[-1].lower()

            #Chekt ob es sich bei der Dateiendung um Audio-File handelt
            if any([ allowed_filetypes in fileending for allowed_filetypes in ["wav","mp3","ogg","flac"] ]):

                #Fügt neue Zeile an
                samples.append( file )

    return samples

#Lädt Gelabelte Daten
def load_existing_df():
    
    #Lädt CSV
    if os.path.exists(DF_NAME):
        return pd.read_csv(DF_NAME)
    
    #Erzeugt neues DF
    return pd.DataFrame(

        #Cols    
        columns = ["filename", "start", "end"],
        
        #Data
        data    = {
            "filename" : load_filenames()
        }
    )

#Lädt Audiodaten
def load_waveform(filename):
    y, sr = librosa.load(os.path.join(PATH,filename), sr=SAMPLE_RATE, mono=True, dtype="float64")
    return y

def get_non_silent_audio():
    start = librosa.time_to_samples(st.session_state.non_silent[0],sr=SAMPLE_RATE)
    end   = librosa.time_to_samples(st.session_state.non_silent[1],sr=SAMPLE_RATE)
    return st.session_state.waveform[start : end]

def get_silent_audio():
    start = librosa.time_to_samples(st.session_state.non_silent[0],sr=SAMPLE_RATE)
    end   = librosa.time_to_samples(st.session_state.non_silent[1],sr=SAMPLE_RATE)
    return np.concatenate([st.session_state.waveform[:start], st.session_state.waveform[end:]])

#Um IDX zu Ändern
def load_sample_by_idx(idx):
    
    #Ändert idx
    st.session_state.curr_sample_idx = idx
    
    #Ändert filename
    st.session_state.filename = st.session_state.samples.iloc[st.session_state.curr_sample_idx].filename

    #Ändert Waveform
    st.session_state.waveform   = load_waveform(st.session_state.filename)
    
    #Non-Silent
    if pd.isnull(st.session_state.samples.iloc[st.session_state.curr_sample_idx].start) or pd.isnull(st.session_state.samples.iloc[st.session_state.curr_sample_idx].end):
        st.session_state.non_silent = (0., librosa.get_duration(y=st.session_state.waveform, sr=SAMPLE_RATE))
    else:
        st.session_state.non_silent = (float(st.session_state.samples.iloc[st.session_state.curr_sample_idx].start), float(st.session_state.samples.iloc[st.session_state.curr_sample_idx].end))
    
    #Träckt Änderungen
    st.session_state.edited = False

def next_sample(increment):

    #Speichert Werte für aktuelles Sample
    if st.session_state.edited:
        st.session_state.samples.at[st.session_state.curr_sample_idx, "start"] = st.session_state.non_silent[0]
        st.session_state.samples.at[st.session_state.curr_sample_idx, "end"]   = st.session_state.non_silent[1]

    #Lädt neues Sample
    new_idx = st.session_state.curr_sample_idx + increment
    new_idx %= len(st.session_state.samples)
    load_sample_by_idx(new_idx)

def save_dataframe():
    st.session_state.samples.to_csv(DF_NAME,index=False)

def change_sample_start(time_in_seconds = 0):
    start, end = st.session_state.non_silent
    start += time_in_seconds
    st.session_state.non_silent = (start,end)
    st.session_state.edited = True

def change_sample_end(time_in_seconds = 0):
    start, end = st.session_state.non_silent
    end += time_in_seconds
    st.session_state.non_silent = (start,end)
    st.session_state.edited = True

def change_sample_start_end():
    start, end = st.session_state.non_silent
    st.session_state.non_silent = (round(start,2), round(end,2))
    st.session_state.edited = True

#Startup
if "samples" not in st.session_state:
    st.session_state.samples = load_existing_df()
    load_sample_by_idx(0)

st.sidebar.header(f"Sample '{st.session_state.filename}'")
st.sidebar.text(f"Sample {st.session_state.curr_sample_idx + 1} von {len(st.session_state.samples)}")
st.sidebar.progress((st.session_state.curr_sample_idx + 1) / len(st.session_state.samples), text="Fortschritt")
st.sidebar.button("Nächstes Sample",   on_click=next_sample, kwargs=dict(increment=1))
st.sidebar.button("Vorheriges Sample", on_click=next_sample, kwargs=dict(increment=-1))
st.sidebar.button("Speichern",         on_click=save_dataframe)

#UI
st.title("Label Data")
st.header(f"Sample '{st.session_state.filename}'")

fig = plt.figure()
plt.plot(st.session_state.waveform)
plt.ylim((-1,1))

plt.fill_between(
    x=(
        librosa.time_to_samples(times=st.session_state.non_silent[0], sr=SAMPLE_RATE),
        librosa.time_to_samples(times=st.session_state.non_silent[1], sr=SAMPLE_RATE)
    ),
    y1=-1,
    y2=1,
    color="green",
    alpha=0.1
)
st.pyplot(fig)

st.slider("Sprache",
          label_visibility ="collapsed",
          min_value = 0.,
          max_value = librosa.get_duration(y=st.session_state.waveform, sr=SAMPLE_RATE),
          value = st.session_state.non_silent,
          key="non_silent",
          on_change=change_sample_start_end)
#Buttons
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.text("Start")
    st.button("+0.01 S",key="start+0.01", on_click=change_sample_start, kwargs=dict(time_in_seconds=0.01))
    st.button("+0.05 S",key="start+0.05", on_click=change_sample_start, kwargs=dict(time_in_seconds=0.05))
    st.button("+0.1  S",key="start+0.1" , on_click=change_sample_start, kwargs=dict(time_in_seconds=0.10))
with col2:
    st.text("Start")
    st.button("-0.01 S",key="start-0.01", on_click=change_sample_start, kwargs=dict(time_in_seconds=-0.01))
    st.button("-0.05 S",key="start-0.05", on_click=change_sample_start, kwargs=dict(time_in_seconds=-0.05))
    st.button("-0.1  S",key="start-0.1" , on_click=change_sample_start, kwargs=dict(time_in_seconds=-0.10))
with col3:
    st.text("Ende")
    st.button("+0.01 S",key="ende+0.01", on_click=change_sample_end, kwargs=dict(time_in_seconds=0.01))
    st.button("+0.05 S",key="ende+0.05", on_click=change_sample_end, kwargs=dict(time_in_seconds=0.05))
    st.button("+0.1  S",key="ende+0.1" , on_click=change_sample_end, kwargs=dict(time_in_seconds=0.10))
with col4:
    st.text("Ende")
    st.button("-0.01 S",key="ende-0.01", on_click=change_sample_end, kwargs=dict(time_in_seconds=-0.01))
    st.button("-0.05 S",key="ende-0.05", on_click=change_sample_end, kwargs=dict(time_in_seconds=-0.05))
    st.button("-0.1  S",key="ende-0.1" , on_click=change_sample_end, kwargs=dict(time_in_seconds=-0.10))

st.text("Sprache")
st.audio(get_non_silent_audio(), sample_rate=SAMPLE_RATE)
st.text("Noise")
st.audio(get_silent_audio(), sample_rate=SAMPLE_RATE)


st.header("Alle Samples")
st.dataframe(st.session_state.samples)