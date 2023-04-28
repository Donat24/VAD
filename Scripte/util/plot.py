import torch
import matplotlib.pyplot as plt
import librosa
import numpy as np
import pandas as pd
import math

from .util import *

#Torch Tensor -> NP Array
#Librosa kann nicht mit Toch Tensoren arbeiten ....
def _get_np_array(tensor):
    
    #tensor.numpy()
    if hasattr (tensor,"numpy"):
        return tensor.numpy()
    
    return tensor

#Erstellt neue Figure
def _create_new_fig():
    fig = plt.figure()
    ax = plt.axes()
    return fig, ax

def plot_waveform(waveform, sr = None, x_in_sec = True, y_axis_0dbfs_scale = False, ax = None):

    #Fixt x_in_sec
    if sr is None:
        x_in_sec = False

    #Erzeugt neuen Plot
    if ax is None:
        fig, ax = _create_new_fig()

    #x-Achse
    x_axis = torch.arange(0, waveform.shape[-1], dtype=torch.float32)
    if x_in_sec:
        x_axis /= sr

    #y-Achse
    y = waveform

    ax.plot(x_axis,y)

    #X und Y Labels
    if x_in_sec:
        ax.set_xlabel("Zeit in Sekunden")
    else:
        ax.set_xlabel("Samples")
    
    ax.set_ylabel("Amplitute")

    #Y Axe zwischen -1 und 1
    if y_axis_0dbfs_scale:
        ax.set_ylim((-1,1))

    ax.set_title("Waveform Plot")

def plot_waveform_with_voice(waveform, voice, sr = None, x_in_sec = True, ax = None, alpha_voice = 0.1,**kwargs):

    #Fixt x_in_sec
    if sr is None:
        x_in_sec = False

    #Erzeugt neuen Plot
    if ax is None:
        fig, ax = _create_new_fig()

    #Plot Tensor
    plot_waveform(waveform, sr = sr, x_in_sec = x_in_sec, ax = ax, **kwargs)

    #Plot
    for start, end in get_parts(voice):

        if x_in_sec:
            start = librosa.samples_to_time(start, sr=sr)
            end   = librosa.samples_to_time(end, sr=sr)

        #Grüner Hintergrund
        ax.axvspan( xmin = start, xmax = end, alpha = alpha_voice, color="green", label="Sprache")
    

def plot_model_result(x, y, sample_length, hop_length, sr = None, x_in_sec = True, prediction = None, ax = None, **kwargs):

    #Fixt x_in_sec
    if sr is None:
        x_in_sec = False

    #Erzeugt neuen Plot
    if ax is None:
        fig, ax = _create_new_fig()

    #Plottet Waveform
    waveform   = reverse_unfold(x, sample_length, hop_length)
    voice      = reverse_unfold(y.unsqueeze(-1).expand(y.size(-1), sample_length), sample_length, hop_length) 
    plot_waveform_with_voice(waveform, voice, sr = sr, x_in_sec = x_in_sec, ax=ax, alpha_voice = 0.3, **kwargs)
    
    #Plot für Model Prediction
    if prediction != None:
        for start, end in get_parts( reverse_unfold(prediction.unsqueeze(-1).expand(prediction.size(-1), sample_length), sample_length, hop_length) ):
            
            if x_in_sec:
                start = librosa.samples_to_time(start, sr=sr)
                end   = librosa.samples_to_time(end,   sr=sr)
        
            ax.axvspan(
                xmin = start, xmax = end, alpha = 0.05, color="red", lw=1, label = "Vorhersage")


def plot_batch(sample_list, x, y, sample_length, hop_length, prediction=None, **kwargs):

    #Pandas
    sample_list = pd.Series(sample_list, dtype=int)

    #Plot Layout
    X_AXIS_PLOTS = 4
    Y_AXIS_PLOTS = math.ceil( len(sample_list.unique()) / X_AXIS_PLOTS)
    SUBPLOT_WIDTH  = 3
    SUBPLOT_HEIGHT = 1

    #Erzeugt neuen Plot
    fig = plt.figure(figsize = (SUBPLOT_WIDTH * X_AXIS_PLOTS, SUBPLOT_HEIGHT * Y_AXIS_PLOTS))

    for counter, sample_idx in enumerate(sample_list.drop_duplicates(keep='first')):
        
        #Axis
        curr_x  = counter % X_AXIS_PLOTS
        curr_y  = counter // Y_AXIS_PLOTS
        curr_ax = plt.subplot2grid((Y_AXIS_PLOTS, X_AXIS_PLOTS), (curr_y, curr_x), fig=fig)

        #Idx
        idx        = sample_list[sample_list == sample_idx].index
        first_idx  = idx.min()
        last_idx   = idx.max()

        #Params for Plot
        plot_x    = x[first_idx: last_idx + 1]
        plot_y    = y[first_idx: last_idx + 1]
        plot_pred = prediction[first_idx: last_idx + 1] if prediction != None else None
        
        #Subplot
        plot_model_result(plot_x, plot_y, sample_length, hop_length, prediction = plot_pred, ax = curr_ax)
        curr_ax.set_title(f"Sample {sample_idx}")
        curr_ax.set_xlabel("")
        curr_ax.set_ylabel("")
    
    #Plot
    fig.tight_layout()



def plot_spectorgram(waveform, sr, sample_length, hop_length):
    
    #Werte
    window    = np.hanning(sample_length)
    stft      = librosa.stft(y= _get_np_array(waveform), n_fft=sample_length, hop_length=hop_length, window=window, center=False)
    magnitude = np.abs(stft)
    dbfs      = librosa.amplitude_to_db(2 * magnitude / sum(window), ref=1)

    #Plot
    spec = librosa.display.specshow(dbfs, y_axis='log', sr=sr, hop_length=hop_length, x_axis='time')
    plt.colorbar(spec, format="%+2.f dB")

    plt.title("Spectogramm")