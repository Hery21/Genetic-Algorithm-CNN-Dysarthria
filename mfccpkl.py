import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from python_speech_features import mfcc
from python_speech_features import logfbank
from python_speech_features.base import delta
import pickle


def plot_signals(signals):
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False,
                             sharey=True, figsize=(20,5))
    fig.suptitle('Time Series', size=16)
    i = 0
    for x in range(2):
        for y in range(5):
            axes[x,y].set_title(list(signals.keys())[i])
            axes[x,y].plot(list(signals.values())[i])
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1

def plot_fft(fft):
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False,
                             sharey=True, figsize=(20,5))
    fig.suptitle('Fourier Transforms', size=16)
    i = 0
    for x in range(2):
        for y in range(5):
            data = list(fft.values())[i]
            Y, freq = data[0], data[1]
            axes[x,y].set_title(list(fft.keys())[i])
            axes[x,y].plot(freq, Y)
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1

def plot_mfccs(mfccs):
    fig, axes = plt.subplots(nrows=4, ncols=4, sharex=False,
                             sharey=True, figsize=(20,5))
    fig.suptitle('Mel Frequency Cepstrum Coefficients', size=16)
    i = 0
    for x in range(4):
        for y in range(4):
            axes[x,y].set_title(list(mfccs.keys())[i])
            axes[x,y].imshow(list(mfccs.values())[i],
                    cmap='hot', interpolation='nearest')
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1

def calc_fft(y, rate):
    n = len(y)
    freq = np.fft.rfftfreq(n, d=1/rate)
    Y = abs(np.fft.rfft(y)/n)
    return (Y,freq)

lists = {'filename':[], 'label':[]}
for i in os.listdir():
  if i.endswith(".wav"):
    lists['filename'].append(i)
    lists['label'].append(str(i[-8]))
#print(lists)
df = pd.DataFrame(lists)
df.set_index('filename', inplace=True)

signals = {}
fft = {}
mfccs = {}
fbank = {}

for g in df.index:
    rate, signal = wavfile.read(g)
    signals[g] = signal
    fft[g] = calc_fft(signal, rate)
    signal = signal.astype(float)
    mel = mfcc(signal, samplerate=rate,
               numcep=13, nfilt=26).T
    d = delta(mel, 2)
    dd = delta(d, 2)
    features = np.array([mel, d, dd])
    #features = features.reshape(len(mel),13,3)
    mfccs[g] = features

#print(mfccs["F02_B1_D5_M2.wav"][:10])

'''
plot_signals(signals)
plt.show()

plot_fft(fft)
plt.show()

plot_fbank(fbank)
plt.show()


plot_mfccs(mfccs)
plt.show()
'''
f = open("F03_186*13.pkl", "wb")
pickle.dump(mfccs,f)
f.close()

print(mfccs['F03_B2_D7_M5.wav'].shape)