import numpy as np
import librosa
import librosa.display
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure


def melspectrogram(audio, sr=44100, n_mels=128):
    return librosa.amplitude_to_db(librosa.feature.melspectrogram(audio, sr=sr, n_mels=n_mels))

def show_melspectrogram(mel, sr=44100):
    plt.figure(figsize=(14,4))
    librosa.display.specshow(mel, sr=sr, x_axis='time', y_axis='mel')
    plt.title('Log mel spectrogram')
    plt.colorbar(format='%+02.0f dB')
    plt.tight_layout()

def save_melspectrogram(audio, fname, label, dir='../data/audio'):
    np.savez(f'{dir}/{fname}.npz', data=melspectrogram(audio), label=label)

