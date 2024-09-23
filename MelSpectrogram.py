import numpy as np
import glob
import gc
import matplotlib.pyplot as plt
import librosa
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import RMSprop
import tensorflow as tf
import matplotlib.pyplot as plt
import os

trainPath = "/home/actvn/KeyBoard/pythonProject/.venv/train"
valPath = "/home/actvn/KeyBoard/pythonProject/.venv/val"
testPath = "/home/actvn/KeyBoard/pythonProject/.venv/test"

imgTrainPath = "/home/actvn/KeyBoard/pythonProject/.venv/train_mel"
imgValPath = "/home/actvn/KeyBoard/pythonProject/.venv/val_mel"
imgTestPath = "/home/actvn/KeyBoard/pythonProject/.venv/test_mel"

os.makedirs(imgTrainPath, exist_ok=True)
os.makedirs(imgValPath, exist_ok=True)
os.makedirs(imgTestPath, exist_ok=True)


def create_spectrogram(filename, name, file_path):
    plt.interactive(False)
    clip,sample_rate = librosa.load(filename,sr=None)
    fig = plt.figure(figsize=[0.72,0.72])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    S = librosa.feature.melspectrogram(y=clip,sr=sample_rate)
    librosa.display.specshow(librosa.power_to_db(S,ref=np.max))
    filename = file_path + "/" + name + '.png'
    plt.savefig(filename,dpi=400,bbox_inches='tight',pad_inches=0)
    plt.close()
    fig.clf()
    plt.close(fig)
    plt.close('all')
    del filename,name,clip,sample_rate,fig,ax,S

for root, dirs, files in os.walk(trainPath):
    for file in files:
        if file.endswith(".wav"):
            file_path = os.path.join(root, file)
            file_name = os.path.splitext(file)[0]
            create_spectrogram(file_path, file_name, imgTrainPath)

for root, dirs, files in os.walk(valPath):
    for file in files:
        if file.endswith(".wav"):
            file_path = os.path.join(root, file)
            file_name = os.path.splitext(file)[0]
            create_spectrogram(file_path, file_name, imgValPath)

for root, dirs, files in os.walk(testPath):
    for file in files:
        if file.endswith(".wav"):
            file_path = os.path.join(root, file)
            file_name = os.path.splitext(file)[0]
            create_spectrogram(file_path, file_name, imgTestPath)