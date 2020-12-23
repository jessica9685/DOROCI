# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.8.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import librosa
import os
import numpy as np

# +
sample = 'D:\\졸프\\scream\\scream_3.wav'
y, sr = librosa.load(sample)
print('sr: ', sr)
print('wav shape: ', wav.shape)
print('length: ', wav.shape[0]/float(sr), 'secs')

mfcc = librosa.feature.mfcc(y=y, sr=sr)
print (mfcc.shape)

# +
scream_wav_path = np.zeros(shape=(140), dtype='object')
animal_wav_path = np.zeros(shape=(100), dtype='object')
for i in range(140):
    path = 'D:\\졸프\\scream\\scream_'
    n = i + 1
    scream_wav_path[i] = path + str(n) + '.wav'
    print(scream_wav_path[i])

for j in range(100):
    path = 'D:\\졸프\\animal\\animal_'
    m= j + 1
    animal_wav_path[j] = path + str(m) + '.wav'
    print(animal_wav_path[j])

# +
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

frame_length = 0.025
frame_stride = 0.010
hop_length = 512
n_fft = 2048 

for i in range(140):
    audio_path = 'D:\\졸프\\scream\\scream_'
    mel_path = 'D:\\졸프\\scream\\MFCC\\scream_'
    num = i + 1
    a_name = audio_path + str(num) + '.wav'
    m_name = mel_path + str(num) + '.png'

    # mel-spectrogram
    # 음성 로드
    # y = 오디오 시간, sr = y의 sampling rate
    y, sr = librosa.load(a_name, sr=16000)

    # wav_length = len(y)/sr
    # 시간 관점의 음성 데이터를 주파수 관점으로 변환
    input_nfft = int(round(sr*frame_length))
    input_stride = int(round(sr*frame_stride))

    # wav 파일을 melspectrogram으로 변환
    MFCCs = librosa.feature.mfcc(y, sr, hop_length=hop_length, n_mfcc=13)
    print("Wav length: {}, MFCC shape:{}".format(len(y)/sr,np.shape(MFCCs)))
  

    # display MFCCs
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(MFCCs, sr=sr, hop_length=hop_length)
  
    # show plots
    plt.savefig(m_name, format='png', bbox_inches='tight', pad_inches=0, facecolor='black')
    plt.show()
# -


