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
import scipy.io.wavfile
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import librosa.display

# +
frame_length = 0.025 # 프레임을 일정 시간 단위로 쪼갬 (25ms)
frame_stride = 0.010 # 프레임이 겹치는 구간 (10ms)

for i in range(2008):
    audio_path = 'D:\\졸프\\sound\\scream\\scream_'
    mel_path = 'D:\\졸프\\sound\\scream\\mel-spectrogram\\scream_'
    num = i + 1
    a_name = audio_path + str(num) + '.wav'
    m_name = mel_path + str(num) + '.png'

    # mel-spectrogram
    # 음성 로드
    # y = 음성 데이터, sr = y의 sampling rate
    # sr은 default값 (22050) -> 44100으로 바꿔봄
    y, sr = librosa.load(a_name, sr=44100, duration=5)

    # wav_length = len(y)/sr
    # 시간 관점의 음성 데이터를 주파수 관점으로 변환
    input_nfft = int(round(sr*frame_length))
    input_stride = int(round(sr*frame_stride))

    # wav 파일을 melspectrogram으로 변환
    S = librosa.feature.melspectrogram(y=y, n_mels=40, n_fft=input_nfft, hop_length=input_stride)

    print("Wav length: {}, Mel_S shape:{}".format(len(y)/sr, np.shape(S)))

    plt.figure(figsize=(5, 2), frameon=False, dpi=50)
    # librosa.display.specshow(librosa.power_to_db(S, ref=np.max), y_axis='mel', sr=sr, hop_length=input_stride, x_axis='time')
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max), sr=sr, hop_length=input_stride) # 0.8.0 ver.
    # librosa.display.specshow(librosa.logamplitude(S, ref_power=np.max), sr=sr, hop_length=input_stride) # 0.4.2 ver.
    
    
    plt.savefig(m_name, format='png', bbox_inches='tight', pad_inches=0, facecolor='black')
    plt.show()
# -


