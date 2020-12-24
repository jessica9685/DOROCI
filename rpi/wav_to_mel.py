#!/usr/bin/env python
# coding: utf-8

#import warnings
#warnings.filterwarnings("ignore")

import librosa
import scipy.io.wavfile
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import librosa.display
import time
import warnings


frame_length = 0.025 # 프레임을 일정 시간 단위로 쪼갬 (25ms)
frame_stride = 0.010 # 프레임이 겹치는 구간 (10ms)


def wav_to_mel():
    i = 1
    turn=1
    warnings.filterwarnings('ignore')
    while(turn<3):
        while(i <= 60):
            warnings.filterwarnings('ignore')
            try:
                audio_path = '/home/pi/Scream_Rec/data'+str(turn)+'/'
                mel_path = '/home/pi/Scream_Rec/data'+str(turn)+'/'
                a_name = audio_path + str(i) + '.wav'
                m_name = mel_path + str(i) + '.png'

                # mel-spectrogram
                # 음성 로드
                # y = 음성 데이터, sr = y의 sampling rate
                # sr은 default값 (22050)
                y, sr = librosa.load(a_name, duration=5)

                # wav_length = len(y)/sr
                # 시간 관점의 음성 데이터를 주파수 관점으로 변환
                input_nfft = int(round(sr*frame_length))
                input_stride = int(round(sr*frame_stride))

                # wav 파일을 melspectrogram으로 변환
                S = librosa.feature.melspectrogram(y=y, n_mels=40, n_fft=input_nfft, hop_length=input_stride)

                print("Wav length: {}, Mel_S shape:{}".format(len(y)/sr, np.shape(S)))

                plt.figure(figsize=(5, 2), frameon=False, dpi=50)
                librosa.display.specshow(librosa.power_to_db(S, ref=np.max), sr=sr, hop_length=input_stride)
                #librosa.display.specshow(librosa.logamplitude(S, ref_power=np.max), sr=sr, hop_length=input_stride)
                plt.savefig(m_name, format='png', bbox_inches='tight', pad_inches=0, facecolor='black')
                #plt.show()
                
                i += 1
                if(i > 60):
                    i = 1
                    num = 1
                    if turn == 1:
                        turn=2
                    else:
                        turn=1
            except:
                time.sleep(5)
                continue



