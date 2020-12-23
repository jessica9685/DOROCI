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

# # 파라미터 설정

SAMPLING_RATE = 16000 # sampling rate. wav의 경우 16000Hz
D = 5 # 음성 시간 (초)
FRAME_LENGTH = 0.032 # frame_length = input_nfft / sr -> 20 ~ 40ms 단위여야 함
FRAME_STRIDE = 0.016 # 16ms마다 frequency 추출
N_MFCC = 40 # filter 개수 -> row 개수

# numpy size를 맞춰주기 위한 padding 함수
# 빈 자리는 0으로 채워줌
pad2D = lambda a, i: a[:, 0:i] if a.shape[1] > i else np.hstack((a, np.zeros((a.shape[0], i - a.shape[1]))))


# # MFCC 특징 추출

def get_mfcc(file_path, data_size):
    f_array = []
    
    for i in range(data_size):
        num = i + 1    
        a_name = file_path + str(num) + '.wav'

        # 음성파일 불러오기
        # y: 로드된 음성데이터, sr: sampling rate
        y, sr = librosa.load(a_name, sr=SAMPLING_RATE, duration=D)

        # fft: Fast Fourier Transform
        # 연속적인 시간 관점의 음성 신호를 주파수로 변환
        # n_fft: window 개수 = 16000 * 0.032 = 512
        input_nfft = int(round(sr*FRAME_LENGTH))

        # hop_length
        # 음성을 얼만큼 겹쳐서 자를 것인지?
        # (window_length - frame_stride)만큼 겹치는 것
        # 최소 50% 겹치게 분할해야 함
        input_stride = int(round(sr*FRAME_STRIDE))

        S = librosa.feature.mfcc(y=y, sr=SAMPLING_RATE, 
                                 n_fft=input_nfft, hop_length=input_stride,
                                 n_mfcc=N_MFCC)

        #print("Wav length: {}, MFCC_shape: {}".format(len(y)/sr, np.shape(S)))
        f_array.append(pad2D(S, 300))
    
    f_array = np.asarray(f_array) # list -> numpy 변환
    return f_array


# 라벨링 함수
def labeling(size, label):
    result = np.zeros(shape=(size))
    for i in range(size):
        result[i] = label
        
    return result


# # 데이터 전처리

from keras.utils.np_utils import to_categorical

# +
sos_1_path = 'D:\\졸프\\sound\\help\\구해줘\\convert+original\\구해줘_'
sos_2_path = 'D:\\졸프\\sound\\help\\그만\\convert+original\\그만_'
sos_3_path = 'D:\\졸프\\sound\\help\\도와주세요\\convert+original\\도와주세요_'
sos_4_path = 'D:\\졸프\\sound\\help\\도와줘\\convert+original\\도와줘_'
sos_5_path = 'D:\\졸프\\sound\\help\\사람살려\\convert+original\\사람살려_'
sos_6_path = 'D:\\졸프\\sound\\help\\살려주세요\\convert+original\\살려주세요_'
sos_7_path = 'D:\\졸프\\sound\\help\\살려줘\\convert+original\\살려줘_'

#human_path = 'D:\\졸프\\sound\\human\\human_'
#animal_path = 'D:\\졸프\\sound\\animal\\animal_'
#etc_path = 'D:\\졸프\\sound\\etc\\etc_'

sos_1 = get_mfcc(sos_1_path, 1194) # 구해줘
sos_2 = get_mfcc(sos_2_path, 1178) # 그만
sos_3 = get_mfcc(sos_3_path, 824) # 도와주세요
sos_4 = get_mfcc(sos_4_path, 966) # 도와줘
sos_5 = get_mfcc(sos_5_path, 955) # 사람살려
sos_6 = get_mfcc(sos_6_path, 1340) # 살려주세요
sos_7 = get_mfcc(sos_7_path, 1141) # 살려줘

#human = get_mfcc(human_path, 43)
#animal = get_mfcc(animal_path, 927)
#etc = get_mfcc(etc_path, 660)

print(sos_6.shape)
#print(human.shape)
#print(animal.shape)
#print(etc.shape)

# +
sos_1_id = labeling(sos_1.shape[0], 0)
sos_2_id = labeling(sos_2.shape[0], 1)
sos_3_id = labeling(sos_3.shape[0], 2)
sos_4_id = labeling(sos_4.shape[0], 3)
sos_5_id = labeling(sos_5.shape[0], 4)
sos_6_id = labeling(sos_6.shape[0], 5)
sos_7_id = labeling(sos_7.shape[0], 6)

#human_id = labeling(human.shape[0], 0)
#animal_id = labeling(animal.shape[0], 0)
#etc_id = labeling(etc.shape[0], 0)
print(sos_6_id.shape)

# +
from sklearn.model_selection import train_test_split

sos_1_train, sos_1_test, sos_1_id_train, sos_1_id_test = train_test_split(sos_1, sos_1_id, test_size=0.3, random_state=12)
sos_2_train, sos_2_test, sos_2_id_train, sos_2_id_test = train_test_split(sos_2, sos_2_id, test_size=0.3, random_state=12)
sos_3_train, sos_3_test, sos_3_id_train, sos_3_id_test = train_test_split(sos_3, sos_3_id, test_size=0.3, random_state=12)
sos_4_train, sos_4_test, sos_4_id_train, sos_4_id_test = train_test_split(sos_4, sos_4_id, test_size=0.3, random_state=12)
sos_5_train, sos_5_test, sos_5_id_train, sos_5_id_test = train_test_split(sos_5, sos_5_id, test_size=0.3, random_state=12)
sos_6_train, sos_6_test, sos_6_id_train, sos_6_id_test = train_test_split(sos_6, sos_6_id, test_size=0.3, random_state=12)
sos_7_train, sos_7_test, sos_7_id_train, sos_7_id_test = train_test_split(sos_7, sos_7_id, test_size=0.3, random_state=12)

#human_train, human_test, human_id_train, human_id_test = train_test_split(human, human_id, test_size=0.3, random_state=12)
#animal_train, animal_test, animal_id_train, animal_id_test = train_test_split(animal, animal_id, test_size=0.3, random_state=12)
#etc_train, etc_test, etc_id_train, etc_id_test = train_test_split(etc, etc_id, test_size=0.3, random_state=12)

x_train = np.concatenate((sos_1_train, sos_2_train, sos_3_train, sos_4_train, sos_5_train, sos_6_train, sos_7_train), axis=0)
x_test = np.concatenate((sos_1_test, sos_2_test, sos_3_test, sos_4_test, sos_5_test, sos_6_test, sos_7_test), axis=0)
y_train = np.concatenate((sos_1_id_train, sos_2_id_train, sos_3_id_train, sos_4_id_train, sos_5_id_train, sos_6_id_train, sos_7_id_train), axis=0)
y_test = np.concatenate((sos_1_id_test, sos_2_id_test, sos_3_id_test, sos_4_id_test, sos_5_id_test, sos_6_id_test, sos_7_id_test), axis=0)

#x_train = np.concatenate((sos_6_train, human_train, animal_train, etc_train), axis=0)
#x_test = np.concatenate((sos_6_test, human_test, animal_test, etc_test), axis=0)
#y_train = np.concatenate((sos_6_id_train, human_id_train, animal_id_train, etc_id_train), axis=0)
#y_test = np.concatenate((sos_6_id_test, human_id_test, animal_id_test, etc_id_test), axis=0)

# random shuffle
train_index = np.arange(x_train.shape[0])
np.random.shuffle(train_index)

test_index = np.arange(x_test.shape[0])
np.random.shuffle(test_index)

x_train = x_train[train_index]
y_train = y_train[train_index]

x_test = x_test[test_index]
y_test = y_test[test_index]

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

# +
x_train = x_train.reshape(x_train.shape[0], 40, 300, 1).astype('float32')
x_test = x_test.reshape(x_test.shape[0], 40, 300, 1).astype('float32')

#y_train = to_categorical(y_train, 7)
#y_test = to_categorical(y_test, 7)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
# -

# # 모델 학습

import os
import keras
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, BatchNormalization
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping

# +
MODEL_SAVE_PATH = './model/'

if not os.path.exists(MODEL_SAVE_PATH):
    os.mkdir(MODEL_SAVE_PATH)

model_path = MODEL_SAVE_PATH + 'sos-' + '{epoch:02d}-{val_loss:.4f}.hdf5'
cb_checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss', verbose=1, save_best_only=True)
cb_early_stopping = EarlyStopping(monitor='val_loss', patience=10)

# +
num_classes = 7
input_shape = x_train[0].shape

# raspberrypi 호환성을 위해 batch_size 작게 조정
model = Sequential()

# 1st conv layer
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape,
                    kernel_regularizer=keras.regularizers.l2(0.001)))
model.add(BatchNormalization())
model.add(MaxPooling2D((3, 3), strides=(2,2), padding='same'))

# 2nd conv layer
model.add(Conv2D(16, (3, 3), activation='relu',
                    kernel_regularizer=keras.regularizers.l2(0.001)))
model.add(BatchNormalization())
model.add(MaxPooling2D((3, 3), strides=(2,2), padding='same'))

# 3rd conv layer
model.add(Conv2D(16, (2, 2), activation='relu',
                    kernel_regularizer=keras.regularizers.l2(0.001)))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2), strides=(2,2), padding='same'))

# flatten output and feed into dense layer
model.add(Flatten())
model.add(Dense(32, activation='relu'))
Dropout(0.3)

# softmax output layer
model.add(Dense(num_classes, activation='softmax'))

# +
# 모델 컴파일 (학습 과정 설정)
model.compile(loss = 'sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

history = model.fit(x_train, y_train, validation_split=0.3,
                    epochs=100, batch_size=32, verbose=2,
                    callbacks=[cb_checkpoint, cb_early_stopping])
score = model.evaluate(x_test, y_test)

print('\nLoss: {:.4f}'.format(score[0]))
print('\nAccuracy: {:.4f}'.format(score[1]))
# -

from tensorflow.python.keras.models import load_model
model.save('dnn_sos_model.h5')
