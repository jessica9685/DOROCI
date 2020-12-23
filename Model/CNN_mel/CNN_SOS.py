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

# # [데이터 불러오기]

# +
import os
import numpy as np
import matplotlib.pyplot as plt

from skimage.io import imread
from skimage import img_as_float
from skimage.color import rgb2gray

# +
# label == 1
_1_mel_path = 'D:\\졸프\\sound\\help\\구해줘\\mel-pi\\구해줘_'
_2_mel_path = 'D:\\졸프\\sound\\help\\그만\\mel-pi\\그만_'
_3_mel_path = 'D:\\졸프\\sound\\help\\도와주세요\\mel-pi\\도와주세요_'
_4_mel_path = 'D:\\졸프\\sound\\help\\도와줘\\mel-pi\\도와줘_'
_5_mel_path = 'D:\\졸프\\sound\\help\\사람살려\\mel-pi\\사람살려_'
_6_mel_path = 'D:\\졸프\\sound\\help\\살려주세요\\mel-pi\\살려주세요_'
_7_mel_path = 'D:\\졸프\\sound\\help\\살려줘\\mel-pi\\살려줘_'

# label == 0
crowd_mel_path = 'D:\\졸프\\sound\\crowd\\mel-pi\\crowd_'
etc_mel_path = 'D:\\졸프\\sound\\etc\\mel-pi\\etc_'
human_mel_path = 'D:\\졸프\\sound\\human\\mel-pi\\human_'
scream_mel_path = 'D:\\졸프\\sound\\scream\\mel-pi\\scream_'
# -

# # 1) 구해줘

# +
# 이미지 크기 = 84 * 200
_1 = np.zeros(shape=(1194, 84, 200))
_1_id = np.zeros(shape=(1194))

for i in range(1194):
    num = i + 1

    mel_name = _1_mel_path + str(num) + '.png'
    _1[i] = rgb2gray(img_as_float(imread(mel_name)))
    _1_id[i] = 1 # 구해줘 = 21


fig = plt.figure(figsize=(5, 2))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.1, wspace=0.1)

j = 1
for i in range(8):
    ax = fig.add_subplot(4, 4, j, xticks=[], yticks=[])
    mel = _1[i, :, :]
    ax.imshow(mel, cmap='bone', interpolation='nearest')
    j += 1
# -

# # 2) 그만

# +
# 이미지 크기 = 84 * 200
_2 = np.zeros(shape=(1178, 84, 200))
_2_id = np.zeros(shape=(1178))

for i in range(1178):
    num = i + 1

    mel_name = _2_mel_path + str(num) + '.png'
    _2[i] = rgb2gray(img_as_float(imread(mel_name)))
    _2_id[i] = 1 # 그만 = 1


fig = plt.figure(figsize=(5, 2))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.1, wspace=0.1)

j = 1
for i in range(8):
    ax = fig.add_subplot(4, 4, j, xticks=[], yticks=[])
    mel = _2[i, :, :]
    ax.imshow(mel, cmap='bone', interpolation='nearest')
    j += 1
# -

# # 3) 도와주세요

# +
# 이미지 크기 = 84 * 200
_3 = np.zeros(shape=(824, 84, 200))
_3_id = np.zeros(shape=(824))

for i in range(824):
    num = i + 1

    mel_name = _3_mel_path + str(num) + '.png'
    _3[i] = rgb2gray(img_as_float(imread(mel_name)))
    _3_id[i] = 1 # 도와주세요 = 1


fig = plt.figure(figsize=(5, 2))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.1, wspace=0.1)

j = 1
for i in range(8):
    ax = fig.add_subplot(4, 4, j, xticks=[], yticks=[])
    mel = _3[i, :, :]
    ax.imshow(mel, cmap='bone', interpolation='nearest')
    j += 1
# -

# # 4) 도와줘

# +
# 이미지 크기 = 84 * 200
_4 = np.zeros(shape=(966, 84, 200))
_4_id = np.zeros(shape=(966))

for i in range(966):
    num = i + 1

    mel_name = _4_mel_path + str(num) + '.png'
    _4[i] = rgb2gray(img_as_float(imread(mel_name)))
    _4_id[i] = 1 # 도와줘 = 1


fig = plt.figure(figsize=(5, 2))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.1, wspace=0.1)

j = 1
for i in range(8):
    ax = fig.add_subplot(4, 4, j, xticks=[], yticks=[])
    mel = _4[i, :, :]
    ax.imshow(mel, cmap='bone', interpolation='nearest')
    j += 1
# -

# # 5) 사람살려

# +
# 이미지 크기 = 84 * 200
_5 = np.zeros(shape=(955, 84, 200))
_5_id = np.zeros(shape=(955))

for i in range(955):
    num = i + 1

    mel_name = _5_mel_path + str(num) + '.png'
    _5[i] = rgb2gray(img_as_float(imread(mel_name)))
    _5_id[i] = 1 # 사람살려 = 1


fig = plt.figure(figsize=(5, 2))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.1, wspace=0.1)

j = 1
for i in range(8):
    ax = fig.add_subplot(4, 4, j, xticks=[], yticks=[])
    mel = _5[i, :, :]
    ax.imshow(mel, cmap='bone', interpolation='nearest')
    j += 1
# -

# # 6) 살려주세요

# +
# 이미지 크기 = 84 * 200
_6 = np.zeros(shape=(1340, 84, 200))
_6_id = np.zeros(shape=(1340))

for i in range(1340):
    num = i + 1

    mel_name = _6_mel_path + str(num) + '.png'
    _6[i] = rgb2gray(img_as_float(imread(mel_name)))
    _6_id[i] = 1 # 살려주세요 = 1


fig = plt.figure(figsize=(5, 2))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.1, wspace=0.1)

j = 1
for i in range(8):
    ax = fig.add_subplot(4, 4, j, xticks=[], yticks=[])
    mel = _6[i, :, :]
    ax.imshow(mel, cmap='bone', interpolation='nearest')
    j += 1
# -

# # 7) 살려줘

# +
# 이미지 크기 = 77 * 196
_7 = np.zeros(shape=(1141, 84, 200))
_7_id = np.zeros(shape=(1141))

for i in range(1141):
    num = i + 1

    mel_name = _7_mel_path + str(num) + '.png'
    _7[i] = rgb2gray(img_as_float(imread(mel_name)))
    _7_id[i] = 1 # 살려줘 = 1


fig = plt.figure(figsize=(5, 2))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.1, wspace=0.1)

j = 1
for i in range(8):
    ax = fig.add_subplot(4, 4, j, xticks=[], yticks=[])
    mel = _7[i, :, :]
    ax.imshow(mel, cmap='bone', interpolation='nearest')
    j += 1
# -

# # 8) Crowd

# +
# 이미지 크기 = 84 * 200
crowds = np.zeros(shape=(1775, 84, 200))
crowds_id = np.zeros(shape=(1775))

for i in range(1775):
    num = i + 1

    mel_name = crowd_mel_path + str(num) + '.png'
    crowds[i] = rgb2gray(img_as_float(imread(mel_name)))
    crowds_id[i] = 0 # 구조 요청이 아니면 0

# sample 출력
fig = plt.figure(figsize=(5, 2))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.1, wspace=0.1)

j = 1
for i in range(8):
    ax = fig.add_subplot(4, 4, j, xticks=[], yticks=[])
    mel = crowds[i, :, :]
    ax.imshow(mel, cmap='bone', interpolation='nearest')
    j += 1
# -

# # 9) Scream

# +
# 이미지 크기 = 84 * 200
screams = np.zeros(shape=(2008, 84, 200))
screams_id = np.zeros(shape=(2008))

for i in range(2008):
    num = i + 1

    mel_name = scream_mel_path + str(num) + '.png'
    screams[i] = rgb2gray(img_as_float(imread(mel_name)))
    screams_id[i] = 0 # 비명소리는 false

# sample 출력
fig = plt.figure(figsize=(5, 2))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.1, wspace=0.1)

j = 1
for i in range(8):
    ax = fig.add_subplot(4, 4, j, xticks=[], yticks=[])
    mel = screams[i, :, :]
    ax.imshow(mel, cmap='bone', interpolation='nearest')
    j += 1
# -

# # 10) Etc

# +
# 이미지 크기 = 84 * 200
etcs = np.zeros(shape=(660, 84, 200))
etcs_id = np.zeros(shape=(660))

for i in range(660):
    num = i + 1

    mel_name = etc_mel_path + str(num) + '.png'
    etcs[i] = rgb2gray(img_as_float(imread(mel_name)))
    etcs_id[i] = 0 # 비명소리가 아니면 false

# sample 출력
fig = plt.figure(figsize=(5, 2))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.1, wspace=0.1)

j = 1
for i in range(8):
    ax = fig.add_subplot(4, 4, j, xticks=[], yticks=[])
    mel = etcs[i, :, :]
    ax.imshow(mel, cmap='bone', interpolation='nearest')
    j += 1
# -

# # 11) human

# +
# 이미지 크기 = 84 * 200
humans = np.zeros(shape=(43, 84, 200))
humans_id = np.zeros(shape=(43))

for i in range(43):
    num = i + 1

    mel_name = human_mel_path + str(num) + '.png'
    humans[i] = rgb2gray(img_as_float(imread(mel_name)))
    humans_id[i] = 0 # 비명소리가 아니면 false

# sample 출력
fig = plt.figure(figsize=(5, 2))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.1, wspace=0.1)

j = 1
for i in range(8):
    ax = fig.add_subplot(4, 4, j, xticks=[], yticks=[])
    mel = humans[i, :, :]
    ax.imshow(mel, cmap='bone', interpolation='nearest')
    j += 1
# -

# # [데이터 전처리]

import keras
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping

# +
# 일정 비율 랜덤
from sklearn.model_selection import train_test_split

# 구조요청
_1_train, _1_test, _1_id_train, _1_id_test = train_test_split(_1, _1_id, test_size=0.3, random_state=12)
_2_train, _2_test, _2_id_train, _2_id_test = train_test_split(_2, _2_id, test_size=0.3, random_state=12)
_3_train, _3_test, _3_id_train, _3_id_test = train_test_split(_3, _3_id, test_size=0.3, random_state=12)
_4_train, _4_test, _4_id_train, _4_id_test = train_test_split(_4, _4_id, test_size=0.3, random_state=12)
_5_train, _5_test, _5_id_train, _5_id_test = train_test_split(_5, _5_id, test_size=0.3, random_state=12)
_6_train, _6_test, _6_id_train, _6_id_test = train_test_split(_6, _6_id, test_size=0.3, random_state=12)
_7_train, _7_test, _7_id_train, _7_id_test = train_test_split(_7, _7_id, test_size=0.3, random_state=12)

# 군중소리
crowd_train, crowd_test, crowd_id_train, crowd_id_test = train_test_split(crowds, crowds_id, test_size=0.3, random_state=12)
scream_train, scream_test, scream_id_train, scream_id_test = train_test_split(screams, screams_id, test_size=0.3, random_state=12)
etc_train, etc_test, etc_id_train, etc_id_test = train_test_split(etcs, etcs_id, test_size=0.3, random_state=12)
human_train, human_test, human_id_train, human_id_test = train_test_split(humans, humans_id, test_size=0.3, random_state=12)

x_train = np.concatenate((scream_train, etc_train, human_train, crowd_train, _1_train, _2_train, _3_train, _4_train, _5_train, _6_train, _7_train), axis=0)
x_test = np.concatenate((scream_test, etc_test, human_test, crowd_test, _1_test, _2_test, _3_test, _4_test, _5_test, _6_test, _7_test), axis=0)
y_train = np.concatenate((scream_id_train, etc_id_train, human_id_train, crowd_id_train, _1_id_train, _2_id_train, _3_id_train, _4_id_train, _5_id_train, _6_id_train, _7_id_train), axis=0)
y_test = np.concatenate((scream_id_test, etc_id_test, human_id_test, crowd_id_test, crowd_id_test, _1_id_test, _2_id_test, _3_id_test, _4_id_test, _5_id_test, _6_id_test, _7_id_test),  axis=0)



# 군중소리 & 구조요청 소리 랜덤으로 shuffle
train_index = np.arange(x_train.shape[0])
np.random.shuffle(train_index)

test_index = np.arange(x_test.shape[0])
np.random.shuffle(test_index)

x_train = x_train[train_index]
y_train = y_train[train_index]

x_test = x_test[test_index]
y_test = y_test[test_index]


# train, test 데이터 출력
print(x_train)
print(y_train)
print(x_test)
print(y_test)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

# +
x_train = x_train.reshape(x_train.shape[0], 84, 200, 1).astype('float32')
x_test = x_test.reshape(x_test.shape[0], 84, 200, 1).astype('float32')

y_train = to_categorical(y_train, 2)
y_test = to_categorical(y_test, 2)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
# -

# # [모델 학습]

# +
MODEL_SAVE_PATH = './model/'

if not os.path.exists(MODEL_SAVE_PATH):
    os.mkdir(MODEL_SAVE_PATH)

model_path = MODEL_SAVE_PATH + 'sos-' + '{epoch:02d}-{val_loss:.4f}.hdf5'
cb_checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss', verbose=1, save_best_only=True)
cb_early_stopping = EarlyStopping(monitor='val_loss', patience=10)

# +
num_classes = 2 # SOS == 1, crowd == 0
input_shape = (84, 200, 1)

model = Sequential()
model.add(Conv2D(16, kernel_size=(3, 3), input_shape=input_shape, activation='relu'))
model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# +
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(x_train, y_train, validation_split=0.3,
                    epochs=100, batch_size=32, verbose=2,
                    callbacks=[cb_checkpoint, cb_early_stopping])

score = model.evaluate(x_test, y_test)

print('\nLoss: {:.4f}'.format(score[0]))
print('\nAccuracy: {:.4f}'.format(score[1]))

from tensorflow.python.keras.models import load_model
model.save('[Final]CNN_SOS_model.h5')
# -


