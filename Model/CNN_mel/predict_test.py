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

# +
import numpy as np
from skimage.io import imread
from skimage import img_as_float
from skimage.color import rgb2gray
from keras.models import load_model
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

xhat = np.zeros(shape=(10, 77, 196))

for i in range(10):
    num = i + 1
    #xhat[i] = rgb2gray(img_as_float(imread('D:\\졸프\\sound\\test\\test_' + str(num) + '.png')))
    temp = rgb2gray(img_as_float(imread('D:\\졸프\\sound\\test\\test_' + str(num) + '.png')))
    xhat[i] = np.resize(temp, (77, 196))

xhat = xhat.reshape(xhat.shape[0], 77, 196, 1).astype('float32')
print(xhat.shape)

model = load_model('D:\\졸프\\CNN_mel\\[Before]cnn_scream_model.h5')
y_pred = model.predict(xhat, batch_size=20, verbose=2) # 확률값임
print(y_pred)
y_pred = np.argmax(y_pred, axis=1).reshape(-1, 1) # 정수 형태의 레이블로 변환


for j in range(10):
    print('Predict: ', str(y_pred[j]))
    
y_test = np.zeros(10)
print(y_test)
# y_test = np.argmax(y_test, axis=1).reshape(-1, 1)
# confusion matrix 결과 출력
print("----------------- Confusion Matrix -----------------\n")
print(confusion_matrix(y_test, y_pred))

# classification_report 결과 출력
print("\n\n--------------- Classification Report ---------------\n")
print(classification_report(y_test, y_pred))
# -


