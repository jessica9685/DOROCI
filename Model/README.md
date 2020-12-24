# 비명소리 인식 모델 (CNN)

### CNN_MFCC 

학습데이터를 MFCC로 변환하여 CNN으로 학습

* `CNN_MFCC.ipynb`
* `CNN_MFCC.py`  



### CNN_mel

학습데이터를 Mel-Spectrogram으로 변환하여 CNN으로 학습

* 구조요청 음성 인식 모델
  * `CNN_SOS.ipynb`
  * `CNN_SOS.py`
  * 모델 파일: [Final]CNN_SOS_model.h5

* 비명소리 인식 모델
  * `CNN_Scream_or_Not.ipynb`
  * `CNN_Scream_or_Not.py`
  * 모델 파일: [init2]cnn_scream_model.h5

* 모델 검증용 코드
  * `predict_test.ipynb`
  * `predict_test.py`  



### preprocessing

* 학습데이터(wav)를 mel-spectrogram으로 변환
  * `wav_to_mel.ipynb`
  * `wav_to_mel.py`

* 학습데이터(wav)를 MFCC으로 변환
  * `wav_to_mfcc.ipynb`
  * `wav_to_mfcc.py`

