# **Raspberry Pi 3 B+**

Distributor ID:	Raspbian
Description:	Raspbian GNU/Linux 10 (buster)
Release:	10
Codename:	buster



# Requirements

### Python 3.6 (pip3 install)

##### Google SpeechRecognition API

* 구조요청 음성 인식
  * google-auth==1.20.1
  * google-auth-oauthlib==0.4.1
  * google-pasta==0.2.0
  * SpeechRecognition==3.7.1



##### Keras & Tensorflow

* 비명소리 인식 모델을 불러오고, 모델을 통해 비명소리 여부를 인식
  * Keras==2.4.3
  * Keras-Applications==1.0.8
  * Keras-Preprocessing==1.1.2
  * tensorboard==2.0.2
  * tensorboard-plugin-wit==1.7.0
  * tensorflow==2.2.0
  * tensorflow-estimator==1.14.0
  * h5py==2.10.0



##### Librosa

* 녹음된 음성을 Mel-Spectrogram으로 변환 후 저장
  * librosa==0.8.0
  * llvmlite==0.34.0
  * numba==0.51.0



##### Other Package & Library

* GPS 모듈
  * pynmea2==1.15.0
* 녹음
  * PyAudio==0.2.11

* Python IDE
  * thonny==3.2.6
* 머신러닝 라이브러리
  * scikit-image==0.17.2
  * scikit-learn==0.23.2
  * scipy==1.2.2
  * matplotlib==3.0.2
  * numpy==1.16.2





### Raspbian Package (apt-get install)

* Librosa가 의존하는 패키지
  * cmake version - 3.18.1
  * llvm - 10.0.1



* 카메라 모듈
  * mjpg-streamer: https://github.com/jacksonliam/mjpg-streamer


