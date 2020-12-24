# 도로시 (도와줘 로드위의 CCTV)

## Model

### 비명소리 인식 모델

* preprocessing
  * 학습데이터(wav)를 Mel-Spectrogram으로 변환
    * `wav_to_mel.ipynb`
    * `wav_to_mel.py`
    * 학습데이터: https://drive.google.com/drive/folders/1iWdshc6dJvUfGbl5_NXfNBAEkhmPl0Zp?usp=sharing

* CNN_mel
  * 비명소리 인식 CNN 모델
    * `CNN_Scream_or_Not.py`
    * `CNN_Scream_or_Not.ipynb`
    * 모델 파일: `[init2]cnn_scream_model.h5`  





## rpi

### 도로시 프로세스 (Raspberry Pi)

* 실행방법:   `python3 process.py`
  * `recording.py` : 실시간으로 음성 녹음
  * `wav_to_mel.py` : 녹음된 음성 파일을 Mel-Spectrogram으로 변환
  * `predict2.py` : 변환된 Mel-Spectrogram으로 비명소리 여부 인식
  * `sen_recognition.py` : 녹음된 음성 파일로 구조요청 음성 인식
  * `gps.py` : 비명소리 또는 구조요청 음성이 인식되었을 경우(True), 웹서버로 위치 정보 전송
  * `server.py` : 웹서버에서 "사이렌 울리기"를 수행할 경우, 서버로부터 신호를 받아 사이렌을 재생
  * `delete.py` : 주기적으로 음성 파일 삭제
  * backup 폴더: 비명소리 또는 구조요청 음성으로 인식된 음성파일 저장





## Web

라즈베리파이에서 구조요청 음성이나 비명소리를 인식하면 "도로시" 웹서버로 결과 전송

* Web Server: https://github.com/Yejin6911/GraduationProject
