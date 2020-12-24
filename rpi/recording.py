import pyaudio
import wave
import time
import warnings
import os
warnings.filterwarnings('ignore')

CHUNK = 512
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 5

def recording():
    num = 1
    turn=1
    warnings.filterwarnings('ignore')
    while(turn<3):
        while(num <= 60):
            try:
                WAVE_OUTPUT_FILENAME = "/home/pi/Scream_Rec/data"+str(turn)+"/"+str(num)+".wav"

                p = pyaudio.PyAudio()

                stream = p.open(format=FORMAT,
                                channels=CHANNELS,
                                rate=RATE,
                                input=True,
                                frames_per_buffer=CHUNK)

                print("Start to record the audio.")

                frames = []

                for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                    data = stream.read(CHUNK)
                    frames.append(data)

                print("Recording is finished.")

                stream.stop_stream()
                stream.close()
                p.terminate()

                wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(p.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(frames))
                wf.close()
                
                num += 1
                
                if(num > 60):
                    num = 1
                    if turn == 1:
                        turn=2
                    else:
                        turn=1

                
            except:
                time.sleep(5)
                continue
            

