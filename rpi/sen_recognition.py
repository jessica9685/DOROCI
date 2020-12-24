import speech_recognition as sr
import time
import datetime
import os, shutil
import serial
import pynmea2
import string
import gps
import socket
import server

def sen_recognition():
    i = 1
    turn=1
    #warnings.filterwarnings('ignore')
    r = sr.Recognizer()
    while(turn<3):
        while(i <= 60):
            print(i)
            try:
                flag = 0
                AUDIO_FILE = '/home/pi/Scream_Rec/data'+str(turn)+'/'
                
                a_name = AUDIO_FILE + str(i) + '.wav'
                
                with sr.AudioFile(a_name) as source:
                    audio = r.record(source)
                
                sen=r.recognize_google(audio, language='ko')

                if '살' in sen:
                    index=sen.index('살')
                    if(sen[index+1] == '려'):
                        flag = 1
                        print(sen)
                elif '도' in sen:
                    index=sen.index('도')
                    if(sen[index+1] == '와'):
                        flag = 1
                        print(sen)
                elif '구' in sen:
                    index=sen.index('구')
                    if(sen[index+1] == '해'):
                        flag = 1
                        print(sen)
                else:
                    print(sen)
                    print(0)

                if(flag == 1):
                    now = datetime.datetime.now()
                    original_path = '/home/pi/Scream_Rec/data'+str(turn)+'/' + str(i) + '.wav'
                    new_path = '/home/pi/Scream_Rec/backup/'+"sen_"+str(now.strftime('%Y-%m-%d %H:%M:%S'))+'.wav'
                    shutil.copy(original_path, new_path)
                    print("sen backup success")
                    # 이것도 켜야돼애애애애애 이거 봣으면 predict2도 켜얃돼애애
                    gps.gps()
                    print("gps success")
                    server.server()
                i += 1
                if(i > 60):
                    i = 1
                    num = 1
                    if turn == 1:
                        turn=2
                    else:
                        turn=1

            except sr.UnknownValueError:
                print("Google Speech Recognition could not understand audio")
                i += 1
                if(i > 60):
                    i = 1
                    num = 1
                    if turn == 1:
                        turn=2
                    else:
                        turn=1
                continue
            except sr.RequestError as e:
                print("Could not request results from Google Speech Recognition service; {0}".format(e))
                i += 1
                if(i > 60):
                    i = 1
                    num = 1
                    if turn == 1:
                        turn=2
                    else:
                        turn=1
                continue
            except:
                time.sleep(5)
                continue




