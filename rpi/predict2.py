import numpy as np
from skimage.io import imread
from skimage import img_as_float
from skimage.color import rgb2gray
from tensorflow.python.keras.models import load_model
import datetime
import warnings
import shutil
import socket
import os
import time
import gps
from sudo import run_as_sudo
import server

def predict():
    model = load_model('/home/pi/Scream_Rec/[init2]cnn_scream_model.h5')
    # init2가 old보다 잘됨 
    print("model loaded")
    
    turn=1
    i = 1
    warnings.filterwarnings('ignore')
    while(turn<3):
        while(i <= 60):
            try:
                xhat = np.zeros(shape=(1, 77, 196))
                #xhat = np.zeros(shape=(1, 84, 200))
                temp = rgb2gray(img_as_float(imread('/home/pi/Scream_Rec/data'+str(turn)+'/' + str(i) + '.png')))
                xhat[0] = np.resize(temp, (77, 196))
                #xhat[0] = temp
                
                
                #print("hi")
                xhat = np.reshape(xhat, (xhat.shape[0], 77, 196, 1)).astype('float32')
                #xhat = np.reshape(xhat, (xhat.shape[0], 84, 200, 1)).astype('float32')
                #print(xhat.shape)


                yhat = model.predict(xhat, batch_size=20, verbose=2)
                #yhat_scream_prec = np.argmax(yhat_scream, axis=1).reshape(-1, 1)
                print(yhat)
                now = datetime.datetime.now()
                print(now)

                if(yhat[0][1] > 0.9):
                    scream_prec = 1
                    
                    original_path = '/home/pi/Scream_Rec/data'+str(turn)+'/' + str(i) + '.wav'
                    new_path = '/home/pi/Scream_Rec/backup/'+str(now.strftime('%Y-%m-%d %H:%M:%S'))+'.wav'
                    shutil.copy(original_path, new_path)
                    print("backup success")
                    # gps나중에 이거 켜야돼애애애애애
                    gps.gps()
                    print("gps success")
                    server.server()
                    
                else:
                    scream_prec = 0
                
                #print(yhat_scream)
            
                print('File '+str(i)+' Predict: ' + str(scream_prec))
                
                i += 1
                if(i > 60):
                    i = 1
                    if turn == 1:
                        turn = 2
                        shutil.rmtree(r'/home/pi/Scream_Rec/data1')
                        os.mkdir( "/home/pi/Scream_Rec/data1")
                        
                        
                    else:
                        turn = 1
                        shutil.rmtree(r'/home/pi/Scream_Rec/data2')
                        os.mkdir( "/home/pi/Scream_Rec/data2")
                
            except:
                time.sleep(5)
                continue
            

