import serial
import pynmea2
import time
import string
import os
import json
import requests

#url = 'http://54.238.218.120:8080/Alert/'
url = 'http://8a8f72a9cece.ngrok.io/Alert/'

def gps():
    #while True:
    #    port = "/dev/ttyAMA0"
    #    os.system('sudo chmod 777 /dev/ttyAMA0')
    #    ser = serial.Serial(port, baudrate=9600, timeout=0.5)
    #    dataout = pynmea2.NMEAStreamReader()
    #    newdata = ser.readline()
    #    newdata = newdata.decode("utf-8")
        #print(newdata)

    #    if "$GPRMC" in newdata :
    #        newmsg = pynmea2.parse(newdata)
    #        lat = newmsg.latitude
    #        lng = newmsg.longitude
    #        gps = "Latitude = " + str(lat) + " and Longitude = " + str(lng)
    #        print(gps)
    #        break

    client = requests.session()
    client.get(url)
    print(client)
    cookies = client.cookies
    print(cookies)

    data = {
        'latitude': 37.5616185,
        'longitude': 126.9437005,
        #'latitude': lat,
        #'longitude': lng,
        #'csrfmiddlewaretoken': csrftoken,
    }

    data = json.dumps(data)

    headers = {'Content-Type': 'application/json' }
    r = requests.post(url, headers=headers, data=data)

