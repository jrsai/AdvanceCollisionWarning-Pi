import numpy as np
import cv2 
import matplotlib.pyplot as plt
#! /usr/bin/env python3
import struct
import enum
import serial
import time

import sys
import argparse
import paho.mqtt.client as mqtt


import logging
from time import sleep
from pprint import pprint



cap = cv2.VideoCapture(0)
frame_middle_x = 0
if cap.isOpened():
    frame_middle_x = cap.get(cv2.CAP_PROP_FRAME_WIDTH)/2
    #frame_middle_y = cap.get(cv2.CV_CAP_PROP_FRAME_HEIGHT)

face_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')



def printMessage(s):
    return ' '.join("{:02x}".format(c) for c in s)

class MessageType(enum.Enum):
    Text = 0
    Numeric = 1
    Logic = 2

def decodeMessage(s, msgType):
    payloadSize = struct.unpack_from('<H', s, 0)[0]
    
    remnant = 0
    if payloadSize < 5:       # includes the mailSize
        raise BufferError('Payload size is too small')
    
    a,b,c,d = struct.unpack_from('<4B', s, 2)
    if a != 1 or b != 0 or c != 0x81 or d != 0x9e:
        raise BufferError('Header is not correct.  Expecting 01 00 81 9e')
    
    mailSize = struct.unpack_from('<B', s, 6)[0]
    
    if payloadSize < (5 + mailSize):  # includes the valueSize
        raise BufferError('Payload size is too small')
    
    mailBytes = struct.unpack_from('<' + str(mailSize) + 's', s, 7)[0]
    mail = mailBytes.decode('ascii')[:-1]
    
    valueSize = struct.unpack_from('<H', s, 7 + mailSize)[0]
    if payloadSize < (7 + mailSize + valueSize):  # includes the valueSize
        raise BufferError('Payload size does not match the packet')

    if msgType == MessageType.Logic:
        if valueSize != 1:
            raise BufferError('Value size is not one byte required for Logic Type')
        valueBytes = struct.unpack_from('<B', s, 9 + mailSize)[0]
        value = True if valueBytes != 0 else False
    elif msgType == MessageType.Numeric:
        if valueSize != 4:
            raise BufferError('Value size is not four bytes required for Numeric Type')
        value = struct.unpack_from('<f', s, 9 + mailSize)[0]
    else:
        valueBytes = struct.unpack_from(payloadSize + 2)
        remnant = s[(payloadSize) + 2:]
        
    return (mail, value, remnant)

def encodeMessage(msgType, mail, value):
    mail = mail + '\x00'
    mailBytes = mail.encode('ascii') 
    mailSize = len(mailBytes)
    fmt = '<H4BB' + str(mailSize) + 'sH'
    
    if msgType == MessageType.Logic:
        valueSize = 1
        valueBytes = 1 if value is True else 0
        fmt += 'B'
    elif msgType == MessageType.Numeric:
        valueSize = 4
        valueBytes = float(value)
        fmt += 'f'
    else:
        value = value + '\x00'
        valueBytes = value.encode('ascii')
        valueSize = len(valueBytes)
        fmt += str(valueSize) + 's'
    
    payloadSize = 7 + mailSize + valueSize
    s = struct.pack(fmt, payloadSize, 0x01, 0x00, 0x81, 0x9e, mailSize, mailBytes, valueSize, valueBytes)
    return s


def detect_face(img):
    
  
    face_img = img.copy()
    direction = ""
  
    face_rects = face_cascade.detectMultiScale(face_img) 
    
    for i,(x,y,w,h) in enumerate(face_rects):
        if(i == 0):
            cv2.rectangle(face_img, (x,y), (x+w,y+h), (255,255,255), 10)
            x_center = int((x+(w/2)))
            y_center = int((y+(w/2)))
            cv2.circle(face_img, (x_center, y_center), 10, (255,255,255), -1)
            
            if(x_center > (frame_middle_x + 100)):
                direction = "left"
            elif(x_center < (frame_middle_x - 100)):
                direction = "right"
            else:
                direction = "stop"
            
    
    return face_img, direction
    
if __name__ == "__main__":

    EV3 = serial.Serial('/dev/rfcomm0')

    while True: 
        
        ret, frame = cap.read(0) 
         
        frame, direction = detect_face(frame)
        
        if (direction == "left"):
            s = encodeMessage(MessageType.Numeric, 'go', "1")
            EV3.write(s)
            print("left")
            #sleep(1000)
        elif(direction == "right"):
            s = encodeMessage(MessageType.Numeric, 'go', "2")
            EV3.write(s)
            #sleep(1000)
            print("right")
        else:
            s = encodeMessage(MessageType.Numeric, 'go', "0")
            EV3.write(s)
            #sleep(1000)
            print("stop")
     
        cv2.imshow('Video Face Detection', frame) 
     
        c = cv2.waitKey(1) 
        if c == 27: 
            break 
            
    EV3.close()        
    cap.release()
    cv2.destroyAllWindows()
    
        

