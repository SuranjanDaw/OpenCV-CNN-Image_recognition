import cv2
import numpy as np
import os

face_dec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cam = cv2.VideoCapture(0)
c = 0
count = 0
while True:

    count = 0  
    ret, frame = cam.read()
    if count % 500 == 0:
        name = 'myface/'+str(c)+'.png'
        cv2.imwrite(name,frame)
        c+=1
    cv2.imshow("Image", frame)
    count+=1
    if c == 100 :
        break
cam.release()
cv2.destroyAllWindows