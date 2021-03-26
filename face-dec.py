import cv2
import numpy as np

dec_face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
dec_eye = cv2.CascadeClassifier('haarcascade_eye.xml')

cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_writer = cv2.VideoWriter('face_detection_mp4.mp4', fourcc, 20.0, (480,640))

while True:
    ret, frame  = cap.read()

    

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = dec_face.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

        face_roi = frame[y:y+h, x:x+w]
        eyes = dec_eye.detectMultiScale(face_roi)

        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(face_roi, (ex,ey), (ex+ew,ey+eh), (255,0,0), 2)
    


    cv2.imshow("frame",frame)

    #print(frame.shape)
    video = cv2.resize(frame,(480,640))
    video_writer.write(video)
    
    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
video_writer.release()
cv2.destroyAllWindows()
