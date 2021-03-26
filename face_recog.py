import cv2
import numpy
import pickle


cam = cv2.VideoCapture(0)
face_dec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter("celeb1.avi", fourcc, 20.0, (600,600))



recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')
font= cv2.FONT_HERSHEY_SIMPLEX

label_map ={'Aamir_Khan': 0, 'Abhay_Deol': 1,
'Abhishek_Bachchan': 2, 'Aftab_Shivdasani': 3,
'Aishwarya_Rai': 4, 'Ajay_Devgn': 5,
'Akshay_Kumar': 6, 'Alia_Bhatt': 7,
'Amitabh_Bachchan': 8, 'Anil_Kapoor': 9,
'Anushka_Sharma': 10, 'Anushka_Shetty': 11,
'Arshad_Warsi': 12, 'Ayushmann_Khurrana': 13,
'Bhumi_Pednekar': 14, 'Bipasha_Basu': 15,
'Deepika_Padukone': 16, 'Disha_Patani': 17,
'Emraan_Hashmi': 18, 'Farhan_Akhtar': 19,
'Govinda': 20, 'Hrithik_Roshan': 21, 'Ileana_D_Cruz': 22}

while True:
    ret, frame = cam.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_dec.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:

        face_roi = gray[y:y+h, x:x+w]
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)

        id1, confidence = recognizer.predict(face_roi)

        cv2.putText(
                    frame, 
                    list(label_map.keys())[list(label_map.values()).index(id1)], 
                    (x+5,y-5), 
                    font, 
                    0.5, 
                    (0,0,255), 
                    2
                   )
        cv2.putText(
                    frame, 
                    str(confidence), 
                    (x+5,y+h-5), 
                    font, 
                    0.5, 
                    (0,0,255), 
                    2
                   )  

    cv2.imshow("Image", frame)
    video = cv2.resize(frame,(600,600))
    out.write(video)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

out.release()
cam.release()
cv2.destroyAllWindows()



