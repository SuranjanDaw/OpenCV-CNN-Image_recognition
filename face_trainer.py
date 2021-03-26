import cv2
import numpy as np
import os
import random
import pickle

face_dec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
dir_path = 'dataset2'
dirc = os.listdir(dir_path)
print(dirc)
known_faces = []
lables = []
c = 0
labelConverter ={}

for directory in dirc:
    labelConverter[directory] = c
    c+=1
c = 0
for directory in dirc:
    files = os.listdir(dir_path+'/'+directory)
    c = c + 1
    if c > 5:
        break
    for f in files:
        if not f.endswith("jpg") and not f.endswith("jpeg") and not f.endswith("png"):
            continue
        print("Image Number {}".format(c))
        img = cv2.imread(dir_path+'/'+directory+'/'+f)
        try:
          img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        except:
          print('An exception occurred', f)
          cv2.imshow("asd", img)

        
        faces  = face_dec.detectMultiScale(img)
        for (x,y,w,h) in faces:
            
            
            face_roi = img[y:y+h, x:x+w]
            identity = directory

            known_faces.append(face_roi)
            lables.append(labelConverter[identity])


print(labelConverter)

np_lables = np.array(lables)
print(np_lables)
s_list = list(zip(known_faces, np_lables))
random.shuffle(s_list)

s_face , s_labels = zip(*s_list) 

s_face_pickle = open('face_pickel.pickle', 'wb')
pickle.dump(s_face, s_face_pickle)
s_labels_pickle = open('label_pickel.pickle', 'wb')
pickle.dump(s_labels, s_labels_pickle)




print(s_labels)

recognizer = cv2.face.LBPHFaceRecognizer_create()
s_labels = np.array(s_labels)
recognizer.train(s_face,s_labels)

recognizer.write('trainer.yml')




