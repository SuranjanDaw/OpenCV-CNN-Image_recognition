import cv2
import numpy as np
import pickle
from sklearn.model_selection import train_test_split

c = 0


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

faces = pickle.load(open('face_pickel.pickle','rb'))
lables = pickle.load(open('label_pickel.pickle','rb'))

train_face, test_face, train_lable, test_lable = train_test_split(faces, lables)


train_lable = np.array(train_lable)
print(train_lable)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(train_face, train_lable)
k = 0
print(len(test_lable))

for (face, lable) in zip(test_face, test_lable):
    k+=1
    id1, confidence = recognizer.predict(face)

    #name = list(label_map.keys())[list(label_map.values()).index(id1)]
    #real_name = list(label_map.keys())[list(label_map.values()).index(lable)]
    print(k)
    if lable == id1 :
        c += 1
percent = c / len(test_lable)
print('Percentage of match ', percent)