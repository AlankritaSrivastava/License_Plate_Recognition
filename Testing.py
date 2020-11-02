# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 17:15:35 2020

@author: AKS
"""

import glob
import cv2
import resize
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

model = load_model("model_classifier.h5")
dataset = pd.read_csv(r"c:\Users\alankrita.s\Desktop\ML_assignment\dataset.csv")

ls_dict= { '0':0, '1':1,  '2':2, '3':3, '4':4 ,'5':5 ,'6':6, '7':7, '8':8, '9':9,
          'A':10, 'B':11, 'C':12, 'D':13, 'E':14, 'F':15, 'G':16, 'H':17, 'I':18, 
          'J':19, 'K':20, 'L':21, 'M':22, 'N':23, 'O':24, 'P':25, 'Q':26, 'R':27, 
          'S':28, 'T':29, 'U':30, 'V':31, 'W':32, 'X':33, 'Y':34, 'Z':35}

def GetKey(val):
  for key,value in ls_dict.items():
    if val == value:
      return key

y_pred_text = []
y_text = []

test_path=r"c:\Users\alankrita.s\Desktop\ML_assignment\Image_Dataset\test"
for i in glob.glob(test_path+'\*.png'):
    m=[]
    cropped=[]
    y_pred_v=''
    resize.image_resize(i,300,300)
    img_file_path = i
    img = cv2.imread(img_file_path)

    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(imgray,(5,5),0)
    ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    contours, hierarchy = cv2.findContours(th3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

    cropped = []
    for cnt in contours:
        (x,y,w,h) = cv2.boundingRect(cnt)
        if (cv2.contourArea(cnt) > 100 and cv2.contourArea(cnt) < 2000):
            if (w >= 13 and w <= 30 and h >= 30 and h <= 60):
                thm=cv2.rectangle(img, (x,y), (x+w,y+h), (255, 0, 0), 2)
                c = th3[y:y+h,x:x+w]
                c = np.array(c)
                c = cv2.bitwise_not(c)
                c = resize.square(c)
                c = cv2.resize(c,(28,28), interpolation = cv2.INTER_AREA)
                cropped.append(c)
    dataset['Filename'] = dataset['Filename'].apply(lambda x: x.replace('/','_'))  
    val= dataset[dataset['Filename'] == i.split('\\')[-1]]
    filename= val['Filename'].values[0]
    text = val['Text'].values[0] 
    y_text.append(text)
    for img in cropped:
        img = np.resize(img, (28,28,1))
        im2arr = np.array(img)  
        im2arr = im2arr.reshape(1,28,28,1)
        y_pred = model.predict_classes(im2arr)
        y_pred = GetKey(y_pred[0])
        m.append(y_pred)
        y_pred_v = ''.join(v for v in m[::-1])
    y_pred_text.append(y_pred_v)

df_compare = pd.DataFrame({'Original_Text': y_text, 'Predicted_Text': y_pred_text})

def f(x):
    return 'True' if x['Original_Text'] == x['Predicted_Text'] else 'False'

df_compare['Answer'] = df_compare.apply(f, axis=1)

accuracy_score = (len(df_compare[df_compare['Answer'] == 'True'])/len(df_compare))*100
print(round(accuracy_score))

df_compare.to_csv("Result.csv")