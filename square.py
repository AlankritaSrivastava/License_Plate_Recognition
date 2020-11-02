# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 17:08:46 2020

@author: AKS
"""

import shutil
import os
import cv2
import numpy as np
import glob
import resize
import pandas as pd

def split_train_test(source1,test_path,train_path):
    files = os.listdir(source1)
    if not os.path.exists(source1+'\\test'):
        os.makedirs(source1+'\\test')
    if not os.path.exists(source1+'\\train'):
        os.makedirs(source1+'\\train')
    for f in files:
        if np.random.rand(1) < 0.2:
            shutil.move(source1 + '\\'+ f, test_path + '\\'+ f)
    for g in glob.glob(source1+'\\.png'):
        shutil.move(g, train_path + '\\'+ g.split('\\')[-1])
          

source1=r"C:\Users\AKS\Desktop\ML_assignment_IDfy\Image_Dataset"
test_path=r"C:\Users\AKS\Desktop\ML_assignment_IDfy\Image_Dataset\test"
train_path=r"C:\Users\AKS\Desktop\ML_assignment_IDfy\Image_Dataset\train"

split_train_test(source1,test_path,train_path)

dataset = pd.read_csv(r"C:\Users\AKS\Desktop\ML_assignment_IDfy\dataset.csv")

for i in glob.glob(test_path+'\*.png'):
    resize.image_resize(i,300,300)
    img_file_path = i
    img = cv2.imread(img_file_path)

    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(imgray,(5,5),0)
    ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cv2.imwrite("th1.png",th3)

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

    PATH_SAVED=os.path.join(train_path,'Training\\')
    dataset['Filename'] = dataset['Filename'].apply(lambda x: x.replace('/','_'))
  
    val= dataset[dataset['Filename'] == i.split('\\')[-1]]
    filename= val['Filename'].values[0]
    text = val['Text'].values[0] 
    text = text[::-1]
    s=0
    if len(text) == len(cropped):
        for i,j in zip(text,cropped):
            PATH_SAVED = os.path.join(PATH_SAVED, i)
            if not os.path.exists(PATH_SAVED):
                os.makedirs(PATH_SAVED)
            os.chdir(PATH_SAVED)
            cv2.imwrite(filename + '_' + text + '_' + str(s) + '.png', j)
            s+=1
            PATH_SAVED = os.path.join(train_path,'Training\\')
    os.chdir(r"C:\Users\AKS\Desktop\ML_assignment_IDfy")