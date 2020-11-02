# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 17:14:41 2020

@author: AKS
"""

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import np_utils
from PIL import Image
import numpy as np
import os

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1).astype('float32')

ls_dict= { '0':0, '1':1,  '2':2, '3':3, '4':4 ,'5':5 ,'6':6, '7':7, '8':8, '9':9,
          'A':10, 'B':11, 'C':12, 'D':13, 'E':14, 'F':15, 'G':16, 'H':17, 'I':18, 
          'J':19, 'K':20, 'L':21, 'M':22, 'N':23, 'O':24, 'P':25, 'Q':26, 'R':27, 
          'S':28, 'T':29, 'U':30, 'V':31, 'W':32, 'X':33, 'Y':34, 'Z':35}

def load_images_to_data(image_label, image_directory, features_data, label_data):
    list_of_files = os.listdir(image_directory)
    for file in list_of_files:
        image_file_name = os.path.join(image_directory, file)
        if ".png" in image_file_name:
            img = Image.open(image_file_name).convert("L")
            img = np.resize(img, (28,28,1))
            im2arr = np.array(img)
            im2arr = im2arr.reshape(1,28,28,1)
            features_data = np.append(features_data, im2arr, axis=0)
            img_label=ls_dict[image_label]
            label_data = np.append(label_data, [img_label], axis=0)
    return features_data, label_data

train_path = r"C:\Users\AKS\Desktop\ML_assignment_IDfy\Image_Dataset\train\Training"

lsdr = os.listdir(train_path)
for i in lsdr:
    X_train, y_train = load_images_to_data(i, os.path.join(train_path,i), X_train, y_train)

X_train/=255

input_shape = (28,28,1)
number_of_classes = 36


# create model
model = Sequential()
model.add(Conv2D(28, (3, 3), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(number_of_classes, activation='softmax'))

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

model.fit(x=X_train, y=y_train, epochs=10)

model.save("model_classifier.h5")