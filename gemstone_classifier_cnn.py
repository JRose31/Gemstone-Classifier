# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 20:37:50 2022

@author: jrose
"""
import os
import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


train_dir = r'C:\Users\jrose\Desktop\Gemstone Classifier\train'
test_dir = r'C:\Users\jrose\Desktop\Gemstone Classifier\test'

labels = []

for name in os.listdir(train_dir):
    labels.append(name)
    
print(f'Length: {len(labels)} >>> {labels}')
    

train_images = []
train_labels = []

#process images while getting index of test image to compare post processing
for label in labels:
    for filename in glob.glob(f'C:/Users/jrose/Desktop/Gemstone Classifier/train/{label}/*.jpg'):      
        im = cv2.imread(filename,cv2.COLOR_BGR2RGB) # BGR -> RGB
        img = cv2.resize(im, (100, 100))
        train_images.append(img)
        train_labels.append(label)
        
trainX = np.array(train_images)

labels_series = pd.Series(train_labels)
trainy = pd.get_dummies(labels_series)


print(trainX.shape)
print(trainy.shape)