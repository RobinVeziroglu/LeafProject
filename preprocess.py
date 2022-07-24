# -*- coding: utf-8 -*-
"""
Created on Sat Jul 23 20:00:06 2022

@author: a403922
"""

import os
import cv2
import numpy as np
import tensorflow as tf
import keras
from tensorflow.keras import layers

path=r"C:\Users\A403922\Desktop\LeafImage"

def load_data(path: str, IMG_SIZE = 224)->list:
    lst=[]
    resize_and_rescale = tf.keras.Sequential([
      layers.Resizing(IMG_SIZE, IMG_SIZE),
      layers.Rescaling(1./255)
    ])
    for idx,k in enumerate(os.listdir(path)):
        classes=[]
        for item in os.listdir(path+f"\{k}"):
            img=resize_and_rescale(cv2.imread(path+f"\{k}\{item}"))
            classes.append(tf.image.rgb_to_grayscale(img))
        lst.append([classes,k])       
    return lst

@tf.function
def augmentation(imgs: list)->list:
     x=tf.image.random_flip_up_down(tf.image.random_flip_left_right(imgs))
     return tf.image.rot90(x)