#Autor: Carlos Alberto Hernández Nava
#ASVspoof2017 spectrograms 100 x 100 color

import cv2, sys, os, gc
import numpy as np
import pandas as pd
import seaborn as sns
import openpyxl as pxl
import tensorflow as tf
from tensorflow import keras
from numpy.random import randn
from pandas import ExcelWriter
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from pyllr.pav_rocch import PAV, ROCCH
from scipy.interpolate import interp1d
from scipy.special import expit as sigmoid
from sklearn.metrics import make_scorer, roc_curve
from sklearn.metrics import classification_report, confusion_matrix

def modelo(x):
  
  input_layer = tf.keras.layers.Input([x,x,3])
  conv1=tf.keras.layers.Conv2D(filters= 32, kernel_size=(5,5), padding='Same', activation='relu')(input_layer)
  batch1=tf.keras.layers.BatchNormalization()(conv1)
  pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(batch1)
  batch2=tf.keras.layers.BatchNormalization()(pool1)
  conv2=tf.keras.layers.Conv2D(filters= 64, kernel_size=(3,3), padding='Same', activation='relu')(batch2)  
  batch3=tf.keras.layers.BatchNormalization()(conv2)
  pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2))(batch3)
  batch4=tf.keras.layers.BatchNormalization()(pool2)
  conv3=tf.keras.layers.Conv2D(filters= 96, kernel_size=(3,3), padding='Same', activation='relu')(batch4)
  batch5=tf.keras.layers.BatchNormalization()(conv3)
  pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2))(batch5)
  batch6=tf.keras.layers.BatchNormalization()(pool3)
  conv4=tf.keras.layers.Conv2D(filters= 96, kernel_size=(3,3), padding='Same', activation='relu')(batch6)
  batch7=tf.keras.layers.BatchNormalization()(conv4)
  pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2))(batch7)
  batch8=tf.keras.layers.BatchNormalization()(pool4)
  fit1 = tf.keras.layers.Flatten()(batch8)
  dn1 = tf.keras.layers.Dense(512, activation='relu')(fit1)
  out = tf.keras.layers.Dense(1, activation='sigmoid')(dn1)

  return input_layer, out

def calculate_eer(y_true, y_score):
  pav = PAV(y_score,y_true)
  rocch = ROCCH(pav)
      
  fig, ax = plt.subplots(2, 2)
  sc, llr = pav.scores_vs_llrs()
  ax[0,0].plot(sc,llr)
  ax[0,0].grid()
  ax[0,0].set_title("PAV: score --> log LR")

  pmiss,pfa = rocch.Pmiss_Pfa()
  ax[0,1].plot(pfa,pmiss,label='rocch')
  ax[0,1].plot(np.array([0,1]),np.array([0,1]),label="Pmiss = Pfa")
  ax[0,1].grid()
  ax[0,1].set_title("ROC convex hull")
  ax[0,1].legend(loc='best', frameon=False)

  plo = np.linspace(-5,5,100)
  ax[1,0].plot(sigmoid(plo),rocch.Bayes_error_rate(plo),label='minDCF')
  ax[1,0].grid()
  ax[1,0].legend(loc='best', frameon=False)
  ax[1,0].set_xlabel("P(target)")
      
  ber, pmiss, pfa = rocch.Bayes_error_rate(plo,True)
  ax[1,1].plot(sigmoid(plo),ber,label='minDCF')
  ax[1,1].plot(sigmoid(plo),pmiss,label='Pmiss')
  ax[1,1].plot(sigmoid(plo),pfa,label='Pfa')
  ax[1,1].legend(loc='best', frameon=False)
  ax[1,1].grid()

  eer=rocch.EER()
  eerpercent=eer*100
  print("% EER = "+str(eerpercent))
  plt.show()
  return eerpercent

trainaccu=[]
devaccu=[]
evalaccu=[]
eers=[]
for x in range(10):
  #----------------------------------------TRAIN----------------------------------------
  input_layer, out = modelo(100)
  model = tf.keras.Model(input_layer, out)
  model.compile(optimizer='RMSprop', loss='binary_crossentropy', metrics=['accuracy']) #RMSprop
  print("\nEntrenamiento "+ str(x+1)+":")
  history = model.fit(x_train, y_train , validation_data=(x_dev, y_dev), batch_size=15, epochs = 5)
  model.save('/content/drive/MyDrive/2017/models/spectro/AR1baseviejo34a-'+str(x+1)+'.h5')
  #----------------------------------------EVALUATION----------------------------------------
  print("\nEvaluacion "+ str(x+1)+":")
  results = model.evaluate(x_eval, y_eval, verbose = 1)
  trainaccu.append(history.history['accuracy'][4])
  devaccu.append(history.history['val_accuracy'][4])
  evalaccu.append(results[1])
  #----------------------------------------PREDICT----------------------------------------
  prediction = model.predict(x_eval)
  predics = prediction.reshape(13306,)
  #----------------------------------------EER----------------------------------------
  eerpercent=calculate_eer(y_eval, predics)
  eers.append(eerpercent)
  print('\n% EER:'+str(eerpercent))
  #----------------------------------------EER----------------------------------------
  pre = np.rint(prediction)
  pre = pre.astype(int)
  pre = pre.flatten()
  cm = confusion_matrix(y_eval, pre)
  plt.title("Confusion Matrix 2017", fontsize =15)
  sns.heatmap(cm, annot=True, linewidths=.5, cmap="Blues", fmt=".5g")
  plt.show()