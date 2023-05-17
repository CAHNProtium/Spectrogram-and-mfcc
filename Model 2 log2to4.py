#Autor: Carlos Alberto Hernández Nava
#ASVspoof2017 spectrograms Log 2 to Log 4

import cv2, sys, os, gc
import librosa
import numpy as np
import pandas as pd
import seaborn as sns
import openpyxl as pxl
import tensorflow as tf
from tensorflow import keras
import IPython.display as ipd
from numpy.random import randn
from pandas import ExcelWriter
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from pyllr.pav_rocch import PAV, ROCCH
from scipy.interpolate import interp1d
from scipy.special import expit as sigmoid
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import make_scorer, roc_curve
from sklearn.metrics import classification_report, confusion_matrix


x_train = []
data1 = pd.read_csv("/csv/spectraindevlog2a4.csv", header=None)
y_train = data1.iloc[:, 0].to_numpy()
data1 = data1.iloc[: , 1:] 
for index, row in data1.iterrows(): 
  row = row.to_numpy()
  row = row.reshape([100,100,3])
  x_train.append(row)
x_train = np.array(x_train, dtype=np.float32)

x_dev = []
data1 = pd.read_csv("/csv/specdevlog2a4.csv", header=None)
y_dev = data1.iloc[:, 0].to_numpy() 
data1 = data1.iloc[: , 1:] 
for index, row in data1.iterrows(): 
  row = row.to_numpy()
  row = row.reshape([100,100,3])
  x_dev.append(row)
x_dev = np.array(x_dev, dtype=np.float32)

x_eval = []
data1 = pd.read_csv("/csv/specevallog2a4.csv", header=None)
y_eval = data1.iloc[:, 0].to_numpy() 
data1 = data1.iloc[: , 1:] 
for index, row in data1.iterrows(): 
  row = row.to_numpy()
  row = row.reshape([100,100,3])
  x_eval.append(row)
x_eval = np.array(x_eval, dtype=np.float32)

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
  plt.show()
  return eerpercent


for x in range(1):
  #----------------------------------------TRAIN----------------------------------------
  input_layer, out = modelo()
  model = tf.keras.Model(input_layer, out)
  model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy']) #RMSprop
  print("\nEntrenamiento "+ str(x+1)+":")
  history = model.fit(x_train, y_train , validation_data=(x_dev, y_dev), batch_size=20, epochs = 10)
  model.save('/Model2-'+str(x+1)+'.h5')
  #----------------------------------------EVALUATION----------------------------------------
  print("\nEvaluacion "+ str(x+1)+":")
  results = model.evaluate(x_eval, y_eval, verbose = 1)
  #----------------------------------------PREDICT----------------------------------------
  prediction = model.predict(x_eval)
  predics = prediction.reshape(13306,)
  #----------------------------------------EER----------------------------------------
  eerpercent=calculate_eer(y_eval, predics)
  print('\n% EER:'+str(eerpercent))
  #----------------------------------------MATRIX----------------------------------------
  pre = np.rint(prediction)
  pre = pre.astype(int)
  pre = pre.flatten()
  cm = confusion_matrix(y_eval, pre)
  plt.title("Confusion Matrix 2017", fontsize =15)
  sns.heatmap(cm, annot=True, linewidths=.5, cmap="Blues", fmt=".5g")
  plt.show()
