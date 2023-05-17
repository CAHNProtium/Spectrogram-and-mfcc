#Autor: Carlos Alberto Hernández Nava
#ASVspoof2017 spectrograms MFCC 2D

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
from sklearn.utils import class_weight
from scipy.special import expit as sigmoid
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import make_scorer, roc_curve
from sklearn.metrics import classification_report, confusion_matrix

x_train = []
data1 = pd.read_csv("/csv/mfccmatrixtraindev.csv", header=None)
y_train = data1.iloc[:, 0].to_numpy()
data1 = data1.iloc[: , 1:]
for index, row in data1.iterrows(): 
  row = row.to_numpy()
  row = row.reshape([298,13,1])
  x_train.append(row)
x_train = np.array(x_train, dtype=np.float32)

x_dev = []
data1 = pd.read_csv("/csv/mfccmatrixdev.csv", header=None)
y_dev = data1.iloc[:, 0].to_numpy()
data1 = data1.iloc[: , 1:]
for index, row in data1.iterrows(): 
  row = row.to_numpy()
  row = row.reshape([298,13,1])
  x_dev.append(row)
x_dev = np.array(x_dev, dtype=np.float32)

x_eval = []
data1 = pd.read_csv("/csv/mfccmatrixeval.csv", header=None)
y_eval = data1.iloc[:, 0].to_numpy()
data1 = data1.iloc[: , 1:]
for index, row in data1.iterrows(): 
  row = row.to_numpy()
  row = row.reshape([298,13,1])
  x_eval.append(row)
x_eval = np.array(x_eval, dtype=np.float32)

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
  #print("% EER = "+str(eerpercent))
  plt.show()
  return eerpercent

def modelo():
  
  input_layer = tf.keras.layers.Input(shape=(298, 13, 1, 1))
  time1=tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(filters= 36, kernel_size=(1,1), activation='relu'))(input_layer)
  batch1=tf.keras.layers.BatchNormalization()(time1)
  time2=tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(filters= 64, kernel_size=(1,1), activation='relu'))(batch1)
  time3=tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D(pool_size=(2,1),padding='same'))(time2)
  time4=tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(time3)
  time5=tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(24, activation='sigmoid'))(time4)
  time6=tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(24, activation='sigmoid'))(time5)
  flat1=tf.keras.layers.Flatten()(time6)
  out = tf.keras.layers.Dense(1, activation='sigmoid')(flat1)

  return input_layer, out


for x in range(5):
  #----------------------------------------TRAIN----------------------------------------
  input_layer, out = modelo()
  model = tf.keras.Model(input_layer, out)
  model.compile(optimizer='RMSprop',loss='binary_crossentropy', metrics=['accuracy'])
  print("\nEntrenamiento "+ str(x+1)+":")
  history = model.fit(x_train, y_train, validation_data=(x_dev, y_dev), batch_size=300, epochs = 30)
  model.save('/Model5-'+str(x+1)+'.h5')
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
