#Autor: Carlos Alberto Hernández Nava
#ASVspoof2017 MFCC extractor an convert to csv

import cv2, os, sys, csv
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from sklearn import preprocessing
import seaborn as sns

pathwav1="/ASVspoof2017/traindev/"
pathwav2="/ASVspoof2017/dev/"
pathwav3="/ASVspoof2017/eval/"

dataclase1 = pd.read_excel("/traindev2017.xlsx", header=None ,names=["Nombre", "Clase"])
dataclase2 = pd.read_excel("/dev2017.xlsx", header=None ,names=["Nombre", "Clase"])
dataclase3 = pd.read_excel("/eval2017.xlsx", header=None ,names=["Nombre", "Clase"])

def convertirtraindev():
  data1=[]
  cont=0
  for i in dataclase4.index:
    if dataclase1["Clase"][i] == 'spoof':
      archivospo = dataclase1["Nombre"][i]
      data, rate = librosa.load(pathwav1+archivospo+'.wav', sr=16000)
      #extract MFCC features
      mfcc_ori = librosa.feature.mfcc(data,hop_length=512,n_fft=2048,n_mfcc=13)
      z_scaler = preprocessing.StandardScaler()
      mfcc_z = z_scaler.fit_transform(mfcc_ori)
      mfcc_tran = np.transpose(mfcc_z)
      if len(mfcc_tran)<341:
        difer=341-len(mfcc_tran)
        for x in range(0,difer):
          mfcc_tran = np.append(mfcc_tran, np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), axis=0)
      label = 1
      audio = np.array(mfcc_tran, dtype=np.float32)
      audio = np.append(label, audio)
      audio = pd.DataFrame(audio)
      audio = audio.transpose()
      if not os.path.isfile('/csv/matrixtraindev.csv'):
        audio.to_csv('/csv/mfccmatrixtraindev.csv', header = None, index = None)
      else:
        audio.to_csv('/csv/mfccmatrixtraindev.csv', mode="a" ,header = None, index = None) 
    else:
      archivobon = dataclase1["Nombre"][i]
      data, rate = librosa.load(pathwav1+archivobon+'.wav', sr=16000)
      #extract MFCC features
      mfcc_ori = librosa.feature.mfcc(data,hop_length=512,n_fft=2048,n_mfcc=13)
      z_scaler = preprocessing.StandardScaler()
      mfcc_z = z_scaler.fit_transform(mfcc_ori)
      mfcc_tran = np.transpose(mfcc_z)
      if len(mfcc_tran)<341:
        difer=341-len(mfcc_tran)
        for x in range(0,difer):
          mfcc_tran = np.append(mfcc_tran, np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), axis=0)
      label = 0
      audio = np.array(mfcc_tran, dtype=np.float32)
      audio = np.append(label, audio)
      audio = pd.DataFrame(audio)
      audio = audio.transpose()
      if not os.path.isfile('/csv/matrixtraindev.csv'):
        audio.to_csv('/csv/matrixtraindev.csv', header = None, index = None)
      else:
        audio.to_csv('/csv/matrixtraindev.csv', mode="a" ,header = None, index = None) 
    cont=cont+1
    print('\rConvertidos: '+str(cont), end='')
  return data1

def convertirdev():
  data2=[]
  cont=0
  for i in dataclase2.index:
    if dataclase2["Clase"][i] == 'spoof':
      archivospo = dataclase2["Nombre"][i]
      data, rate = librosa.load(pathwav2+archivospo+'.wav', sr=16000)
      #extract MFCC features
      mfcc_ori = librosa.feature.mfcc(data,hop_length=512,n_fft=2048,n_mfcc=13)
      z_scaler = preprocessing.StandardScaler()
      mfcc_z = z_scaler.fit_transform(mfcc_ori)
      mfcc_tran = np.transpose(mfcc_z)
      if len(mfcc_tran)<341:
        difer=341-len(mfcc_tran)
        for x in range(0,difer):
          mfcc_tran = np.append(mfcc_tran, np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), axis=0)
      label = 1
      audio = np.array(mfcc_tran, dtype=np.float32)
      audio = np.append(label, audio)
      audio = pd.DataFrame(audio)
      audio = audio.transpose()
      if not os.path.isfile('/csv/mfccmatrixdev.csv'):
        audio.to_csv('/csv/mfccmatrixdev.csv', header = None, index = None)
      else:
        audio.to_csv('/csv/mfccmatrixdev.csv', mode="a" ,header = None, index = None) 
    else:
      archivobon = dataclase2["Nombre"][i]
      data, rate = librosa.load(pathwav2+archivobon+'.wav', sr=16000)
      #extract MFCC features
      mfcc_ori = librosa.feature.mfcc(data,hop_length=512,n_fft=2048,n_mfcc=13)
      z_scaler = preprocessing.StandardScaler()
      mfcc_z = z_scaler.fit_transform(mfcc_ori)
      mfcc_tran = np.transpose(mfcc_z)
      if len(mfcc_tran)<341:
        difer=341-len(mfcc_tran)
        for x in range(0,difer):
          mfcc_tran = np.append(mfcc_tran, np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), axis=0)
      label = 0
      audio = np.array(mfcc_tran, dtype=np.float32)
      audio = np.append(label, audio)
      audio = pd.DataFrame(audio)
      audio = audio.transpose()
      if not os.path.isfile('/csv/mfccmatrixdev.csv'):
        audio.to_csv('/csv/mfccmatrixdev.csv', header = None, index = None)
      else:
        audio.to_csv('/csv/mfccmatrixdev.csv', mode="a" ,header = None, index = None) 
    cont=cont+1
    print('\rConvertidos: '+str(cont), end='')
  return data2

def convertireval():
  data3=[]
  cont=0
  for i in dataclase3.index:
    if dataclase3["Clase"][i] == 'spoof':
      archivospo = dataclase3["Nombre"][i]
      data, rate = librosa.load(pathwav3+archivospo+'.wav', sr=16000)
      #extract MFCC features
      mfcc_ori = librosa.feature.mfcc(data,hop_length=512,n_fft=2048,n_mfcc=13)
      z_scaler = preprocessing.StandardScaler()
      mfcc_z = z_scaler.fit_transform(mfcc_ori)
      mfcc_tran = np.transpose(mfcc_z)
      if len(mfcc_tran)<341:
        difer=341-len(mfcc_tran)
        for x in range(0,difer):
          mfcc_tran = np.append(mfcc_tran, np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), axis=0)
      label = 1
      audio = np.array(mfcc_tran, dtype=np.float32)
      audio = np.append(label, audio)
      audio = pd.DataFrame(audio)
      audio = audio.transpose()
      if not os.path.isfile('/csv/mfccmatrixeval.csv'):
        audio.to_csv('/csv/mfccmatrixeval.csv', header = None, index = None)
      else:
        audio.to_csv('/csv/mfccmatrixeval.csv', mode="a" ,header = None, index = None) 
    else:
      archivobon = dataclase3["Nombre"][i]
      data, rate = librosa.load(pathwav3+archivobon+'.wav', sr=16000)
      #extract MFCC features
      mfcc_ori = librosa.feature.mfcc(data,hop_length=512,n_fft=2048,n_mfcc=13)
      z_scaler = preprocessing.StandardScaler()
      mfcc_z = z_scaler.fit_transform(mfcc_ori)
      mfcc_tran = np.transpose(mfcc_z)
      if len(mfcc_tran)<341:
        difer=341-len(mfcc_tran)
        for x in range(0,difer):
          mfcc_tran = np.append(mfcc_tran, np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), axis=0)
      label = 0
      audio = np.array(mfcc_tran, dtype=np.float32)
      audio = np.append(label, audio)
      audio = pd.DataFrame(audio)
      audio = audio.transpose()
      if not os.path.isfile('/csv/mfccmatrixeval.csv'):
        audio.to_csv('/csv/mfccmatrixeval.csv', header = None, index = None)
      else:
        audio.to_csv('/csv/mfccmatrixeval.csv', mode="a" ,header = None, index = None) 
    cont=cont+1
    print('\rConvertidos: '+str(cont), end='')
  return data3


convertirtraindev()

convertirdev()

convertireval()
