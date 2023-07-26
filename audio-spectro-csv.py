#Autor: Carlos Alberto Hernández Nava
#ASVspoof2017 Spectrogram extractor an convert to csv

import sys, csv, os, time, shutil, cv2
import pydub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import soundfile as sf
from PIL import Image
from pydub import AudioSegment
from os import remove
import librosa

#paths y vars
pathwav1="/ASVspoof2017-V2/traindev/"
pathwav2="/ASVspoof2017-V2/dev/"
pathwav3="/ASVspoof2017-V2/eval/"

path1="/log2a4/traindev/"
path2="/log2a4/dev/"
path3="/log2a4/eval/"

path4="/log3a4/traindev/"
path5="/log3a4/dev/"
path6="/log3a4/eval/"

path4="/spectro/traindev/"
path5="/spectro/dev/"
path6="/spectro/eval/"

dataclase1 = pd.read_excel("/traindev2017.xlsx", header=None ,names=["Nombre", "Clase"])
dataclase2 = pd.read_excel("/dev2017.xlsx", header=None ,names=["Nombre", "Clase"])
dataclase3 = pd.read_excel("/eval2017.xlsx", header=None ,names=["Nombre", "Clase"])

def convertirtraindev():
  cont=0
  for i in dataclase1.index:
    if dataclase1["Clase"][i] == 'spoof':
      archivospo = dataclase1["Nombre"][i]
      #spectrogram
      sig, fs = librosa.load(pathwav1+archivospo+'.wav', sr=16000, duration=3)
      Pxx, freqs, bins, im = plt.specgram(sig, Fs=fs)
      im.set_cmap('jet')
      plt.yscale('log') #remove this line to linear spectrogram
      yticks = [100,10000] #yticks line to erase to log3 to 4
      plt.yticks(yticks) #remove this line to log3 to log 4
      plt.axis("off")
      plt.savefig(path1+'spoof/'+archivospo+'.png', bbox_inches='tight', pad_inches = 0)
      plt.close()
      #cut, resize
      image = cv2.imread(path1+'spoof/'+archivospo+'.png')
      crop_img = image[30:217, 0:334]
      image = cv2.resize(crop_img, (100, 100))
      cv2.imwrite(path1+'spoof/'+archivospo+'.png',image)
    else:
      archivobon = dataclase1["Nombre"][i]
      #spectrogram
      sig, fs = librosa.load(pathwav1+archivobon+'.wav', sr=16000, duration=3)
      Pxx, freqs, bins, im = plt.specgram(sig, Fs=fs)
      im.set_cmap('jet')
      plt.yscale('log') #remove this line to linear spectrogram
      yticks = [100,10000] #yticks line to erase to log3 to 4
      plt.yticks(yticks) #remove this line to log3 to log 4
      plt.axis("off")
      plt.savefig(path1+'bonafide/'+archivobon+'.png', bbox_inches='tight', pad_inches = 0)
      plt.close()
      #cut, resize
      image = cv2.imread(path1+'bonafide/'+archivobon+'.png')
      crop_img = image[30:217, 0:334]
      image = cv2.resize(crop_img, (100, 100))
      cv2.imwrite(path1+'bonafide/'+archivobon+'.png',image)
    cont=cont+1
    print('\rConvertidos: '+str(cont), end='')
    
def convertirdev():
  #convertidor
  cont=0
  for i in dataclase2.index:
    if dataclase2["Clase"][i] == 'spoof':
      archivospo = dataclase2["Nombre"][i]
      #spectrogram
      sig, fs = librosa.load(pathwav2+archivospo+'.wav', sr=16000, duration=3)
      Pxx, freqs, bins, im = plt.specgram(sig, Fs=fs)
      im.set_cmap('jet')
      plt.yscale('log') #remove this line to linear spectrogram
      yticks = [100,10000] #yticks line to erase to log3 to 4
      plt.yticks(yticks) #remove this line to log3 to log 4
      plt.axis("off")
      plt.savefig(path2+'spoof/'+archivospo+'.png', bbox_inches='tight', pad_inches = 0)
      plt.close()
      #cut, resize
      image = cv2.imread(path2+'spoof/'+archivospo+'.png')
      crop_img = image[30:217, 0:334]
      image = cv2.resize(crop_img, (100, 100))
      cv2.imwrite(path2+'spoof/'+archivospo+'.png',image)
    else:
      archivobon = dataclase2["Nombre"][i]
      #spectrogram
      sig, fs = librosa.load(pathwav2+archivobon+'.wav', sr=16000, duration=3)
      Pxx, freqs, bins, im = plt.specgram(sig, Fs=fs)
      im.set_cmap('jet')
      plt.yscale('log') #remove this line to linear spectrogram
      yticks = [100,10000] #yticks line to erase to log3 to 4
      plt.yticks(yticks) #remove this line to log3 to log 4
      plt.axis("off")
      plt.savefig(path2+'bonafide/'+archivobon+'.png', bbox_inches='tight', pad_inches = 0)
      plt.close()
      #cut, resize
      image = cv2.imread(path2+'bonafide/'+archivobon+'.png')
      crop_img = image[30:217, 0:334]
      image = cv2.resize(crop_img, (100, 100))
      cv2.imwrite(path2+'bonafide/'+archivobon+'.png',image)
    cont=cont+1
    print('\rConvertidos: '+str(cont), end='')
    
def convertireval():
  cont=0
  for i in dataclase3.index:
    if dataclase3["Clase"][i] == 'spoof':
      archivospo = dataclase3["Nombre"][i]
      #spectrogram
      sig, fs = librosa.load(pathwav3+archivospo+'.wav', sr=16000, duration=3)
      Pxx, freqs, bins, im = plt.specgram(sig, Fs=fs)
      im.set_cmap('jet')
      plt.yscale('log') #remove this line to linear spectrogram
      yticks = [100,10000] #yticks line to erase to log3 to 4
      plt.yticks(yticks) #remove this line to log3 to log 4
      plt.axis("off")
      plt.savefig(path3+'spoof/'+archivospo+'.png', bbox_inches='tight', pad_inches = 0)
      plt.close()
      #cut, resize
      image = cv2.imread(path3+'spoof/'+archivospo+'.png')
      crop_img = image[30:217, 0:334]
      image = cv2.resize(crop_img, (100, 100))
      cv2.imwrite(path3+'spoof/'+archivospo+'.png',image)
      del image
      del crop_img
    else:
      archivobon = dataclase3["Nombre"][i]
      #spectrogram
      sig, fs = librosa.load(pathwav3+archivobon+'.wav', sr=16000, duration=3)
      Pxx, freqs, bins, im = plt.specgram(sig, Fs=fs)
      im.set_cmap('jet')
      plt.yscale('log') #remove this line to linear spectrogram
      yticks = [100,10000] #yticks line to erase to log3 to 4
      plt.yticks(yticks) #remove this line to log3 to log 4
      plt.axis("off")
      plt.savefig(path3+'bonafide/'+archivobon+'.png', bbox_inches='tight', pad_inches = 0)
      plt.close()
      #cut, resize
      image = cv2.imread(path3+'bonafide/'+archivobon+'.png')
      crop_img = image[30:217, 0:334]
      image = cv2.resize(crop_img, (100, 100))
      cv2.imwrite(path3+'bonafide/'+archivobon+'.png',image)
      del image
      del crop_img
    cont=cont+1
    print('\rConvertidos: '+str(cont), end='')
    
#IMPORTANT
#To convert to linear spectrograms erase yticks lines and the crop_img line, only resize to 100, 100, and change paths
#To convert log3 to log 4 spectrograms erase the yticks lines and change paths
    
  
#To convert the spectrograms to a csv file
#The same process is used to convert all spectrograms to csv for use in models, only need to change the paths

def imgtocsv():
  categories = ['bonafide','spoof']
  cont=0
  conttotal=0
  for category in categories:
    path = os.path.join(path3, category)
    label = categories.index(category)
    print('\n'+path)
    for image_name in os.listdir(path):
      image_path = os.path.join(path, image_name)
      image = cv2.imread(image_path)
      image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
      try:
        image = np.array(image, dtype=np.float32)
        image = image.flatten()
        image = np.append(label, image)
        image = pd.DataFrame(image)
        image = image.transpose()
        if not os.path.isfile('/csv/specevallog2a4.csv'):
          image.to_csv('/csv/specevallog2a4.csv', header = None, index = None)
        else:
          image.to_csv('/csv/specevallog2a4.csv', mode="a" ,header = None, index = None)      
      except Exception as e:
        pass
      cont=cont+1
      print('\rCargados: '+str(cont), end='')
    conttotal=conttotal+cont
    cont=0
  print('\nImagenes transformadas: '+str(conttotal), end='')
