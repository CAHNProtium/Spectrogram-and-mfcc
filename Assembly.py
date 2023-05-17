#Autors: Carlos Alberto Hernández Nava
#ASVspoof2017 Assembly for classification

import cv2, os, sys, csv
import numpy as np
import pandas as pd
import keras.models
import tensorflow
from keras.models import Sequential
from keras.layers import Activation, Dense
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import expit as sigmoid
from sklearn.metrics import classification_report, confusion_matrix
from pyllr.pav_rocch import PAV, ROCCH

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

x_eval = []
data1 = pd.read_excel("/results/Predictions.xlsx", header=None)
y_eval = data1.iloc[:, 0].to_numpy() #extract labels convert to numpy
data1 = data1.iloc[: , 1:] #extract images


for index, row in data1.iterrows(): 
  row = row.to_numpy()
  x_eval.append(row)
x_eval = np.array(x_eval, dtype=np.float32)



training_data = x_eval
target_data = y_eval

model = Sequential()
model.add(Dense(20, input_dim=5, activation='relu'))
model.add(Dense(15, input_dim=5, activation='relu'))
model.add(Dense(10, input_dim=5, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])

model.fit(training_data, target_data, epochs=1000)

# evaluate
results = model.evaluate(training_data, target_data)

print("\n%s: %.2f%%" % (model.metrics_names[1], results[1]*100))
print (model.predict(training_data).round())

model.save('/Finalmodel.h5')
