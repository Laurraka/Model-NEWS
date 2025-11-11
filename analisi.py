# -*- coding: utf-8 -*-
import pandas as pd
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.backend_bases
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking
from tensorflow.keras.optimizers import Adam

from data_prep import data1

print(data1.head())
print(data1.info())
    
#Busquem correlació
#data1.columns
variables = ['edat_alta', 'potassi','ph', 'lact', 'hb', 'oxigenoterapia', 
             'glasgow', 'ta_sist', 'ta_diast', 'ta_mitja', 'fc', 'sato2', 't_axilar', 
             'f_respi', 'outcome']
corr_subset = data1[variables].corr()
print(corr_subset)
plt.figure(figsize=(10, 4))
sns.heatmap(corr_subset, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlació entre variables seleccionades')
plt.show()
del(variables)

#Separem train set i data set
outcome=data1.filter(["outcome"])
outcome=outcome.values
training_data_len=int(np.ceil(len(outcome)*0.8))

training_data=outcome[:training_data_len]

X_train, y_train= [], []

for i in range(24, len(training_data)):
    X_train.append(training_data[i-24:i,0])
    y_train.append(training_data[i,0])
  
X_train, y_train=np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

#Normalitzar dades

#Plotejar 24h anteriors a NEWS>6 
cols_to_drop = [
    'outcome', 'numicu', 'data', 'fecha_alta', 'estada', 'edat_alta', 'serveialta',
    'tipus_assistencia', 'resultat_alta', 'sexo', 'antecedent_mpoc', 'NEWS'
]

for idx in data1.index[data1['outcome'] == 1]:
    start = max(0, idx - 24)
    window = data1.iloc[start:idx]   

    if window.empty:
        continue

    plot_df = window.drop(columns=cols_to_drop, errors='ignore')
    if plot_df.shape[1] == 0:
        continue

    plt.figure(figsize=(10, 4))
    plot_df.plot(ax=plt.gca())
    plt.title(f"Evolució 24 mostres abans d'outcome=1 (índex {idx})")
    plt.xlabel('Temps / índex')
    plt.ylabel('Valor')
    plt.legend(loc='best')
    plt.show()
del(cols_to_drop, idx, plot_df, start, window)
