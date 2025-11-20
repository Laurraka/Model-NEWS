# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 10:28:53 2025

@author: UDM-AFIC
"""
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
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import shap
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import joblib

data1=pd.read_csv("C:/Users/UDM-AFIC/Desktop/Model NEWS/Ahora si que si/Codi/data1(1H).csv")
data2=pd.read_csv("C:/Users/UDM-AFIC/Desktop/Model NEWS/Ahora si que si/Codi/data2(1H).csv")
data3=pd.read_csv("C:/Users/UDM-AFIC/Desktop/Model NEWS/Ahora si que si/Codi/data3(1H).csv")

"Seleccionem diagnòstics"
prefixos = tuple(f"J{n}" for n in range(10, 19))
data1 = data1[data1["c_diag_1"].str.startswith(prefixos, na=False)]
(data1['outcome'] == 1).sum()
data2 = data2[data2["c_diag_1"].str.startswith(prefixos, na=False)]
(data2['outcome'] == 1).sum()
data3 = data3[data3["c_diag_1"].str.startswith(prefixos, na=False)]
(data3['outcome'] == 1).sum()
data = pd.concat([data1, data2, data3], ignore_index=True)

"Separem train set i data set"
scaler = StandardScaler()
features = [
    'potassi','ph','lact', 'hb','oxigenoterapia', 'glasgow', 'ta_sist', 
    'ta_diast', 'ta_mitja', 'fc', 'sato2', 't_axilar', 'f_respi'
]

pacients = data['numicu'].unique()
cut = int(len(pacients) * 0.8)
train_pacients = pacients[:cut]
test_pacients  = pacients[cut:]
data_train = data[data['numicu'].isin(train_pacients)].reset_index(drop=True)
data_test  = data[data['numicu'].isin(test_pacients)].reset_index(drop=True)

data_train[features] = scaler.fit_transform(data_train[features])
data_test[features] = scaler.fit_transform(data_test[features])

X_data_train, y_data_train=[], []
X_data_test, y_data_test=[], []

timesteps = 24

for pid, group in data_train.groupby('numicu'):
    data = group[features].values
    labels = group['outcome'].values
    for i in range(timesteps, len(group)):
        X_data_train.append(data[i-timesteps:i]) #Tots els pacients seguits
        y_data_train.append(labels[i])
        
for pid, group in data_test.groupby('numicu'):
    data = group[features].values
    labels = group['outcome'].values
    for i in range(timesteps, len(group)):
        X_data_test.append(data[i-timesteps:i]) #Tots els pacients seguits
        y_data_test.append(labels[i])
        
X_data_train = np.array(X_data_train)
y_data_train = np.array(y_data_train)
X_data_test = np.array(X_data_test)
y_data_test = np.array(y_data_test)

"Model LSTM"
model = Sequential([
    Masking(mask_value=0.0, input_shape=(timesteps, X_data_train.shape[2])),
    LSTM(128, return_sequences=True),
    Dropout(0.3),
    LSTM(64),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(1e-3),
              loss='binary_crossentropy',
              metrics=['accuracy', 'AUC', 'Precision', 'Recall'])

model.summary()

"Entrenament" 
history = model.fit(
    X_data_train, y_data_train,
    validation_split=0.1,
    epochs=30,
    batch_size=64,
    class_weight={0:1, 1:4},  # per compensar si hi ha poques observacions amb NEWS ≥6
    verbose=1
)

joblib.dump(model, "versio31.pkl")