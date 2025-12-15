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
from itertools import chain

data1=pd.read_csv("C:/Users/UDM-AFIC/Desktop/Model NEWS/Ahora si que si/Codi/data1(1H).csv")
data2=pd.read_csv("C:/Users/UDM-AFIC/Desktop/Model NEWS/Ahora si que si/Codi/data2(1H).csv")
data3=pd.read_csv("C:/Users/UDM-AFIC/Desktop/Model NEWS/Ahora si que si/Codi/data3(1H).csv")
data4=pd.read_csv("C:/Users/UDM-AFIC/Desktop/Model NEWS/Ahora si que si/Codi/data4(1H).csv")

"Seleccionem diagnòstics"
#prefixos = tuple(f"J")
prefixos = tuple(f"I{n}" for n in range(30, 53))
# prefixos = tuple(chain(
#      (f"I{n}" for n in range(0, 30)),
#      (f"I{m}" for m in range(70, 100))
#  )) 
data1 = data1[data1["c_diag_1"].str.startswith(prefixos, na=False)]
(data1['outcome'] == 1).sum()
data2 = data2[data2["c_diag_1"].str.startswith(prefixos, na=False)]
(data2['outcome'] == 1).sum()
data3 = data3[data3["c_diag_1"].str.startswith(prefixos, na=False)]
(data3['outcome'] == 1).sum()
data4 = data4[data4["c_diag_1"].str.startswith(prefixos, na=False)]
(data4['outcome'] == 1).sum()
data = pd.concat([data1, data2, data3, data4], ignore_index=True)

print(sum(data['outcome']==1)/data.shape[0])

"Separem train set i data set"
scaler = StandardScaler()
features = [
    'potassi','hb','oxigenoterapia', 'glasgow','ta_sist', 
    'ta_mitja', 'fc', 'sato2', 't_axilar', 'f_respi'
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
    class_weight={0:1, 1:5},  # per compensar si hi ha poques observacions amb NEWS ≥6
    verbose=1
)

#joblib.dump(model, "versio533.pkl")
#model=joblib.load("versio302.pkl")

"Evaluem el model"
results = model.evaluate(X_data_test, y_data_test, verbose=0)
print(f"\nResultats LSTM - Predicció NEWS≥6:")
print(f"Exactitud: {results[1]:.3f} | AUC: {results[2]:.3f} | Precisió: {results[3]:.3f} | Recall: {results[4]:.3f}")

"Precision-Recall (Serveis descartats)"
probs = model.predict(X_data_test).ravel()
precision, recall, thresholds = precision_recall_curve(y_data_test, probs)
plt.plot(recall, precision)
plt.xlabel("Recall (sensibilitat)")
plt.ylabel("Precision (precisió)")
plt.title("Precision-Recall Curve")
plt.grid(True)
plt.show()

thresholds = np.linspace(0, 1, 200)

f1_scores = []
for thr in thresholds:
    preds = (probs >= thr).astype(int)
    f1_scores.append(f1_score(y_data_test, preds))

best_thr_f1 = thresholds[np.argmax(f1_scores)]

final_predictions = (probs >= best_thr_f1).astype(int)
cm = confusion_matrix(y_data_test, final_predictions)
TN, FP, FN, TP= cm.ravel()
sensibilitat = TP / (TP + FN)
especificitat = TN / (TN + FP)
precisio=TP/(TP+FP)
auc_pr = auc(recall, precision)

print(f"AUC-PR: {auc_pr:.4f}")
print(f"Precisió: {precisio:.4f}")
print(f"Sensibilitat: {sensibilitat:.4f}") 
print(f"Especificitat: {especificitat:.4f}")

idx = np.where(recall < 0.80)[0][0]
precision_80 = precision[idx]
print(f"Precisió amb aprox. un 80% de sensibilitat: {precision_80:.2f}")