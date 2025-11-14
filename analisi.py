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
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import shap
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from data_prep import data1

print(data1.head())
print(data1.info())

"Separem train set i data set"
scaler = StandardScaler()
features = [
    'potassi', 'ph', 'lact', 'hb', 'oxigenoterapia', 'glasgow', 'ta_sist', 
    'ta_diast', 'ta_mitja', 'fc', 'sato2', 't_axilar', 'f_respi'
]

pacients = data1['numicu'].unique()
cut = int(len(pacients) * 0.8)
train_pacients = pacients[:cut]
test_pacients  = pacients[cut:]
data_train = data1[data1['numicu'].isin(train_pacients)].reset_index(drop=True)
data_test  = data1[data1['numicu'].isin(test_pacients)].reset_index(drop=True)

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
    epochs=20 ,
    batch_size=64,
    class_weight={0:1, 1:4},  # per compensar si hi ha poques observacions amb NEWS ≥6
    verbose=1
)

results = model.evaluate(X_data_test, y_data_test, verbose=0)
print(f"\nResultats LSTM - Predicció NEWS≥6:")
print(f"Exactitud: {results[1]:.3f} | AUC: {results[2]:.3f} | Precisió: {results[3]:.3f} | Recall: {results[4]:.3f}")
#TOT A SACO: Exactitud: 0.933 | AUC: 0.943 | Precisió: 0.241 | Recall: 0.836
#ELIMINANT SERVEIS DESCARTATS MENYS PSIQUIATRIA: Exactitud: 0.927 | AUC: 0.957 | Precisió: 0.228 | Recall: 0.883
#ELIMINANT SERVEIS DESCARTATS: Exactitud: 0.929 | AUC: 0.969 | Precisió: 0.237 | Recall: 0.902

"Precision-Recall"
probs = model.predict(X_data_test).ravel()
precision, recall, thresholds = precision_recall_curve(y_data_test, probs)
plt.plot(recall, precision)
plt.xlabel("Recall (sensibilitat)")
plt.ylabel("Precision (precisió)")
plt.title("Precision-Recall Curve")
plt.grid(True)
plt.show()

thresholds = np.linspace(0, 1, 200)  # 200 llindars entre 0 i 1
precision_list = []
recall_list = []

for thr in thresholds:
    preds = (probs >= thr).astype(int)
    precision_list.append(precision_score(y_data_test, preds))
    recall_list.append(recall_score(y_data_test, preds))
    
idx_best_recall = np.argmax(recall_list)
thr_best_recall = thresholds[idx_best_recall]
print("Llindar amb recall màxim:", thr_best_recall, "→ Recall:", recall_list[idx_best_recall])

f1_scores = []
for thr in thresholds:
    preds = (probs >= thr).astype(int)
    f1_scores.append(f1_score(y_data_test, preds))

best_thr_f1 = thresholds[np.argmax(f1_scores)]
print(f"\n>>> MILLOR LLINDAR F1 = {best_thr_f1:.3f}")
print(f">>> F1-score = {np.argmax(f1_scores):.4f}")

final_predictions = (probs >= best_thr_f1).astype(int)
cm = confusion_matrix(y_data_test, final_predictions)
TN, FP, FN, TP = cm.ravel()
sensibilitat = TP / (TP + FN)
especificitat = TN / (TN + FP)

print(f"Sensibilitat: {sensibilitat:.4f}") #0.6303
print(f"Especificitat: {especificitat:.4f}") #0.9854
