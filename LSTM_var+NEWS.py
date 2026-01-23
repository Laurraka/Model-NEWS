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

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import joblib
from itertools import chain

"""

En aquest fitxer generem un model LSTM mirant el valor (normalitzat) de cada variable que 
puntua al NEWS2. Carregar els arxius segons la mida de la finestra desitjada. Ajustar també 
les variables timesteps (hores anteriors que es miraran) i windowsize.
Al final de tot, s'imprimeix la confusion matrix que mesura si s'ha activat l'alerta en 
el moment que tocava per al milor threshold trobat. Tenir en compte que a la funció per a calcular 
la confusion matrix es miren també X valors posteriors, cosa que potser no és del tot correcte.
Modificar també el diagnòstic pel qual es vol estudiar el model.

"""

data1=pd.read_csv("C:/Users/UDM-AFIC/Desktop/Model NEWS/Ahora si que si/Codi/data1(1H).csv")
data2=pd.read_csv("C:/Users/UDM-AFIC/Desktop/Model NEWS/Ahora si que si/Codi/data2(1H).csv")
data3=pd.read_csv("C:/Users/UDM-AFIC/Desktop/Model NEWS/Ahora si que si/Codi/data3(1H).csv")
data4=pd.read_csv("C:/Users/UDM-AFIC/Desktop/Model NEWS/Ahora si que si/Codi/data4(1H).csv")

"Seleccionem diagnòstics"
prefixos = tuple(f"I")
#prefixos = tuple(f"I{n}" for n in range(60, 70))
# prefixos = tuple(chain( 
#       (f"I"),
#       (f"J"),
#       (f"C")
# )) 
data1 = data1[data1["c_diag_1"].str.startswith(prefixos, na=False)]
(data1['outcome'] == 1).sum()
data2 = data2[data2["c_diag_1"].str.startswith(prefixos, na=False)]
(data2['outcome'] == 1).sum()
data3 = data3[data3["c_diag_1"].str.startswith(prefixos, na=False)]
(data3['outcome'] == 1).sum()
data4 = data4[data4["c_diag_1"].str.startswith(prefixos, na=False)]
(data4['outcome'] == 1).sum()
data = pd.concat([data1, data2, data3, data4], ignore_index=True)

"Separem train set i data set"   
scaler = StandardScaler()
features = [
    'glasgow',
    'oxigenoterapia',
    'ta_sist', 
    'fc', 
    'sato2', 
    't_axilar', 
    'f_respi',
    'NEWS'
]

pacients = data['numicu'].unique()
cut = int(len(pacients) * 0.8)
train_pacients = pacients[:cut] 
test_pacients  = pacients[cut:]   
data_train = data[data['numicu'].isin(train_pacients)].reset_index(drop=True)
data_test  = data[data['numicu'].isin(test_pacients)].reset_index(drop=True)

features_to_scale = [col for col in features if col != 'NEWS']

data_train[features_to_scale] = scaler.fit_transform(
    data_train[features_to_scale]
)

data_test[features_to_scale] = scaler.fit_transform(
    data_test[features_to_scale]
)

X_data_train, y_data_train=[], []
X_data_test, y_data_test=[], []

timesteps = 12
windowsize = 1  

for pid, group in data_train.groupby('numicu'):
    data = group[features].values
    labels = group['outcome'].values
    for i in range(timesteps//windowsize, len(group)):
        X_data_train.append(data[i-timesteps//windowsize:i]) #Tots els pacients seguits
        y_data_train.append(
            np.array([pid, labels[i]])
        )
        
for pid, group in data_test.groupby('numicu'):
    data = group[features].values
    labels = group['outcome'].values
    for i in range(timesteps//windowsize, len(group)):
        X_data_test.append(data[i-timesteps//windowsize:i]) #Tots els pacients seguits
        y_data_test.append(
            np.array([pid, labels[i]])
        )
        
X_data_train = np.array(X_data_train)
y_data_train = np.array(y_data_train)
X_data_test = np.array(X_data_test)
y_data_test = np.array(y_data_test)

"Model LSTM"
model = Sequential([
    Masking(mask_value=0.0, input_shape=(timesteps//windowsize, X_data_train.shape[2])),
    LSTM(128, return_sequences=True),
    Dropout(0.3),
    LSTM(64),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

def focal_loss(alpha=1.0, gamma=2.0):
    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        bce = -(y_true * tf.math.log(y_pred) +
                (1 - y_true) * tf.math.log(1 - y_pred))
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        fl = alpha * tf.pow(1 - p_t, gamma) * bce
        return tf.reduce_mean(fl)
    return loss

model.compile(
    optimizer=Adam(1e-3),
    loss=focal_loss(alpha=1, gamma=1.5),
    metrics=[
        'accuracy',
        tf.keras.metrics.AUC(curve="PR", name="auc_pr"),
        tf.keras.metrics.Precision(),
        tf.keras.metrics.Recall()
    ]
)

model.summary()

"Entrenament" 
history = model.fit(
    X_data_train, y_data_train[:,1],
    validation_split=0.1,
    epochs=40,
    batch_size=64, 
    verbose=1
) 

#joblib.dump(model, "versio533.pkl")
#model=joblib.load("versio302.pkl")

"Evaluem el model"
results = model.evaluate(X_data_test, y_data_test[:,1], verbose=0)
print(f"\nResultats LSTM - Predicció NEWS≥6:")
print(f"Exactitud: {results[1]:.3f} | AUC: {results[2]:.3f} | Precisió: {results[3]:.3f} | Recall: {results[4]:.3f}")

"Precision-Recall (Serveis descartats)"
probs = model.predict(X_data_test).ravel()
probs2=probs.reshape(-1,1)
comparativa_numicu=np.concatenate((y_data_test,probs2), axis=1)
precision, recall, thresholds = precision_recall_curve(y_data_test[:,1], probs)
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
    f1_scores.append(f1_score(y_data_test[:,1], preds))

best_thr_f1 = thresholds[np.argmax(f1_scores)]

final_predictions = (probs >= best_thr_f1).astype(int)
cm = confusion_matrix(y_data_test[:,1], final_predictions)
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

def compute_cm_onset(threshold, window=5):
    thres=(probs>=threshold).astype(int)
    thres=thres.reshape(-1,1)
    conc=np.concatenate((comparativa_numicu,thres), axis=1)
    df = pd.DataFrame(conc, columns=['numicu', 'y_test', 'probs', 'thres'])
    TP, FP, FN, TN = 0, 0, 0, 0
    
    for numicu, group in df.groupby('numicu'):
        test_group=group['y_test'].values
        thres_group=group['thres'].values
        X, Y=[], []
    
        Y.append(0)
        for i in range(1, len(test_group)):
            if test_group[i]==1 and test_group[i-1]==0:
                Y.append(1)
            else:
                Y.append(0)
                
        X.append(0)
        for i in range(1, len(thres_group)):
            if thres_group[i]==1 and thres_group[i-1]==0:
                X.append(1)
            else:
                X.append(0)
                
        for i in range(0,len(Y)):
            if Y[i]==0:
                inici=max(0,i-7)
                if all(x == 0 for x in X[inici:i+1]):
                    TN += 1
                    
        for i in range(0,len(X)):
            if X[i]==1:
                if all(y == 0 for y in Y[i:i+window+1]) and Y[i-1]!=1:
                    FP += 1
        
        for i in range(0,len(Y)):
            if Y[i]==1:
                inici=max(0,i-7)
                if all(x == 0 for x in X[inici:i+1]):
                    FN += 1
                    
        for i in range(0,len(Y)):
            if Y[i]==1:
                inici=max(0,i-7)
                if any(x == 1 for x in X[inici:i+1]):
                    TP += 1
                    
        del(X,Y)
    
    return TP, FP, FN, TN

thresholds = np.linspace(0.05, 0.95, 50)

best_f1 = 0
best_threshold = None
best_cm = None

for th in thresholds:
    TP, FP, FN, TN = compute_cm_onset(th)

    if TP + FP != 0 and TP + FN != 0:

        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        
        if precision!=0 and recall!=0:
    
            f1 = 2 * precision * recall / (precision + recall)
        
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = th
                best_cm = (TN, FP, FN, TP)

print("Millor threshold:", best_threshold)
print("Millor F1:", best_f1)

TN, FP, FN, TP = best_cm
cm = np.array([[TN, FP],
               [FN, TP]])

print(cm)