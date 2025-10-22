# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 19:27:28 2025

@author: laura
"""

import pandas as pd
import glob
import os
import numpy as np
from matplotlib.pyplot import subplots
import statsmodels.api as sm
from ISLP import load_data
from ISLP.models import (ModelSpec as MS,summarize)
from ISLP import confusion_table
from ISLP.models import contrast
from sklearn.discriminant_analysis import \
(LinearDiscriminantAnalysis as LDA ,
QuadraticDiscriminantAnalysis as QDA)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Ruta a la carpeta on estan els arxius .txt
carpeta = "C:/Users/laura/OneDrive/Escritorio/Uni/4t curs/Pràctiques en Empresa/Programació/Data Physionet"

# Busquem tots els arxius .txt dins la carpeta
arxius_txt = glob.glob(os.path.join(carpeta, "*.txt"))

# Llista per guardar tots els DataFrames
dataframes = []

for arxiu in arxius_txt:
    # Llegim l'arxiu
    df = pd.read_csv(arxiu, sep=",",header=1, names=["data", "paràmetre", "valor"], encoding="utf-8")
    
    # Extraer el nombre del archivo sin extensión para identificarlo
    df["id_pacient"] = os.path.splitext(os.path.basename(arxiu))[0]
    
    dataframes.append(df)

# Unim tots els DataFrames
df_total = pd.concat(dataframes, ignore_index=True)
df_total= df_total.reindex(['id_pacient', 'data', 'paràmetre', 'valor'], axis=1)

#Fiquem els paràmetres a les columnes
df_cols = df_total.pivot_table(
    index=["id_pacient", "data"],  # identifiers (una fila per pacient i hora)
    columns="paràmetre",          # cada paràmetre serà una columna
    values="valor",               # valors que es posaran a les columnes
).reset_index()

df_cols['Gender'] = df_cols.groupby('id_pacient')['Gender'].ffill()
df_cols['Age'] = df_cols.groupby('id_pacient')['Age'].ffill()

#Netegem
dades=df_cols.drop(columns=["Weight", "Height"])
del(arxiu, arxius_txt, carpeta, dataframes, df, df_cols, df_total)


