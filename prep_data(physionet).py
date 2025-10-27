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

# Ruta a la carpeta on estan els arxius .txt
carpeta = "C:/Users/UDM-AFIC/Desktop/Model NEWS/Fold1"

# Busquem tots els arxius .txt dins la carpeta
arxius_txt = glob.glob(os.path.join(carpeta, "*.txt"))

# Llista per guardar tots els DataFrames
dataframes = []

for arxiu in arxius_txt:
    # Llegim l'arxiu
    df = pd.read_csv(arxiu, sep=",",header=1, names=["data", "paràmetre", "valor"], encoding="utf-8")
    
    # Extreure el nom de l'arxiu sense extensió
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

#Omplim NaNs
dades= dades.fillna({
    'ICUType': dades['ICUType'].ffill(),
    'MechVent': dades['MechVent'].fillna(0)
})


#Extraiem outcome
outcome = pd.read_csv("C:/Users/UDM-AFIC/Desktop/Model NEWS/Fold1_Outcomes.csv", sep=",",header=1, names=["id_pacient", "length_of_stay", "death"], encoding="utf-8")