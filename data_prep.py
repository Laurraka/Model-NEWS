# -*- coding: utf-8 -*-
import pandas as pd
import glob
import os
import numpy as np
from matplotlib.pyplot import subplots
import funcions

dades1= pd.read_excel('C:/Users/UDM-AFIC/Desktop/Model NEWS/Ahora si que si/Dades/Versió mini.xlsx')

date_cols = [col for col in dades1.columns if 'data' in col.lower()]
dades1['data'] = dades1[date_cols].bfill(axis=1).iloc[:, 0]
dades1=dades1.dropna(subset=['data'])
dades1= dades1.drop(columns=date_cols)

dades1 = dades1.groupby('data', as_index=False).first()
dades1= dades1.sort_values(['numicu', 'data']).reset_index(drop=True)
dades1=dades1[['numicu', 'data', 'fecha_alta', 'edat_alta', 'serveialta', 'estada', 
       'tipus_assistencia', 'numerohc', 'resultat_alta', 'descripcion', 
       'sexo', 'potassi', 'ph', 'lact', 'hb', 'oxigenoterapia', 
       'antecedent_mpoc', 'glasgow', 'ta_sist', 'ta_diast', 'ta_mitja', 'fc', 
       'sato2', 't_axilar', 'f_respi']]
del(date_cols)

"Omplim valors NaN"
dades_fill1=funcions.omplir_dades(dades1)
dades_fill1['oxigenoterapia'] = dades_fill1['oxigenoterapia'].map({'SI': 1, 'NO': 0})
dades_fill1=dades_fill1.drop(columns=['ta_mitja', 'numerohc'])
dades_fill1['ta_mitja']=dades_fill1['ta_sist']+(2*dades_fill1['ta_diast']/3)

"Traiem els pacients que tenen menys mostres que dies hospitalitzats"
dades_fill1['comptatge'] = dades_fill1.groupby('numicu')['numicu'].transform('count')
dades_filtrat1 = dades_fill1[dades_fill1['comptatge'] >= 2*dades_fill1['estada']].copy()
dades_filtrat1.drop(columns='comptatge', inplace=True)
dades_filtrat1= dades_filtrat1.sort_values(by=['numicu', 'data'])

"Creem mostres cada 1h hora"
dades_filtrat1['data'] = pd.to_datetime(dades_filtrat1['data'], errors='coerce')
dades_filtrat1['fecha_alta'] = pd.to_datetime(dades_filtrat1['fecha_alta'], errors='coerce')
resample= dades_filtrat1.groupby('numicu', group_keys=False).apply(funcions.resample_pacient)
# Omplim dades que son objects
df2_ultim = dades_filtrat1.drop_duplicates(subset='numicu', keep='last')
resample['fecha_alta'] = resample['numicu'].map(
    df2_ultim.set_index('numicu')['fecha_alta']
)

resample['edat_alta'] = resample['numicu'].map(
    df2_ultim.set_index('numicu')['edat_alta']
)

resample['serveialta'] = resample['numicu'].map(
    df2_ultim.set_index('numicu')['serveialta']
)

resample['estada'] = resample['numicu'].map(
    df2_ultim.set_index('numicu')['estada']
)

resample['tipus_assistencia'] = resample['numicu'].map(
    df2_ultim.set_index('numicu')['tipus_assistencia']
)

resample['resultat_alta'] = resample['numicu'].map(
    df2_ultim.set_index('numicu')['resultat_alta']
)

resample['descripcion'] = resample['numicu'].map(
    df2_ultim.set_index('numicu')['descripcion']
)

resample['sexo'] = resample['numicu'].map(
    df2_ultim.set_index('numicu')['sexo']
)

resample['antecedent_mpoc'] = resample['numicu'].map(
    df2_ultim.set_index('numicu')['antecedent_mpoc']
)

resample=resample[['numicu', 'data', 'fecha_alta', 'edat_alta', 'serveialta', 
                   'estada', 'tipus_assistencia', 'resultat_alta', 'descripcion', 'sexo', 'potassi',
                   'ph', 'lact', 'hb', 'oxigenoterapia', 'antecedent_mpoc', 'glasgow', 'ta_sist', 
                   'ta_diast', 'ta_mitja', 'fc', 'sato2', 't_axilar', 'f_respi']]

"Calculem el NEWS"
resample['NEWS'] = resample.apply(
    lambda x: (
        funcions.Resp_Rate(x["f_respi"]) +
        funcions.Temperature(x["t_axilar"]) +
        funcions.Systolic_BP(x["ta_sist"])+
        funcions.Diastolic_BP(x["ta_diast"])+
        funcions.HR(x["fc"])+
        funcions.Sa_O2(x["sato2"], x["antecedent_mpoc"])+
        funcions.Oxigenoterapia(x["oxigenoterapia"])
    ),
    axis=1
)
del(df2_ultim)

"Calculem l'outcome"
resample['outcome'] = (resample['NEWS'] > 6).astype(int)
(resample['outcome'] == 1).sum()
