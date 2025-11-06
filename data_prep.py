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
dades1[['data']].head()
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
dades_fill1=funcions.omplir_dades(dades1)
dades_fill1=dades_fill1.drop(columns=['ta_mitja', 'numerohc'])

"Traiem els pacients que tenen menys mostres que dies hospitalitzats"
dades_fill1['ta_mitja']=dades_fill1['ta_sist']+(2*dades_fill1['ta_diast']/3)
# Comptar quantes files té cada pacient
dades_fill1['comptatge'] = dades_fill1.groupby('numicu')['numicu'].transform('count')
# Filtrar els pacients que tinguin tantes files com el valor 'estada'
dades_filtrat1 = dades_fill1[dades_fill1['comptatge'] >= 3*dades_fill1['estada']].copy()
# Eliminar la columna auxiliar si vols
dades_filtrat1.drop(columns='comptatge', inplace=True)

"Creem mostres cada 1h hora"
dades_filtrat1['data'] = pd.to_datetime(dades_filtrat1['data'])
df_resampled1= (
    dades_filtrat1.groupby('numicu', group_keys=False)
      .apply(funcions.resample_pacient)
      .reset_index(drop=True)
)
df_resampled1=df_resampled1[['numicu','data','edat_alta','estada','potassi','ph','lact','hb',
                             'antecedent_mpoc','glasgow','ta_sist','ta_diast','fc','sato2',
                             't_axilar','f_respi','ta_mitja']]

"Calculem el NEWS"
#dades_fill1['NEWS'] = dades_fill1.apply(
    #lambda x: (
        #funcions.Resp_Rate(x["f_respi"]) +
        #funcions.Temperature(x["t_axilar"])+
        #funcions.Systolic_BP(x["ta_sist"])+
        #funcions.Diastolic_BP(x["ta_diast"])+
        #funcions.HR(x["fc"])+
        #funcions.Sa_O2(x["sato2"])
    #),
    #axis=1
#)

"Calculem l'outcome"
#dades_fill1['outcome'] = (dades_fill1['NEWS'] > 6).astype(int)
#(dades_fill1['outcome'] == 1).sum()


