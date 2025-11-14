# -*- coding: utf-8 -*-
import pandas as pd
import glob
import os
import numpy as np
from matplotlib.pyplot import subplots
import funcions

#A l'excel, hem borrat el (mEq/L) i el .000 de les dates
dades1= pd.read_excel('C:/Users/UDM-AFIC/Desktop/Model NEWS/Ahora si que si/Dades/2024_1sem.xlsx')

"Excluim pacients"
dades1.drop(dades1[dades1['edat_alta'] < 18].index, inplace=True) 
serveis_descartats=['CIRURGIA PEDIATRICA', 'CIRURGIA PEDIATRICA HOSP', 'CURES PAL_LIATIVES GERIATRIA',
                    'DERMATOLOGIA', 'DROGODEPENDENCIES HOSP', 'GINECOLOGIA HOSP', 'HOSPITAL DE DIA ADOLESCENTS',
                    'HOSPITAL DE DIA TEA', 'HOSPITALITZACIÓ DOMICILIÀRIA H', 'MEDICINA INTERNA H.APTIMA',
                    'OBSTETRICIA HOSP', 'OFTALMOLOGIA HOSP', 'PEDIATRIA HOSP', 'PSIQUIATRIA HOSP ','RADIODIAGNOSTIC HOSPITALITZACIO',
                    'TRANSTORN ESPECTRE AUTISTA HOSPITALITZACIÓ', 'TRANSTORNS ALIMENTACIO', 'UROLOGIA H.APTIMA']
dades1=dades1[~dades1['serveialta'].isin(serveis_descartats)]
dades1.drop(dades1[dades1['estada'] < 2].index, inplace=True)
dades1.drop(['numerohc', 'c_diag_1'], axis=1)
del(serveis_descartats)

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
dades_filtrat1 = dades_fill1[dades_fill1['comptatge'] >= 3*dades_fill1['estada']].copy()
dades_filtrat1.drop(columns='comptatge', inplace=True)
dades_filtrat1= dades_filtrat1.sort_values(by=['numicu', 'data'])

"Creem mostres cada 1h hora"
dades_filtrat1['data'] = pd.to_datetime(dades_filtrat1['data'], errors='coerce')
dades_filtrat1['fecha_alta'] = pd.to_datetime(dades_filtrat1['fecha_alta'], errors='coerce')
data1= dades_filtrat1.groupby('numicu', group_keys=False).apply(funcions.resample_pacient)
# Omplim dades que son objects
df2_ultim = dades_filtrat1.drop_duplicates(subset='numicu', keep='last')
data1['fecha_alta'] = data1['numicu'].map(
    df2_ultim.set_index('numicu')['fecha_alta']
)

data1['edat_alta'] = data1['numicu'].map(
    df2_ultim.set_index('numicu')['edat_alta']
)

data1['serveialta'] = data1['numicu'].map(
    df2_ultim.set_index('numicu')['serveialta']
)

data1['estada'] = data1['numicu'].map(
    df2_ultim.set_index('numicu')['estada']
)

data1['tipus_assistencia'] = data1['numicu'].map(
    df2_ultim.set_index('numicu')['tipus_assistencia']
)

data1['resultat_alta'] = data1['numicu'].map(
    df2_ultim.set_index('numicu')['resultat_alta']
)

data1['descripcion'] = data1['numicu'].map(
    df2_ultim.set_index('numicu')['descripcion']
)

data1['sexo'] = data1['numicu'].map(
    df2_ultim.set_index('numicu')['sexo']
)

data1['antecedent_mpoc'] = data1['numicu'].map(
    df2_ultim.set_index('numicu')['antecedent_mpoc']
)

data1=data1[['numicu', 'data', 'fecha_alta', 'edat_alta', 'serveialta', 
                   'estada', 'tipus_assistencia', 'resultat_alta', 'descripcion', 'sexo', 'potassi',
                   'ph', 'lact', 'hb', 'oxigenoterapia', 'antecedent_mpoc', 'glasgow', 'ta_sist', 
                   'ta_diast', 'ta_mitja', 'fc', 'sato2', 't_axilar', 'f_respi']]

"Calculem el NEWS"
data1['NEWS'] = data1.apply(
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
data1['outcome'] = (data1['NEWS'] > 6).astype(int)
(data1['outcome'] == 1).sum()
del(dades1, dades_fill1, dades_filtrat1)