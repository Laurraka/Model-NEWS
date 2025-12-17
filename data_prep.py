# -*- coding: utf-8 -*-
import pandas as pd
import glob
import os
import numpy as np
from matplotlib.pyplot import subplots
import funcions

"Carreguem dades"
dades=pd.read_excel('C:/Users/UDM-AFIC/Desktop/Model NEWS/Ahora si que si/Dades/2025_resta.xlsx')

"Rangs"
rangs = {
    'potassi': (1, 10),
    'ph': (6.8, 7.8),
    'lact': (0, 20),
    'hb': (2, 23),
    'glasgow': (3, 15),
    'ta_sist': (50, 300),
    'ta_diast': (20, 170),
    'fc': (20, 300),
    'sato2': (50, 100),
    't_axilar': (20, 45),
    'f_respi': (2, 70)
}
dades = funcions.elimina_pacients_per_rangs(dades, rangs, 'numicu')

"Excluim pacients"
dades.drop(dades[dades['edat_alta'] < 18].index, inplace=True) 
serveis_descartats=['CIRURGIA PEDIATRICA', 'CIRURGIA PEDIATRICA HOSP', 'CURES PAL_LIATIVES GERIATRIA',
                    'DERMATOLOGIA', 'DROGODEPENDENCIES HOSP', 'GINECOLOGIA HOSP', 'HOSPITAL DE DIA ADOLESCENTS',
                    'HOSPITAL DE DIA TEA', 'HOSPITALITZACIÓ DOMICILIÀRIA H', 'MEDICINA INTERNA H.APTIMA',
                    'OBSTETRICIA HOSP', 'OFTALMOLOGIA HOSP', 'PEDIATRIA HOSP', 'PSIQUIATRIA HOSP ','RADIODIAGNOSTIC HOSPITALITZACIO',
                    'TRANSTORN ESPECTRE AUTISTA HOSPITALITZACIÓ', 'TRANSTORNS ALIMENTACIO', 'UROLOGIA H.APTIMA']
dades=dades[~dades['serveialta'].isin(serveis_descartats)]
dades.drop(dades[dades['estada'] < 2].index, inplace=True)
dades.drop(['numerohc'], axis=1)
del(serveis_descartats)

"Ajuntem mateixa data en una sola fila"
date_cols = [col for col in dades.columns if 'data' in col.lower()]
dades['data'] = dades[date_cols].bfill(axis=1).iloc[:, 0]
dades=dades.dropna(subset=['data'])
dades= dades.drop(columns=date_cols)
dades= dades.groupby('data', as_index=False).first()
dades= dades.sort_values(['numicu', 'data']).reset_index(drop=True)
dades=dades[['numicu', 'data', 'fecha_alta', 'edat_alta', 'serveialta', 'estada', 
       'tipus_assistencia', 'numerohc', 'resultat_alta', 'c_diag_1','descripcion', 
       'sexo', 'potassi', 'ph', 'lact', 'hb', 'oxigenoterapia', 
       'antecedent_mpoc', 'glasgow', 'ta_sist', 'ta_diast', 'ta_mitja', 'fc', 
       'sato2', 't_axilar', 'f_respi']]
del(date_cols)

"Omplim valors NaN"
dades_fill=funcions.omplir_dades(dades)
dades_fill['oxigenoterapia'] = dades_fill['oxigenoterapia'].map({'SI': 1, 'NO': 0})
dades_fill=dades_fill.drop(columns=['ta_mitja', 'numerohc'])
dades_fill['ta_mitja']=dades_fill['ta_sist']+(2*dades_fill['ta_diast']/3)

"Traiem els pacients que tenen menys mostres que dies hospitalitzats"
dades_fill['comptatge'] = dades_fill.groupby('numicu')['numicu'].transform('count')
dades_filtrat = dades_fill[dades_fill['comptatge'] >= 3*dades_fill['estada']].copy()
dades_filtrat.drop(columns='comptatge', inplace=True)
dades_filtrat= dades_filtrat.sort_values(by=['numicu', 'data'])

"Creem mostres cada 1h hora"
dades_filtrat['data'] = pd.to_datetime(dades_filtrat['data'], errors='coerce')
dades_filtrat['fecha_alta'] = pd.to_datetime(dades_filtrat['fecha_alta'], errors='coerce')
data= dades_filtrat.groupby('numicu', group_keys=False).apply(funcions.resample_pacient)
# Omplim dades que son objects
df2_ultim = dades_filtrat.drop_duplicates(subset='numicu', keep='last')
data['fecha_alta'] = data['numicu'].map(
    df2_ultim.set_index('numicu')['fecha_alta']
)

data['edat_alta'] = data['numicu'].map(
    df2_ultim.set_index('numicu')['edat_alta']
)

data['serveialta'] = data['numicu'].map(
    df2_ultim.set_index('numicu')['serveialta']
)

data['estada'] = data['numicu'].map(
    df2_ultim.set_index('numicu')['estada']
)

data['tipus_assistencia'] = data['numicu'].map(
    df2_ultim.set_index('numicu')['tipus_assistencia']
)

data['resultat_alta'] = data['numicu'].map(
    df2_ultim.set_index('numicu')['resultat_alta']
)

data['c_diag_1'] = data['numicu'].map(
    df2_ultim.set_index('numicu')['c_diag_1']
)

data['descripcion'] = data['numicu'].map(
    df2_ultim.set_index('numicu')['descripcion']
)

data['sexo'] = data['numicu'].map(
    df2_ultim.set_index('numicu')['sexo']
)

data['antecedent_mpoc'] = data['numicu'].map(
    df2_ultim.set_index('numicu')['antecedent_mpoc']
)

data=data[['numicu', 'data', 'fecha_alta', 'edat_alta', 'serveialta', 
                   'estada', 'tipus_assistencia', 'resultat_alta', 'c_diag_1', 'descripcion', 'sexo', 'potassi',
                   'ph', 'lact', 'hb', 'oxigenoterapia', 'antecedent_mpoc', 'glasgow', 'ta_sist', 
                   'ta_diast', 'ta_mitja', 'fc', 'sato2', 't_axilar', 'f_respi']]

"Calculem el NEWS"
data['NEWS'] = data.apply(
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
data['outcome'] = (data['NEWS'] > 6).astype(int)
(data['outcome'] == 1).sum()
del(dades, dades_fill, dades_filtrat)

"Anàlisi descriptiu"
descripcio=data.describe()
data.dtypes

"Guardem dades"
data.to_csv('data4(1H).csv', index=False)