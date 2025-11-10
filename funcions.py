# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 11:41:18 2025

@author: UDM-AFIC
"""
import pandas as pd
import glob
import os
import numpy as np

def omplir_dades(dades):
    copia = dades.copy()
    skip_cols = ['numicu', 'data', 'fecha_alta', 'edat_alta', 'serveialta', 'estada',
                 'tipus_assistencia', 'numerohc', 'resultat_alta', 'descripcion', 'sexo', 'ta_mitja']

    for col in copia.columns:
        if col in skip_cols:
            continue

        # Iterem per cada pacient
        for pacient_id, grup in copia.groupby('numicu'):
            n_nans = grup[col].notna().sum()
            sexe = grup['sexo'].iloc[0]
            mpoc= grup['antecedent_mpoc'].iloc[0]

            if n_nans == 0:
                # Totes les files d'aquest pacient són NaN → assignem valor normal segons sexe
                copia.loc[copia['numicu'] == pacient_id, col] = valors_normals(col, sexe)
            else:
                # Omple dins del pacient (endavant i enrere)
                copia.loc[copia['numicu'] == pacient_id, col] = grup[col].ffill().bfill()

    return copia

def resample_pacient(grup):
    grup = grup.set_index('data')
    grup_resampled = grup.resample('1H').mean(numeric_only=True)
    grup_resampled = grup_resampled.interpolate(method='time', limit_direction='both')
    grup_resampled['numicu'] = grup['numicu'].iloc[0]

    return grup_resampled.reset_index()

"NEWS"         
def valors_normals(parametre, sexe):
    if parametre=="ta_diast":
        return 75
    if parametre=="fc":
        return 75
    if parametre=="potassi":
        return 4.1
    if parametre=="lact":
        return 1.2
    if parametre=="f_respi":
        return 18
    if parametre=="sato2":
            return 98
    if parametre=="ta_sist":
        return 180
    if parametre=="t_axilar":
        return 35.6
    if parametre=="ph":
        return 7.4
    if parametre=="hb":
        if(sexe=="V"):
            return 150
        else:
            return 135
    if parametre=="glasgow":
        return 15
    if parametre=="antecedent_mpoc":
        return "NO"
    if parametre=="oxigenoterapia":
        return 0
    
def Resp_Rate(valor):
    if 12<=valor<=20:
        return 0
    elif 9<=valor<12:
        return 1
    elif 20<valor<=24:
        return 2
    elif valor<9 or valor>24:
        return 3
    
def Temperature(valor):
    if 36.1<=valor<=38:
        return 0
    elif 35.1<=valor<36.1 or 38<valor<=39:
        return 1
    elif valor>39:
        return 2
    elif valor<35.1:
        return 3
    
def Systolic_BP(valor):
    if 111<=valor<=219:
        return 0
    elif 101<=valor<111:
        return 1
    elif 91<=valor<101:
        return 2
    elif valor<91 or 219<valor:
        return 3
    
def Diastolic_BP(valor):
    if valor<80:
        return 0
    elif 80<=valor<=89:
        return 1
    elif 89<valor<=120:
        return 2
    elif valor>120:
        return 3
    
def HR(valor):
    if 51<=valor<=90:
        return 0
    elif 41<=valor<51 or 90<valor<=110:
        return 1
    elif 110<valor<=130:
        return 2
    elif valor<41 or 130<valor:
        return 3
    
def Sa_O2(valor, mpoc):
    if(mpoc=="NO"):
        if 96<=valor:
            return 0
        elif 94<=valor<96:
            return 1
        elif 92<=valor<94:
            return 2
        elif valor<92:
            return 3
    else:
        if 88<=valor:
            return 0
        if 86<=valor<88:
            return 1
        if 84<=valor<86:
            return 2
        if valor<84:
            return 3
    
def Oxigenoterapia(valor):
    if valor<1:
        return 0
    else: 
        return 2

def Glasgow(valor):
    if 13<=valor:
        return 0
    if 9<=valor<13:
        return 1
    if 3<=valor<9:
        return 3