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
            sexe = grup['sexo'].iloc[0]   # agafem el sexe del pacient (suposem constant dins del grup)

            if n_nans == 0:
                # Totes les files d'aquest pacient són NaN → assignem valor normal segons sexe
                copia.loc[copia['numicu'] == pacient_id, col] = valors_normals(col, sexe)
            else:
                # Omple dins del pacient (endavant i enrere)
                copia.loc[copia['numicu'] == pacient_id, col] = grup[col].ffill().bfill()

    return copia

def resample_pacient(grup):
    # Establim la data com a índex temporal
    grup = grup.set_index('data')

    # Fem resampling cada 1 hora i calculem la mediana dins de la mateixa hora
    grup_resampled = grup.resample('1H').median(numeric_only=True)

    # Interpolem els valors NaN (mitjana entre el valor anterior i posterior)
    grup_resampled = grup_resampled.interpolate(method='time')

    # Recuperem l'id del pacient
    grup_resampled['numicu'] = grup['numicu'].iloc[0]

    # Tornem a posar la data com a columna normal
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
        return 0
    
def Resp_Rate(valor):
    if 12<=valor<=20:
        return 0
    elif 9<=valor<=11:
        return 1
    elif 21<=valor<=24:
        return 2
    elif valor<=8 or valor>=25:
        return 3
    
def Temperature(valor):
    if 36.1<=valor<=38:
        return 0
    elif 35.1<=valor<=36 or 38.1<=valor<=39:
        return 1
    elif valor>=39.1:
        return 2
    elif valor<=35:
        return 3
    
def Systolic_BP(valor):
    if 111<=valor<=219:
        return 0
    elif 101<=valor<=110:
        return 1
    elif 91<=valor<=100:
        return 2
    elif valor<=90 or 220<=valor:
        return 3
    
def Diastolic_BP(valor):
    if valor<80:
        return 0
    elif 80<=valor<=89:
        return 1
    elif 90<=valor<=120:
        return 2
    elif valor>120:
        return 3
    
def HR(valor):
    if 51<=valor<=90:
        return 0
    elif 41<=valor<=50 or 91<=valor<=110:
        return 1
    elif 111<=valor<=130:
        return 2
    elif valor<=40 or 131<=valor:
        return 3
    
def Sa_O2(valor):
    if 96<=valor:
        return 0
    elif 94<=valor<=95:
        return 1
    elif 92<=valor<=93:
        return 2
    elif valor<=91:
        return 3

def Glasgow(valor):
    if 13<=valor:
        return 0
    if 9<=valor<=12:
        return 1
    if 3<=valor<=8:
        return 3