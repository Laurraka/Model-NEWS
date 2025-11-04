# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 12:05:33 2025

@author: UDM-AFIC
"""
import pandas as pd
import glob
import os
import numpy as np

def omplir_dades(dades):
    copia = dades.copy()
    skip_cols = ['id_pacient', 'data', 'Age', 'Gender']

    for col in copia.columns:
        if col in skip_cols:
            continue

        # Iterem per cada pacient
        for pacient_id, grup in copia.groupby('id_pacient'):
            n_nans = grup[col].notna().sum()

            if n_nans == 0:
                # Totes les files d'aquest pacient són NaN
                copia.loc[copia['id_pacient'] == pacient_id, col] = valors_normals(col)
            else:
                # Omple dins del pacient
                copia.loc[copia['id_pacient'] == pacient_id, col] = (
                    grup[col].ffill().bfill()
                )

    return copia

"NEWS"         
def valors_normals(parametre):
    if parametre=="DiasABP":
        return 75
    if parametre=="HR":
        return 75
    if parametre=="K":
        return 4.1
    if parametre=="Lactate":
        return 1.2
    if parametre=="RespRate":
        return 18
    if parametre=="SaO2":
        return 98
    if parametre=="SysABP":
        return 180
    if parametre=="Temp":
        return 35.6
    if parametre=="pH":
        return 7.4
    
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
    elif 101<=valor<110:
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