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
    # Treballem sobre una còpia per evitar efectes laterals inesperats
    df = dades.copy()

    skip_cols = ['client_id', 'data', 'age']

    for col in df.columns:
        if col in skip_cols:
            continue

    # Si tota la columna és NaN -> omplir amb 33
    if df[col].notna().sum() == 0:
        df[col] = 33
            continue

    df[col] = df[col].ffill()
    df[col] = df[col].bfill()

    return df

"NEWS"
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
    elif valor<=39.1:
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
