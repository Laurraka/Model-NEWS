# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 09:46:14 2025

@author: UDM-AFIC
"""
import pandas as pd

data1=pd.read_csv("C:/Users/UDM-AFIC/Desktop/Model NEWS/Ahora si que si/Codi/data1(1H).csv")
data2=pd.read_csv("C:/Users/UDM-AFIC/Desktop/Model NEWS/Ahora si que si/Codi/data2(1H).csv")
data3=pd.read_csv("C:/Users/UDM-AFIC/Desktop/Model NEWS/Ahora si que si/Codi/data3(1H).csv")
data4=pd.read_csv("C:/Users/UDM-AFIC/Desktop/Model NEWS/Ahora si que si/Codi/data4(1H).csv")

prefixos = tuple(f"K") 
#prefixos = tuple(f"J{n}" for n in range(10,19))
# prefixos = tuple(chain(
#      (f"I{n}" for n in range(0, 30)),
#      (f"I{m}" for m in range(70, 100))
#  )) 
data1 = data1[data1["c_diag_1"].str.startswith(prefixos, na=False)]
data2 = data2[data2["c_diag_1"].str.startswith(prefixos, na=False)]
data3 = data3[data3["c_diag_1"].str.startswith(prefixos, na=False)]
data4 = data4[data4["c_diag_1"].str.startswith(prefixos, na=False)]

data = pd.concat([data1, data2, data3, data4], ignore_index=True)

ranges = {
    "J00-J06": r"^J0[0-6]",
    "J10-J18": r"^J1[0-8]",
    "J20-J22": r"^J2[0-2]",
    "J30-J39": r"^J3[0-9]",
    "J40-J47": r"^J4[0-7]",
    "J60-J70": r"^(J6[0-9]|J70)",
    "J80-J84": r"^J8[0-4]",
    "J90-J99": r"^J9[0-9]"
}

resultats = {}

for nom, regex in ranges.items():
    proporcio = data['c_diag_1'].str.contains(regex, regex=True, na=False).mean()
    resultats[nom] = proporcio

df_resultats = pd.DataFrame.from_dict(resultats, orient='index', columns=['proporcio'])
print(df_resultats)
