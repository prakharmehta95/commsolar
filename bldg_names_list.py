# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 14:48:57 2019

@author: iA
"""

#%%

import pandas as pd

zones_all = pd.read_excel(r'C:\Users\iA\Dropbox\Com_Paper\05_Data\01_CEA_Disaggregated\02_Buildings_Info\Zones_to_BldgEGIDs.xlsx', sheetname='ZoneBldg_Names')

zones_cea = pd.read_excel(r'C:\Users\iA\Dropbox\Com_Paper\05_Data\01_CEA_Disaggregated\02_Buildings_Info\Zones_1637_Info.xlsx')

zones_cea_list = zones_cea.Bldg_IDs.tolist()

alist = []
zones_cea_df = pd.DataFrame(data = None)
for i in zones_cea_list:
    zones_cea_df[i] = zones_all[i]
    x = list(zones_all[i].dropna())
    alist.append(x)    
#zones_cea_df.to_excel(r'C:\Users\iA\Dropbox\Com_Paper\05_Data\01_CEA_Disaggregated\02_Buildings_Info\Zones_onlyCEA_to_BldgEGIDs.xlsx')
#%%
blist = []
for j in range(len(alist)):
    for k in range(len(alist[j])):
        blist.append(alist[j][k])

df = pd.DataFrame(data = None)
df["names_4919"] = blist
df.to_excel(r'C:\Users\iA\Dropbox\Com_Paper\05_Data\01_CEA_Disaggregated\02_Buildings_Info\4919_names.xlsx')
