# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 16:50:57 2019

@author: iA
"""

#%% 
import pandas as pd
import numpy as np

sfoe = pd.read_excel(r'C:\Users\iA\Dropbox\Com_Paper\05_Data\03_SFOE_Data\SFOE_Data_Roofs_Zurich.xlsx')
bldgs_info = pd.read_excel(r'C:\Users\iA\Dropbox\Com_Paper\05_Data\01_CEA_Disaggregated\02_Buildings_Info\Bldgs_Info.xlsx')
names = sfoe.names.dropna()

#%%
df_bldg = pd.DataFrame(data = None)

list_names = []
list_egids = []
list_areas = []
list_pv = []
c = 0
for i in names:
    c=c+1
    print(c)
    df_temp = sfoe.loc[sfoe['GWR_EGID']==i]
    list_egids.append(i)
    name = 'B' + str(int(i))
    list_names.append(name)
    area = sum(df_temp.FLAECHE)
    pv = sum(df_temp.STROMERTRAG)
    list_areas.append(area)
    list_pv.append(pv)
    #for j in range(len(sfoe.GWR_EGID)):
df_bldg['EGID'] = list_egids
df_bldg['Name'] = list_names
df_bldg['Area'] = list_areas
df_bldg['PV'] = list_pv

df_bldg.to_csv(r'C:\Users\iA\Dropbox\Com_Paper\05_Data\03_SFOE_Data\SFOE_Data_Bldgs_Concise.csv')


#%%
#df_bldg = df_bldg.set_index('Name')    
compare_sfoe_cea = pd.DataFrame(data = None)

compare_sfoe_cea['Bldg_Name'] = bldgs_info.bldg_name
compare_sfoe_cea['Bldg_EGID'] = bldgs_info.bldg_EGID
compare_sfoe_cea['CEA_RoofArea'] = bldgs_info.roof_area_m2
compare_sfoe_cea['CEA_PV'] = bldgs_info.pv_yearly_kWh

compare_sfoe_cea.to_csv(r'C:\Users\iA\Dropbox\Com_Paper\05_Data\03_SFOE_Data\Compare.csv')
list_sfoe_area = []
list_sfoe_pv = []


for i in compare_sfoe_cea.Bldg_Name:
    try:
        area = df_bldg.loc[i]['Area']
        list_sfoe_area.append(area)
        pv = df_bldg.loc[i]['PV']
        list_sfoe_pv.append(pv)
    except KeyError:
        continue
    
compare_sfoe_cea['SFOE_RoofArea_Flat'] = list_sfoe_area
compare_sfoe_cea['SFOE_PV'] = list_sfoe_pv