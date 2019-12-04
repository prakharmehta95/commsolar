# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 17:16:21 2019

@author: iA
"""

#%%
import pandas as pd

#read in original supply by cea
supply = pd.read_pickle(r'C:\Users\iA\Documents\CEA_Wiedikon\sample_1650\baseline\inputs\building-properties\supply_28nov.pickle')

#%%

occ = pd.read_pickle(r'C:\Users\iA\Documents\CEA_Wiedikon\sample_1650\baseline\inputs\building-properties\occ_28nov.pickle')

#%%

zone_data = pd.read_excel(r'C:\Users\iA\Dropbox\Com_Paper\05_Data\05_CEA_Corrections\Zones_Archetypes_1650Reduced_28Nov.xlsx').set_index('Name')
#%%

corrections = pd.read_excel(r'C:\Users\iA\Dropbox\Com_Paper\05_Data\05_CEA_Corrections\actual_hs_stats.xlsx').set_index('CEA_code')

#%%
zone_data_copy = zone_data
for i in supply.index:
    print(i)
    
    if zone_data.loc[i]['Heating Type'] == corrections.loc['T3']['Heating_Type_Stats']:
        #print('Gas')
        supply.at[i,'type_hs'] = 'T3'
    elif zone_data.loc[i]['Heating Type'] == corrections.loc['T1']['Heating_Type_Stats']:
        supply.at[i,'type_hs'] = 'T1'
    elif zone_data.loc[i]['Heating Type'] == corrections.loc['T11']['Heating_Type_Stats']:
        supply.at[i,'type_hs'] = 'T11'
    elif zone_data.loc[i]['Heating Type'] == corrections.loc['T1_1']['Heating_Type_Stats']:
        supply.at[i,'type_hs'] = 'T1'
    elif zone_data.loc[i]['Heating Type'] == corrections.loc['T99']['Heating_Type_Stats']:
        continue#supply.at[i,'type_hs'] = 'T3'
    elif zone_data.loc[i]['Heating Type'] == corrections.loc['T2']['Heating_Type_Stats']:
        supply.at[i,'type_hs'] = 'T2'
    elif zone_data.loc[i]['Heating Type'] == corrections.loc['T0']['Heating_Type_Stats']:
        supply.at[i,'type_hs'] = 'T0'
    elif zone_data.loc[i]['Heating Type'] == corrections.loc['T98']['Heating_Type_Stats']:
        continue#supply.at[i,'type_hs'] = 'T3'
    elif zone_data.loc[i]['Heating Type'] == corrections.loc['T8']['Heating_Type_Stats']:
        supply.at[i,'type_hs'] = 'T8'
    elif zone_data.loc[i]['Heating Type'] == corrections.loc['T6']['Heating_Type_Stats']:
        supply.at[i,'type_hs'] = 'T6'
    elif zone_data.loc[i]['Heating Type'] == corrections.loc['T5']['Heating_Type_Stats']:
        supply.at[i,'type_hs'] = 'T5'
    elif zone_data.loc[i]['Heating Type'] == corrections.loc['T8_1']['Heating_Type_Stats']:
        supply.at[i,'type_hs'] = 'T8'
    elif zone_data.loc[i]['Heating Type'] == corrections.loc['T9']['Heating_Type_Stats']:
        supply.loc[i]['type_hs'] = 'T9'
    elif zone_data.loc[i]['Heating Type'] == corrections.loc['T4']['Heating_Type_Stats']:
        supply.at[i,'type_hs'] = 'T4'
        

supply.to_csv(r'C:\Users\iA\Dropbox\Com_Paper\05_Data\05_CEA_Corrections\supply_acc_to_stats.csv')
