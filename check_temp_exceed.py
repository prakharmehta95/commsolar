# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 15:11:41 2019

@author: iA
"""
#%%
import pandas as pd
agent_list_final = pd.read_excel(r'C:\Users\iA\OneDrive - ETHZ\Thesis\PM\Data_Prep_ABM\LIST_AGENTS_FINAL.xlsx')
agent_list_final = list(agent_list_final.Bldg_IDs)

#%%
df_temp = pd.DataFrame(data = None)
for FileList in agent_list_final:
    print(FileList)
    path_dem = r"C:\Users\iA\Documents\CEA_Wiedikon\sample_1650\baseline\outputs\data\demand\\"
    zz = pd.read_excel(path_dem + FileList + '.xls')
    df_temp[FileList] = ""
    df_temp[FileList] = zz.T_int

    
#%%
#df_temp.to_pickle(r'C:\Users\iA\OneDrive - ETHZ\RA_SusTec\CEA_Disaggregation\Codes\Excel_Databases\Building_Data\T_int_newreduced_TARGET.pickle')

#df_temp = pd.read_pickle(r'C:\Users\iA\OneDrive - ETHZ\RA_SusTec\CEA_Disaggregation\Codes\Excel_Databases\Building_Data\T_int_newreduced_TARGET.pickle')

#%%
max_temp = pd.DataFrame(data = None)
max_temp["max"] = pd.DataFrame.max(df_temp)

#%%
a = max_temp.loc[max_temp['max'] > 30.000000001]

list_high_temp = list(a.index)
temp_df = pd.DataFrame(data = None)
temp_df['bldgs_exceed_28'] = list_high_temp
#temp_df.to_csv(r'C:\Users\iA\OneDrive - ETHZ\RA_SusTec\CEA_Disaggregation\Codes\Excel_Databases\Building_Data\Exceed_28.csv')
#%%

occupancy = pd.read_excel(r'C:\Users\iA\OneDrive - ETHZ\RA_SusTec\CEA_Disaggregation\Codes\Excel_Databases\Building_Data\occupancy.xlsx')
age       = pd.read_excel(r'C:\Users\iA\OneDrive - ETHZ\RA_SusTec\CEA_Disaggregation\Codes\Excel_Databases\Building_Data\age.xlsx')
loads     = pd.read_excel(r'C:\Users\iA\OneDrive - ETHZ\RA_SusTec\CEA_Disaggregation\Codes\Excel_Databases\Building_Data\loads_new.xlsx')

age = age.set_index('Name')
loads = loads.set_index('Name,C,25')
#%%
check = pd.DataFrame(data = None, index = list_high_temp)

templist = []
templist2 = []
templist3 = []
templist4 = []
check["built"] = ""
check["hvac"] = ""
check["lighting_dens"] = ""
check["envelope"] = ""
for name in list_high_temp:
    templist.append(age.loc[name]['built'])
    templist2.append(loads.loc[name]['El_Wm2,N,36,15'])
    templist3.append(age.loc[name]['HVAC'])
    templist4.append(age.loc[name]['envelope,N,20,0'])
check["built"] = templist
check["hvac"] = templist3
check["envelope"] = templist4
check["lighting_dens"] = templist2
check["Max_Temp"] = list(a['max'])

#check.to_csv(r'C:\Users\iA\OneDrive - ETHZ\RA_SusTec\CEA_Disaggregation\Codes\Excel_Databases\Building_Data\Check_Bldgs_28.csv')

#%%
occupancy = occupancy.set_index("Name")
occupancy_exceed  = occupancy.ix[list_high_temp]
