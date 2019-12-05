# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 14:25:54 2019

@author: iA
"""

#%%
from community_combos import community_combinations

import pandas as pd
import numpy as np

df_solar = pd.read_pickle(r'C:\Users\prakh\Dropbox\Com_Paper\05_Data\01_CEA_Disaggregated\01_PV_Disagg\CEA_Disaggregated_SolarPV_3Dec.pickle')
df_solar = df_solar*0.97
df_demand = pd.read_pickle(r'C:\Users\prakh\Dropbox\Com_Paper\05_Data\01_CEA_Disaggregated\00_Demand_Disagg\CEA_Disaggregated_TOTAL_FINAL_3Dec.pickle')
data = pd.read_excel(r'C:\Users\prakh\Dropbox\Com_Paper\05_Data\01_CEA_Disaggregated\02_Buildings_Info\Bldgs_Info.xlsx')

data['intention'] = ""

data['intention'] = [ 1 if i%2 == 1 else 1 for i in range(len(data.index))] # temporarily setting intention so that I can replicate what happens in the ABM

distances = pd.read_csv(r'C:\Users\prakh\Dropbox\Com_Paper\07_GIS\DataVisualization_newData\distances_nearest_200bldgs_v1.csv') #all the distances to each building 

'''
lets say that the filtering is already done and I have the building which is being considered at the moment 
'''
df_solar_combos_possible = pd.DataFrame(data = None)
df_demand_combos_possible = pd.DataFrame(data = None)

#these will hold the info on the combos - need to send this to the functions...
df_solar_combos_main = pd.DataFrame(data = None)
df_demand_combos_main = pd.DataFrame(data = None)
Combos_formed_Info = pd.DataFrame(data = None)

Combos_Info, NPV_Combos, df_solar_combos_possible, df_demand_combos_possible, comm_name, combos_consider_op = community_combinations(data, distances, df_solar, df_demand, df_solar_combos_main, df_demand_combos_main,Combos_formed_Info)

#%% trying to simulate with already exisitng commnites - hence create such data files

df_solar_combos_main = pd.DataFrame(data = None)
df_demand_combos_main = pd.DataFrame(data = None)
temp_comm_name = 'C_' + comm_name
df_solar_combos_main[temp_comm_name] = df_solar_combos_possible[comm_name] 
df_demand_combos_main[temp_comm_name] = df_demand_combos_possible[comm_name]

Combos_formed_Info = Combos_Info
#%%

