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
#%%
data = data.set_index('bldg_name', drop = False)
data['Adopt_COMM']   = 0       #saves 1 if COMMUNITY  adoption occurs, else stays 0
data['Adopt_IND']   = 0       #saves 1 if COMMUNITY  adoption occurs, else stays 0
data['En_Champ']     = 0         #saves who is the energy champion of that community
data['Adopt_Year']   = 0             #saves year of adoption
data['Community_ID'] = ""    #community ID of the community formed
data['Individual_ID'] = ""    #individual ID of the individual PV formed. Eg = PV_B123456 etc...

#lets say that the filtering is already done and I have the building which is being considered at the moment 
uid = 'B144827'
year = 1 #temoporarily setting it to 1 so the NPV function knows what price to take...


#only entered if intention is 1! Modify if necessary
#if intention == 1:
temp_plot_id = data.loc[uid]['plot_id']
same_plot_agents = data[data['plot_id']==temp_plot_id]

#what temporarily works is without the boolean:
same_plot_agents_positive_intention = same_plot_agents[same_plot_agents['intention'] == 1]# or same_plot_agents['adoption'] == 1] #available to form community
"""
#but what I really want is wit the the boolean
same_plot_agents_positive_intention = same_plot_agents[same_plot_agents['intention'] == 1 or same_plot_agents['adoption'] == 1] #available to form community
"""
#only agents without solar will have the intention variable as '1'. If an agent has individual/community PV then intention is always '0', but adoption will be '1'
                

df_solar_combos_possible = pd.DataFrame(data = None)
df_demand_combos_possible = pd.DataFrame(data = None)

#these will hold the info on the combos - need to send this to the functions...
df_solar_combos_main = pd.DataFrame(data = None)
df_demand_combos_main = pd.DataFrame(data = None)
Combos_formed_Info = pd.DataFrame(data = None)

Combos_Info, NPV_Combos, df_solar_combos_possible, df_demand_combos_possible, comm_name, combos_consider_op = community_combinations(data, same_plot_agents_positive_intention,distances, df_solar, df_demand, df_solar_combos_main, df_demand_combos_main,Combos_formed_Info, uid, year)

#%% trying to simulate with already exisitng commnites - hence create such data files

df_solar_combos_main = pd.DataFrame(data = None)
df_demand_combos_main = pd.DataFrame(data = None)
temp_comm_name = 'C_' + comm_name
df_solar_combos_main[temp_comm_name] = df_solar_combos_possible[comm_name] 
df_demand_combos_main[temp_comm_name] = df_demand_combos_possible[comm_name]

Combos_formed_Info = Combos_Info
#%%

