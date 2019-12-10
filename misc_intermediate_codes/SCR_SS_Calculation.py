# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 14:53:35 2019

@author: iA
"""


#%%
import pandas as pd
import numpy as np

df_demand  = pd.read_pickle(r'C:\Users\iA\Dropbox\Com_Paper\05_Data\01_CEA_Disaggregated\00_Demand_Disagg\CEA_Disaggregated_TOTAL_FINAL_3Dec.pickle')
df_solar_AC = pd.read_pickle(r'C:\Users\iA\Dropbox\Com_Paper\05_Data\01_CEA_Disaggregated\01_PV_Disagg\CEA_Disaggregated_SolarPV_3Dec.pickle')
df_solar_AC = df_solar_AC*0.97

bldgs_info = pd.read_excel(r'C:\Users\iA\Dropbox\Com_Paper\05_Data\01_CEA_Disaggregated\02_Buildings_Info\Bldgs_Info.xlsx')
agent_list_final = bldgs_info.bldg_name

#%% adding hours of the day to the demand and supply dataframes

list_hours = []  
ctr = 0  
for i in range(8760):
    if i % 24 == 0:
        ctr = 0
    list_hours.append(ctr)
    ctr = ctr + 1

df_solar_AC['Hour'] = ""
df_solar_AC['Hour'] = list_hours

df_demand['Hour'] = ""
df_demand['Hour'] = list_hours


##%% adding day of the week 
ctr = 0
days = ['Sat','Sun','Mon','Tue','Wed','Thu',' Fri'] #this order because 2005 started with a Saturday
list_days = []
df_demand['Day'] = ""
df_solar_AC['Day'] = ""
for i in range(365):
    if ctr % 7 == 0:
            ctr = 0
    if ctr == 0:
        for x in range(24):
            list_days.append('Sat')
    if ctr == 1:
        for x in range(24):
            list_days.append('Sun')
    if ctr == 2:
        for x in range(24):
            list_days.append('Mon')
    if ctr == 3:
        for x in range(24):
            list_days.append('Tue')
    if ctr == 4:
        for x in range(24):
            list_days.append('Wed')
    if ctr == 5:
        for x in range(24):
            list_days.append('Thu')
    if ctr == 6:
        for x in range(24):
            list_days.append('Fri')
    ctr = ctr + 1
    
    
df_demand['Day'] = list_days
df_solar_AC['Day'] = list_days

#%% adding info about HIGH/LOW hours of the day
import numpy as np
df_demand['price_level'] = ""
df_solar_AC['price_level'] = ""

#df_solar_AC.loc[df_solar_AC['Hour']>5,]
df_solar_AC['price_level'] = np.where(np.logical_and(np.logical_and(df_solar_AC['Hour'] > 5,df_solar_AC['Hour'] < 22), df_solar_AC['Day'] != 'Sun'),'high','low')
df_demand['price_level'] = np.where(np.logical_and(np.logical_and(df_solar_AC['Hour'] > 5,df_solar_AC['Hour'] < 22), df_solar_AC['Day'] != 'Sun'),'high','low')





#%% Preparation for NPV Calculation - savings and costs estimations
"""        
PV output reduces every year
demand remains constant
discount rate = 5%
Lifetime = 25 years
O&M costs = 0.06 CHF per kWh of solar PV production
EWZ Fee = 4Rp./kWh of Self consumption 

Separate the high and low hours of the year as prices are different, and then calculate the savings for the year

"""
print("Prep for NPV Calculation")
#dataframes to filter high and low times
df_HIGH = pd.DataFrame(data = None)        
df_LOW = pd.DataFrame(data = None)        
# =============================================================================
# READ IN FROM THE MAIN FILE
# 
# fit_high = 8.5/100 #CHF per kWH
# fit_low =  4.45/100 #CHF per kWH
# 
# #ewz_high = 24.3/100 #CHF per kWH
# #ewz_low = 14.4/100 #CHF per kWH
# ewz_solarsplit_fee = 4/100 #CHF per kWH      
# 
PV_lifetime = 1 #years
PV_degradation = 0.994 #(0.6% every year)
# OM_Cost_rate = 0.06 # CHF per kWh of solar PV production
# 
# =============================================================================
#Agents_Savings = pd.DataFrame(data = None, index = agent_list_final)
#Agents_OM_Costs = pd.DataFrame(data = None, index = agent_list_final)
#Agents_EWZ_Costs= pd.DataFrame(data = None, index = agent_list_final)
#Agents_NetSavings = pd.DataFrame(data = None, index = agent_list_final)
Agents_SCRs =  pd.DataFrame(data = None, index = agent_list_final)
Agents_SS =  pd.DataFrame(data = None, index = agent_list_final)

#-------- O&M costs ------------------
for year in range(PV_lifetime):
    print(year)
    col_name = 'Year' + str(year)
    list_om_costs = []
    
    #for fixed O&M costs ( = the O&M costs in the first year, applied for the lifetime of the PV system)
    #for i in agent_list_final:
    #    OM_costs = sum(df_solar_AC[i])*OM_Cost_rate
    #    Agents_OM_Costs[col_name] = ""
    #    list_om_costs.append(OM_costs)
    #Agents_OM_Costs[col_name] = list_om_costs
#---------------------

for year in range(PV_lifetime):
    col_name = 'Year' + str(year)
    list_savings = []
    list_om_costs = []
    list_ewz_costs = []
    list_scrs = []
    list_ss = []
    print(year)
    for i in agent_list_final:#['Z0003','Z0004']:
        print(i)
        total_PV_production = sum(df_solar_AC[i])#for SCR calculation
        total_demand = sum(df_demand[i])
        #dataframe COLUMNS initialization to hold solar and demand during HIGH and LOW hours
        df_HIGH[i + '_solar'] = ""
        df_HIGH[i + '_demand'] = ""
        df_LOW[i + '_solar'] = ""
        df_LOW[i + '_demand'] = ""
        
        #solar PV when solar is generating and prices are high or low 
        df_HIGH[i + '_solar'] = df_solar_AC.loc[np.logical_and(df_solar_AC['price_level'] == 'high',df_solar_AC[i] > 0) , i] 
        df_LOW[i + '_solar'] =  df_solar_AC.loc[np.logical_and(df_solar_AC['price_level'] == 'low', df_solar_AC[i] > 0) , i] 
        
        #demand when solar is generating and prices are high or low
        df_HIGH[i + '_demand'] =  df_demand.loc[np.logical_and(df_demand['price_level'] == 'high', df_solar_AC[i] > 0) , i] 
        df_LOW[i + '_demand'] =   df_demand.loc[np.logical_and(df_demand['price_level'] == 'low',  df_solar_AC[i] > 0) , i] 
        
        #dataframe COLUMNS to hold difference between solar and demand during HIGH and LOW hours
        df_HIGH[i + '_PV-dem'] = ""
        df_LOW[i + '_PV-dem'] = ""
        
        df_HIGH[i + '_PV-dem'] = df_HIGH[i + '_solar'] - df_HIGH[i + '_demand']
        df_LOW[i + '_PV-dem'] = df_LOW[i + '_solar'] - df_LOW[i + '_demand']
        
        #for cases when feed-in occurs (PV > demand i.e. PV-dem is +ve)
        #high times
        list_extraPV_HIGH = []
        list_dem_selfcons_HIGH = []
        list_extraPV_HIGH = df_HIGH.loc[df_HIGH[i + '_PV-dem'] >= 0, i + '_PV-dem']
        list_dem_selfcons_HIGH = df_HIGH.loc[df_HIGH[i + '_PV-dem'] >= 0, i + '_demand']
        sum_extraPV_HIGH = sum(list_extraPV_HIGH) 
        sum_dem_selfcons_HIGH = sum(list_dem_selfcons_HIGH) 
        #low times
        list_extraPV_LOW = []
        list_dem_selfcons_LOW = []
        list_extraPV_LOW = df_LOW.loc[df_LOW[i + '_PV-dem'] >= 0, i + '_PV-dem']
        list_dem_selfcons_LOW = df_LOW.loc[df_LOW[i + '_PV-dem'] >= 0, i + '_demand']
        sum_extraPV_LOW = sum(list_extraPV_LOW) 
        sum_dem_selfcons_LOW = sum(list_dem_selfcons_LOW)
        
        #for cases when only SELF-CONSUMPTION i.e. NO feed-in occurs (PV < demand i.e. PV-dem is -ve)
        #high times
        list_selfcons_HIGH = []
        list_selfcons_HIGH = df_HIGH.loc[df_HIGH[i + '_PV-dem'] < 0, i + '_solar']
        sum_selfcons_HIGH = sum(list_selfcons_HIGH)
        #low times
        list_selfcons_LOW = []
        list_selfcons_LOW =  df_LOW.loc [df_LOW[i + '_PV-dem'] < 0 , i + '_solar']
        sum_selfcons_LOW = sum(list_selfcons_LOW)
        
        
# =============================================================================
#         if diff_prices == 1:                    #wholesale or retail electricity pricing
#             if agents_info.loc[i]['GRID_MWhyr'] >=100:
#                 ewz_high = ewz_high_large       #6/100 #CHF per kWh
#                 ewz_low = ewz_low_large         #5/100 #CHF per kWh
#             elif agents_info.loc[i]['GRID_MWhyr'] < 100:
#                 ewz_high = ewz_high_small       #24.3/100 #CHF per kWh
#                 ewz_low = ewz_low_small         #14.4/100 #CHF per kWh
#         elif diff_prices == 0:                  #retail electricity pricing for all
#             ewz_high = ewz_high_small           #24.3/100 #CHF per kWh
#             ewz_low = ewz_low_small             #14.4/100 #CHF per kWh
#             
# =============================================================================
        #savings = (sum_extraPV_HIGH*fit_high + sum_dem_selfcons_HIGH*ewz_high + sum_selfcons_HIGH * ewz_high +
        #           sum_extraPV_LOW*fit_low   + sum_dem_selfcons_LOW*ewz_low   + sum_selfcons_LOW * ewz_low)
        #print(savings)
        
        total_self_consumption = sum_dem_selfcons_HIGH + sum_dem_selfcons_LOW + sum_selfcons_HIGH + sum_selfcons_LOW
        #ewz_solarsplit_costs = total_self_consumption*ewz_solarsplit_fee
        
        if total_PV_production == 0:
            scrs = 0
        else:
            scrs = total_self_consumption/total_PV_production
        
        if total_demand == 0:
            ss = 0
        else:
            ss = total_self_consumption/total_demand
        
        #Agents_Savings[col_name] = ""
        #Agents_OM_Costs[col_name] = ""
        #Agents_EWZ_Costs[col_name] = ""
        Agents_SCRs[col_name] = ""
        Agents_SS[col_name] = ""

        #list_savings.append(savings)
        #list_om_costs.append(OM_costs)
        #list_ewz_costs.append(ewz_solarsplit_costs)
        list_scrs.append(scrs)
        list_ss.append(ss)
        
        #degrading PV output every year
        df_solar_AC[i] = df_solar_AC[i]*(PV_degradation)
        
    #Agents_Savings[col_name] = list_savings
    #Agents_OM_Costs[col_name] = list_om_costs
    #Agents_EWZ_Costs[col_name] = list_ewz_costs
    Agents_SCRs[col_name] = list_scrs
    Agents_SS[col_name] = list_ss

Agents_SCRs.to_csv(r'C:\Users\iA\Dropbox\Com_Paper\05_Data\01_CEA_Disaggregated\02_Buildings_Info\ZZZ_Intermediate_Files\Agents_SCR_3Dec.csv')
Agents_SS.to_csv(r'C:\Users\iA\Dropbox\Com_Paper\05_Data\01_CEA_Disaggregated\02_Buildings_Info\ZZZ_Intermediate_Files\Agents_SS_3Dec.csv')

