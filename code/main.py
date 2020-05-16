# -*- coding: utf-8 -*-
"""
Created on Wed May  8 18:23:37 2019

@author: Prakhar Mehta
"""

#%%
import pandas as pd
import numpy as np
import time
start = time.time()
import pickle
import random
from datetime import timedelta
import datetime 
    

from Tools import npv_ind, swn, comm_combos
#%%
"""
define all parameters here
"""


#initializing the weights used in the intention function 
w_econ      = 0.30#0.30
w_swn       = 0.30#0.31
w_att       = 0.30#0.39
w_subplot   = 0.1#0.1
threshold   = 0.5
reduction   = -0.05     #allows negative NPV to also install, as long as it -5% of investment cost
diff_prices = 1         #if demand >100 MWh then wholesale prices for them. If set to 0, then retail prices for all irrespective of demand
ZEV         = 1         #DEFAULT - ZEV formation not allowed. Binary variable to turn on/off whether to allow community formation
peer_seed   = 1           #setting seed for the peer network calculations    

no_closest_neighbors_consider = 4
#%%
"""
Agent information read from excel, pickles etc...

!!!!!!TEST EXCEL being read in now!!!!!!!!!!
"""

#path = r'C:\Users\prakh\Dropbox\Com_Paper\\'
path = r'C:\Users\no23sane\Dropbox (Personal)\Com_Paper\\'
agents_info = pd.read_excel(path + r'05_Data\01_CEA_Disaggregated\02_Buildings_Info\Bldgs_Info_ABM_Test.xlsx')
agent_list_final = agents_info.bldg_name

#%%
"""
NPV Calculation call from here - calculates the NPVs of individual buildings
"""
#define the costs etc here which are read in the npv_ind file
PV_price_baseline   = pd.read_excel(path + r'05_Data\02_ABM_input_data\02_pv_prices\PV_Prices.xlsx')
fit_high            = 8.5/100   #CHF per kWH
fit_low             = 4.45/100  #CHF per kWH
ewz_high_large      = 6/100     #CHF per kWh
ewz_low_large       = 5/100     #CHF per kWh
ewz_high_small      = 24.3/100  #CHF per kWh
ewz_low_small       = 14.4/100  #CHF per kWh
ewz_solarsplit_fee  = 4/100     #CHF per kWH      
diff_prices         = 1         #if = 1, then both wholesale and retail prices are applied where appropriate. Else, all retail prices

#PV Panel Properties
PV_lifetime         = 25        #years
PV_degradation      = 0.994     #(0.6% every year)
OM_Cost_rate        = 0.06      #CHF per kWh of solar PV production

#discount rates 
disc_rate           = 0.05      #discount rate for NPV Calculation
pp_rate             = 0         #discount rate for payback period calculation is zero 
# =============================================================================
# keep it the same for all
# disc_rate_homeown   = 0.05      #discount rate for NPV Calculation
# disc_rate_firm      = 0.05      #discount rate for NPV Calculation
# disc_rate_instn     = 0.05      #discount rate for NPV Calculation
# disc_rate_landlord  = 0.05      #discount rate for NPV Calculation
# =============================================================================

#adding hours of the day to the demand and supply dataframes for npv calculation
list_hours = []  
ctr = 0  
for i in range(8760):
    if i % 24 == 0:
        ctr = 0
    list_hours.append(ctr)
    ctr = ctr + 1
    
#adding day of the week to the demand and supply dataframes for npv calculation
weekDays    = ("Mon","Tues","Wed","Thurs","Fri","Sat","Sun")
day_count   = 365                                                               #1 year
daylist     = []
start_date  =  datetime.date(2005,1,1)                                          #reference year is 2005
for single_date in (start_date + timedelta(n) for n in range(day_count)):
    DayAsString = weekDays[single_date.weekday()]
    for i in range (24):
        daylist.append(DayAsString)
    




'''
PV PRICES in the next years. Base PV price data from EnergieSchweiz.
Projections Source = IEA Technology Roadmap 2014
'''
       
PV_price_projection = pd.DataFrame(data = None, columns = ['Year'])
PV_price_projection['Year'] = list(range(2018,2041))#years
for i in list(PV_price_baseline.columns):
    fp_array = [PV_price_baseline.loc[0][i],PV_price_baseline.loc[0][i]/2]
    y = np.interp([i for i in range(1,24)], [1,23],fp_array)
    PV_price_projection[i] = ""
    PV_price_projection[i] = y
    
#runs the NPV calculation code and calculates individual NPVs for the agents involved
Agents_NPVs , Agents_SCRs, Agents_Investment_Costs, Agents_PPs_Norm =npv_ind.npv_calc_individual(path,PV_price_baseline,disc_rate,
                                                                                         pp_rate, fit_high, fit_low,
                                                                                         ewz_high_large,ewz_low_large,
                                                                                         ewz_high_small, ewz_low_small,
                                                                                         diff_prices, ewz_solarsplit_fee,
                                                                                         PV_lifetime, PV_degradation,
                                                                                         OM_Cost_rate, agents_info,agent_list_final,
                                                                                         PV_price_projection,list_hours,daylist)


#%%
#Creation of Small World Network
distances = pd.read_csv(path + r'07_GIS\DataVisualization_newData\distances_nearest_200bldgs_v1.csv') #all the distances to each building 
Agents_Peer_Network = swn.make_swn(distances, agents_info,peer_seed) #calls swn function


#%%
number = len(agent_list_final) #4919   #number of agents
years = 18    #how long should the ABM run for - ideally, 18 years from 2018 - 2035

#empty dictionaries to store results
results_agentlevel = {}
results_emergent = {}
d_gini_correct = {}
d_gini_model_correct = {}
d_agents_info_runs_correct = {}
d_combos_info_runs_correct = {}

att_seed = 3        #initial seed for the attitude gauss function. Used to reproduce results.  
runs = 1         #no of runs. 100 for a typical ABM simulation in this work
randomseed = 22     #initial seed used to set the order of shuffling of agents withing the scheduler

print("Did you change the name of the final pickle storage file?") #so that my results are not overwritten!
from agent_model import *

#main loop for the ABM simulation
for j in range(runs):
    print("run = ",j,"----------------------------------------------------------------")
    randomseed = randomseed + j*642     #642 is just any number to change the seed for every run 
    test = tpb(number,randomseed)       #initializes by calling the model from agent_model and setting up with the class init methods
    att_seed = att_seed + j*10          #seed for attitude changes in a new run


    for i in range(years):
        seed(att_seed)                  #for the environmental attitude which remains constant for an agent in a particular run
        print("YEAR:",i+1)
        test.step()
        temp_name_3 = "agents_info_" + str(j) + "_" + str(i)
        temp_name_4 = "combos_info_" + str(j) + "_" + str(i)
        
        #stores results across multiple Years and multiple runs
        t1 = pd.DataFrame.copy(agents_info)
        t2 = pd.DataFrame.copy(Combos_formed_Info) 
        d_agents_info_runs_correct[temp_name_3] = t1#agents_info
        d_combos_info_runs_correct[temp_name_4] = t2#Agents_Possibles_Combos
    
    temp_name = "gini_" + str(j)
    temp_name_2 = "gini_model_" + str(j)
    agent_vars = test.datacollector.get_agent_vars_dataframe()
    model_vars = test.datacollector.get_model_vars_dataframe()
    
    #stores results across multiple runs
    results_agentlevel[temp_name] = agent_vars
    results_emergent[temp_name_2] = model_vars
    


#%% Export data to pickle to save it!

#f = open("03June_ZEV_d_agents_info.pickle","wb") #enter name of the stored result file
#pickle.dump(d_agents_info_runs_correct,f)
#f.close()

#f = open("03June_ZEV_d_gini.pickle","wb") #enter name of the stored result file
#pickle.dump(results_agentlevel,f)
#f.close()

#f = open("03June_ZEV_d_combos_info_runs.pickle","wb") #enter name of the stored result file
#pickle.dump(d_combos_info_runs_correct,f)
#f.close()

#f = open("03June_ZEV_d_gini_model.pickle","wb") #enter name of the stored result file
#pickle.dump(results_emergent,f)
#f.close()



end = time.time()
print("Code Execution Time = ",end - start)

