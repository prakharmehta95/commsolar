# -*- coding: utf-8 -*-
"""
Created on Wed May  8 18:23:37 2019

@author: Prakhar Mehta
"""

#%% IMPORT PACKAGES AND DEPENDENCIES
import pandas as pd
import time, pickle, random
start = time.time()

from NPV_Calculation import npv_calc_individual
from small_world_network import make_swn
#from adoption_8_NPV_vs_InvCosts import *
#from adoption_8_NPV_vs_InvCosts import scenario

#%% SET MANUAL INPUTS

## Model parameters
#initializing the weights used in the intention function 
w_econ      = 0.9#0.30
w_swn       = 0.9#0.31
w_att       = 0.9#0.39
w_subplot   = 0.9#0.1
threshold   = 0.1
#allows negative NPV to also install, as long as it -5% of investment cost
reduction   = -0.05
#setting seed for the peer network calculations
peer_seed   = 1

## Scenario parameters
#limit of 100 MWh for community size applies
comm_limit  = 1
# DEFAULT - ZEV formation not allowed. Binary variable to turn on/off 
# whether to allow community formation
ZEV         = 1
no_closest_neighbors_consider = 4

## Economic parameters
fit_high            = 8.5/100   #CHF per kWH
fit_low             = 4.45/100  #CHF per kWH
ewz_high_large      = 6/100     #CHF per kWh
ewz_low_large       = 5/100     #CHF per kWh
ewz_high_small      = 24.3/100  #CHF per kWh
ewz_low_small       = 14.4/100  #CHF per kWh
ewz_solarsplit_fee  = 4/100     #CHF per kWH      
# if diff_prices = 1, then both wholesale and retail prices are applied 
# where appropriate. Else, all retail prices
diff_prices         = 1         

#PV Panel Properties
PV_lifetime         = 25        #years
PV_degradation      = 0.994     #(0.6% every year)
OM_Cost_rate        = 0.06      #CHF per kWh of solar PV production

#discount rates 
disc_rate           = 0.05      #discount rate for NPV Calculation
pp_rate             = 0         #discount rate for payback period calculation
# =============================================================================
# keep it the same for all
# disc_rate_homeown   = 0.05      #discount rate for NPV Calculation
# disc_rate_firm      = 0.05      #discount rate for NPV Calculation
# disc_rate_instn     = 0.05      #discount rate for NPV Calculation
# disc_rate_landlord  = 0.05      #discount rate for NPV Calculation
# =============================================================================
#%% IMPORT DATA

# Define the path to the data files
path = r'C:\Users\prakh\Dropbox\Com_Paper\\'
path = r'C:\Users\anunezji\Dropbox\Com_Paper\\'

## Buildings data
# IMPORTANT -> this is the test excel file not the full version file
b_data_file = r'05_Data\01_CEA_Disaggregated\02_Buildings_Info\Bldgs_Info_ABM_Test.xlsx'
agents_info = pd.read_excel(path + b_data_file)
agent_list_final = agents_info.bldg_name

## Economic data
#define the costs etc here which are read in the NPV_Calculation file
e_data_file = r'05_Data\02_ABM_input_data\02_pv_prices\PV_Prices.xlsx'
PV_price_baseline   = pd.read_excel(path + e_data_file)

## Geographical data
g_data_file = r'07_GIS\DataVisualization_newData\distances_nearest_200bldgs_v1.csv'
distances = pd.read_csv(path + g_data_file)

#%% CALCULATE NET-PRESENT-VALUE FOR INDIVIDUAL AGENTS
 
# runs the NPV calculation code and calculates individual NPVs for the agents
# involved
Agents_NPVs, Agents_SCRs, Agents_Investment_Costs, Agents_PPs_Norm =npv_calc_individual(path,PV_price_baseline,disc_rate,
                                                                                         pp_rate, fit_high, fit_low,
                                                                                         ewz_high_large,ewz_low_large,
                                                                                         ewz_high_small, ewz_low_small,
                                                                                         diff_prices, ewz_solarsplit_fee,
                                                                                         PV_lifetime, PV_degradation,
                                                                                         OM_Cost_rate, agents_info,agent_list_final)

#%% CREATE SMALL-WORLD NETWORK

# Use the make_swn function to create the small-world network among agents
Agents_Peer_Network = make_swn(distances, agents_info,peer_seed)

#%% RUN SIMULATIONS

#4919   #number of agents
number = len(agent_list_final)
#how long should the ABM run for - ideally, 18 years from 2018-2035
years = 18     

#empty dictionaries to store results
results_agentlevel = {}
results_emergent = {}
d_gini_correct = {}
d_gini_model_correct = {}
d_agents_info_runs_correct = {}
d_combos_info_runs_correct = {}

#initial seed for the attitude gauss function. Used to reproduce results.  
att_seed = 3        
#no of runs. 100 for a typical ABM simulation in this work
runs = 1         
#initial seed used to set the order of shuffling of agents within the scheduler
randomseed = 22     

print("Did you change the name of the final pickle storage file?") 
#so that my results are not overwritten!

from adoption_8_NPV_vs_InvCosts import *
#from adoption_8_NPV_vs_InvCosts import scenario

#main loop for the ABM simulation
for j in range(runs):
    print("run = ",j,"-------------------------------------------------------")
    
    #642 is just any number to change the seed for every run 
    randomseed = randomseed + j*642

    # initializes by calling the model from adoption_8_NPV_vs_InvCosts and 
    # setting up with the class init methods
    test = tpb(number,randomseed)

    #seed for attitude changes in a new run
    att_seed = att_seed + j*10          


    for i in range(years):
        # for the environmental attitude which remains constant for an agent 
        # in a particular run
        seed(att_seed)
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

#%% RESULTS STORAGE

# Export data to pickle to save it!

#enter name of the stored result file
#f = open("03June_ZEV_d_agents_info.pickle","wb")
pickle.dump(d_agents_info_runs_correct,f)
f.close()

#enter name of the stored result file
#f = open("03June_ZEV_d_gini.pickle","wb")
pickle.dump(results_agentlevel,f)
f.close()

#enter name of the stored result file
#f = open("03June_ZEV_d_combos_info_runs.pickle","wb")
pickle.dump(d_combos_info_runs_correct,f)
f.close()

#enter name of the stored result file
#f = open("03June_ZEV_d_gini_model.pickle","wb")
pickle.dump(results_emergent,f)
f.close()

end = time.time()
print("Code Execution Time = ",end - start)