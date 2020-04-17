# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 17:58:04 2019

@author: Prakhar Mehta
"""

#%%
"""
Initial libraries and data import from the run_adoption_loop file

__main__ is the run_adoption_loop script

"""
from __main__ import *
#%%
"""
useful packages to work with
"""

import random 
import itertools
from random import seed
from random import gauss     
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import networkx as nx

#ABM Packages
from mesa import Agent, Model                                                  #base classes in the mesa library
from Scheduler_StagedActivation_Random import StagedActivation_random          #slightly modified StagedActivation to have control over the random seed
from mesa.datacollection import DataCollector                                  #for data collection, of course

#%%
'''
AGENT INFORMATION
'''
path = r'C:\Users\prakh\Dropbox\Com_Paper\\'
#INFORMATION on agents and all combinations
agents_info = pd.read_excel(path + r'05_Data\01_CEA_Disaggregated\02_Buildings_Info\Bldgs_Info.xlsx')

#CHECK - these might change!
df_demand   = pd.read_pickle(path + r'\05_Data\01_CEA_Disaggregated\00_Demand_Disagg\CEA_Disaggregated_TOTAL_FINAL_3Dec.pickle')
df_solar    = pd.read_pickle(path + r'\05_Data\01_CEA_Disaggregated\01_PV_Disagg\CEA_Disaggregated_SolarPV_3Dec.pickle')
df_solar    = df_solar.copy()*0.97 #converting solar PV DC output to AC 

df_solar_combos     = pd.DataFrame(data = None)     #holds PV potential of the communities formed
df_demand_combos    = pd.DataFrame(data = None)     #holds demands      of the communities formed
Combos_formed_Info  = pd.DataFrame(data = None)     #holds information  of the  communities formed

#NPV information of individual agents
from __main__ import Agents_NPVs as Agents_Ind_NPVs
from __main__ import Agents_SCRs as Agents_Ind_SCRs
from __main__ import Agents_Investment_Costs as Agents_Ind_Investment_Costs 
#from __main__ import Agents_Peer_Network as Agents_Peers
from __main__ import Agents_PPs_Norm as Agents_Ind_PPs_Norm                    #payback periods for individual agents normalized

from __main__ import distances as distances #Holds the distances from each agent to its nearest 200 agents (200 agents because dataframe size is reasonable but still satisfies all criteria for the ABM conceptual model)

list_agents = list(agents_info.bldg_name)

#existing PV installations
PV_already_installed = agents_info.loc[agents_info['pv_already_installed_size_kW']>0]
list_installed_solar_bldgs = PV_already_installed.bldg_name

#creating new columns to hold information on the agents
agents_info['intention']    = 0
agents_info['Comm_NPV']     = 0
agents_info['Ind_NPV']      = 0
agents_info['Reason']       = ""
agents_info['Ind_SCR']      = 0
agents_info['Comm_SCR']     = 0
agents_info['Adopt_IND']    = 0        #saves 1 if INDIVIDUAL adoption occurs, else stays 0
agents_info['Adopt_COMM']   = 0       #saves 1 if COMMUNITY  adoption occurs, else stays 0
agents_info['En_Champ']     = 0         #saves who is the energy champion of that community
agents_info['Adopt_Year']   = 0             #saves year of adoption
agents_info['Community_ID'] = ""    #community ID of the community formed
agents_info['Individual_ID'] = ""    #individual ID of the individual PV formed. Eg = PV_B123456 etc...
agents_info['bldg_names']   = ""
agents_info['bldg_names']   = agents_info['bldg_name']
agents_info = agents_info.set_index('bldg_names')

#agents_subplots = subplots_final

#%% AGENT code

#c = 0
step_ctr = 0

class tpb_agent(Agent):
    """Class for the agents. Agents are initialised in init, and the step_idea
    and step_decision methods execute what the agents are supposed to do
    """
    def tpb_agent(self):
        self.adopt_ind
    def __init__(self,unique_id,model,bldg_type,bldg_own,bldg_zone,bldg_plot,attitude,pp,intention,intention_yr,
                 peer_effect,neighbor_influence,total,counter,adopt_ind,adopt_comm,adopt_year,en_champ,pv_size,dem_total):
        '''
        maybe also attributes like: adopt_ind,adopt_comm
        Agent initialization
        
        Agents have attributes:
            unique_id   = Agent unique identification (B140907,...)
            bldg_type   = Type of the building 
            attitude    = Environmental attitude [0,1]
            pp          = payback period ratio - indicator of economic attractiveness [0,1]
            intention   = initial idea to adopt/not adopt solar: BOOLEAN 0|1
            peer_effect = ratio of people in small world network who install solar [0,1]
            total       = sum of the intention, from the stage1_intention function [0,1]
            counter     = to know number of times the agent crosses intention. Initialize to 0, if 1 then go to stage2,else do not go to stage2
                            also - if 0 or >1, then do not count in subplot effects!
            adopt_ind   = if agent adopts individual system BOOLEAN 0|1
            adopt_comm  = if agent adopts community system BOOLEAN 0|1
            en_champ    = if this agent is the energy champion or not
            pv_size     = size of the pv system
            dem_total   = total demand of the agent 
        
        '''
        super().__init__(unique_id,model) 
        self.unique_id          = unique_id 
        self.bldg_type          = bldg_type
        self.bldg_own           = bldg_own
        self.bldg_zone          = bldg_zone
        self.bldg_plot          = bldg_plot
        self.attitude           = attitude              #environmental attitude
        self.pp                 = pp                    #perceived profitability
        self.peer_effect        = peer_effect           #peer effect - calculated in the step_idea later so just add a placeholder here for agent initialization?
        self.neighbor_influence = neighbor_influence    #stores the neighbor_influence, called with the 'check_neighbours_subplots' function in step_idea(self)
        self.total              = total                 #sum of the intention function
        self.counter            = counter
        self.intention          = intention             #stores the intention of the agent (whether or not it passed idea stage)
        self.intention_yr       = intention_yr          #stores the intention of the agent every year of the simulation
        self.adopt_ind          = adopt_ind
        self.adopt_comm         = adopt_comm
        self.adopt_year         = adopt_year
        self.en_champ           = en_champ
        self.pv_size            = pv_size
        self.dem_total          = dem_total
        #self.egids = egids
        
    def step_idea(self):
        '''
        defines what the agent does in his step.
        Since StagedActivation is used this is the first stage.
        IDEA/INTENTION developments happens in this stage
        '''
        # only run the step_idea if agent has not already adopted!- NEED TO CHANGE THIS TYPE OF REASONING - NOW DEPENDS ON STATE OF THE AGENT
        if self.adopt_comm == 1 or self.adopt_ind == 1:
            self.intention = 0                          #do it again to be safe - so that an agent which adopted is out of the simulation
        else:
            #print("start of new agent uid = ",self.unique_id, "has intention = ",self.intention)
            tpb_functions.check_peers(self,self.unique_id)                            #sets the peer_effect
            tpb_functions.check_neighbours_subplots(self,self.unique_id)              #sets the neighbor_influence
            self.pp = tpb_functions.econ_attr(self, self.unique_id)
            
            #call the intention function
            stage1_intention(self,self.unique_id, self.attitude,self.pp,self.peer_effect,self.neighbor_influence)
            
    def step_decision(self):
        """
        After developing the intention in the step_idea, agents make the final
        decision in this step.
        """
        #only those agents whose intention is HIGH can go to decision making
        if self.intention == 1:
            global r
            r = 0
            stage2_decision(self,self.unique_id,self.intention) #call the decision making function i.e. stage 2 of the ABM simulation
            
        
#%% MODEL code
        
        
class tpb(Model):
    '''
    Model setup. The main ABM runs from here.
    Called the FIRST time when the execution happens. Agents are initiated
    '''    
    
    global list_datacollector
    list_datacollector = []
    
    def __init__(self,N,randomseed):
        print("--TPB MODEL CALLED--")
        agents_info['Adopt_IND'] = 0            #saves 1 if INDIVIDUAL adoption occurs, else stays 0
        agents_info['Adopt_COMM'] = 0           #saves 1 if COMMUNITY  adoption occurs, else stays 0
        agents_info['Community_ID'] = ""        #community ID of the community formed
        agents_info['Year'] = 0
        
        super().__init__()
        self.num_agents = N
        self.randomseed = randomseed
        self.schedule = StagedActivation_random(self,stage_list = ['step_idea','step_decision'],
                                             shuffle = True, shuffle_between_stages = True,seed = self.randomseed) 
        #only called once so the seed is the same, meaning the agents are shuffled the same way every time
        
        
        global agents_objects_list
        agents_objects_list = []            #list of initialized agents
        
        
        global norm_dist_env_attitude_list
        global step_ctr #year of the simulation
        step_ctr = 0
        global i
        for i in list_agents:
            
            
            #these are just placeholders for the agent attributes. They will be set in the agent initialization in the agent class
            intention = 0
            intention_yr = 0
            ratio = 0
            total = 0
            counter = 0
            neighbor_influence = 0
            adopt_ind = 0
            adopt_comm = 0
            adopt_year = 0
            en_champ = 0
            
            #AGENT CREATION - I directly pass attributes in the call to the agent class for agent creation...
            a = tpb_agent(agents_info.loc[i]['bldg_name'],
                          self,
                          agents_info.loc[i]['bldg_type'],
                          agents_info.loc[i]['bldg_owner'],
                          agents_info.loc[i]['zone_id'],
                          agents_info.loc[i]['plot_id'],
                          tpb_functions.env_attitude(agents_info.loc[i]['bldg_name']),
                          tpb_functions.econ_attr(self, self.unique_id),
                          intention,
                          intention_yr,
                          ratio,
                          neighbor_influence,
                          total,
                          counter,
                          adopt_ind,
                          adopt_comm,
                          adopt_year,
                          en_champ,
                          agents_info.loc[i]['pv_size_kw'],
                          agents_info.loc[i]['demand_yearly_kWh'])
            
            agents_objects_list.append(a) #agent objects stored in this list of objects so that they are easily accessible later 
            self.schedule.add(a)
            
        #data collection
        self.datacollector = DataCollector(model_reporters = {"Ind_solar_number"        :functions.cumulate_solar_ind,
                                                              "Ind_PV_Installed_CAP"    :functions.cumulate_solar_ind_sizes,
                                                              "Comm_solar_number"       :functions.cumulate_solar_comm,
                                                              "Num_of_Comms"            :functions.cumulate_solar_champions,
                                                              "Comm_PV_Installed_CAP"   :functions.cumulate_solar_comm_sizes,
                                                              "GYM_PV_CAP"              :functions.agent_type_gym_CAP,
                                                              "HOSPITAL_PV_CAP"         :functions.agent_type_hospital_CAP,
                                                              "HOTEL_PV_CAP"            :functions.agent_type_hotel_CAP,
                                                              "INDUSTRIAL_PV_CAP"       :functions.agent_type_industrial_CAP,
                                                              "LIBRARY_PV_CAP"          :functions.agent_type_library_CAP,
                                                              "MULTI_RES_PV_CAP"        :functions.agent_type_multi_res_CAP,
                                                              "OFFICE_PV_CAP"           :functions.agent_type_office_CAP,
                                                              "PARKING_PV_CAP"          :functions.agent_type_parking_CAP,
                                                              "SCHOOL_PV_CAP"           :functions.agent_type_school_CAP,
                                                              "SINGLE_RES_PV_CAP"       :functions.agent_type_single_res_CAP,
                                                              "Num_GYM"                 :functions.agent_type_gym,
                                                              "Num_HOSPITAL"            :functions.agent_type_hospital,
                                                              "Num_HOTEL"               :functions.agent_type_hotel,
                                                              "Num_INDUSTRIAL"          :functions.agent_type_industrial,
                                                              "Num_LIBRARY"             :functions.agent_type_library,
                                                              "Num_MULTI_RES"           :functions.agent_type_multi_res,
                                                              "Num_OFFICE"              :functions.agent_type_office,
                                                              "Num_PARKING"             :functions.agent_type_parking,
                                                              "Num_SCHOOL"              :functions.agent_type_school,
                                                              "Num_SINGLE_RES"          :functions.agent_type_single_res},
                                            agent_reporters = {"Building_ID"        :"unique_id",
                                                               "Building_Type"      :"bldg_type",
                                                               "Part_Comm"          :"part_comm",
                                                               "PV_Size"            :"pv_size",
                                                               "Demand_MWhyr"       :"dem_total",
                                                               "Intention"          :"intention",
                                                               "Yearly_Intention"   :"intention_yr",
                                                               "Attitude"           :"attitude",
                                                               "Payback Period"     :"pp",
                                                               "SWN Ratio"          :"peer_effect",
                                                               "Intention_Sum"      : "total",
                                                               "Subplot Effect"     :"neighbor_influence",
                                                               "Individual"         :"adopt_ind",
                                                               "Community"          :"adopt_comm",
                                                               "Year_Adoption"      :"adopt_year",
                                                               "Energy_Champion"    :"en_champ"})
                                    
        self.running = True
        
        
    def step(self):
        
        self.schedule.step()
        self.datacollector.collect(self)        #collects data at the end of the step/year
        
        global step_ctr                         #Counter for the year of the simulation. Used to change agent attributes based on the year
        step_ctr += 1
        
    

#%% MORE FUNCTIONS for agent attributes and also for outputs of ABM segregated by agent types

class functions:
    """
    FUNCTIONS for agent attributes, counting installed capacity by building typologies
    
    @Alejandro: most of these functions here are for data collection.
    Check the datacollector definition and you will see. Can be simply modified
    so that we can collect data based on ownership/building-use type separately
    """
        
#-----GYM-----    
    def agent_type_gym(model):
        '''
        to find total number of GYM individual adoptions
        '''
        sum_agent_type_gym_ind = 0
        for i in model.schedule.agents:
            if i.bldg_type == 'GYM':
                sum_agent_type_gym_ind = sum_agent_type_gym_ind + i.adopt_ind
        return sum_agent_type_gym_ind
    
    
    def agent_type_gym_CAP(model):
        '''
        to find total CAPACITY of GYM individual adoptions
        '''
        sum_agent_type_gym_ind_cap = 0
        for i in model.schedule.agents:
            if i.bldg_type == 'GYM' and i.adopt_ind == 1:
                sum_agent_type_gym_ind_cap = sum_agent_type_gym_ind_cap + i.pv_size
        return sum_agent_type_gym_ind_cap

#-----HOSPITAL-----    
    def agent_type_hospital(model):
        '''
        to find total number of HOSPITAL individual adoptions
        '''
        sum_agent_type_hospital_ind = 0
        for i in model.schedule.agents:
            if i.bldg_type == 'HOSPITAL':
                sum_agent_type_hospital_ind = sum_agent_type_hospital_ind + i.adopt_ind
        return sum_agent_type_hospital_ind
    
    def agent_type_hospital_CAP(model):
        '''
        to find total CAPACITY of HOSPITAL individual adoptions
        '''
        sum_agent_type_hospital_ind_cap = 0
        for i in model.schedule.agents:
            if i.bldg_type == 'HOSPITAL' and i.adopt_ind == 1:
                sum_agent_type_hospital_ind_cap = sum_agent_type_hospital_ind_cap + i.pv_size
        return sum_agent_type_hospital_ind_cap

#-----HOTEL-----    
    def agent_type_hotel(model):
        '''
        to find total number of HOTEL individual adoptions
        '''
        sum_agent_type_hotel_ind = 0
        for i in model.schedule.agents:
            if i.bldg_type == 'HOTEL':
                sum_agent_type_hotel_ind = sum_agent_type_hotel_ind + i.adopt_ind
        return sum_agent_type_hotel_ind
    
    def agent_type_hotel_CAP(model):
        '''
        to find total CAPACITY of HOTEL individual adoptions
        '''
        sum_agent_type_hotel_ind_cap = 0
        for i in model.schedule.agents:
            if i.bldg_type == 'HOTEL' and i.adopt_ind == 1:
                sum_agent_type_hotel_ind_cap = sum_agent_type_hotel_ind_cap + i.pv_size
        return sum_agent_type_hotel_ind_cap

#-----INDUSTRIAL-----
    def agent_type_industrial(model):
        '''
        to find total number of INDUSTRIAL individual adoptions
        '''
        sum_agent_type_industrial_ind = 0
        for i in model.schedule.agents:
            if i.bldg_type == 'INDUSTRIAL':
                sum_agent_type_industrial_ind = sum_agent_type_industrial_ind + i.adopt_ind
        return sum_agent_type_industrial_ind
    
    def agent_type_industrial_CAP(model):
        '''
        to find total CAPACITY of INDUSTRIAL individual adoptions
        '''
        sum_agent_type_industrial_ind_cap = 0
        for i in model.schedule.agents:
            if i.bldg_type == 'INDUSTRIAL' and i.adopt_ind == 1:
                sum_agent_type_industrial_ind_cap = sum_agent_type_industrial_ind_cap + i.pv_size
        return sum_agent_type_industrial_ind_cap

#-----LIBRARY-----    
    def agent_type_library(model):
        '''
        to find total number of LIBRARY individual adoptions
        '''
        sum_agent_type_library_ind = 0
        for i in model.schedule.agents:
            if i.bldg_type == 'LIBRARY':
                sum_agent_type_library_ind = sum_agent_type_library_ind + i.adopt_ind
        return sum_agent_type_library_ind
    
    def agent_type_library_CAP(model):
        '''
        to find total CAPACITY of LIBRARY individual adoptions
        '''
        sum_agent_type_library_ind_cap = 0
        for i in model.schedule.agents:
            if i.bldg_type == 'LIBRARY' and i.adopt_ind == 1:
                sum_agent_type_library_ind_cap = sum_agent_type_library_ind_cap + i.pv_size
        return sum_agent_type_library_ind_cap

#-----MULTI_RES-----    
    def agent_type_multi_res(model):
        '''
        to find total number of MULTI_RES individual adoptions
        '''
        sum_agent_type_multi_res_ind = 0
        for i in model.schedule.agents:
            if i.bldg_type == 'MULTI_RES':
                sum_agent_type_multi_res_ind = sum_agent_type_multi_res_ind + i.adopt_ind
        return sum_agent_type_multi_res_ind
    
    def agent_type_multi_res_CAP(model):
        '''
        to find total CAPACITY of MULTI_RES individual adoptions
        '''
        sum_agent_type_multi_res_ind_cap = 0
        for i in model.schedule.agents:
            if i.bldg_type == 'MULTI_RES' and i.adopt_ind == 1:
                sum_agent_type_multi_res_ind_cap = sum_agent_type_multi_res_ind_cap + i.pv_size
        return sum_agent_type_multi_res_ind_cap

#-----OFFICE-----    
    def agent_type_office(model):
        '''
        to find total number of OFFICE individual adoptions
        '''
        sum_agent_type_office_ind = 0
        for i in model.schedule.agents:
            if i.bldg_type == 'OFFICE':
                sum_agent_type_office_ind = sum_agent_type_office_ind + i.adopt_ind
        return sum_agent_type_office_ind
    
    def agent_type_office_CAP(model):
        '''
        to find total CAPACITY of OFFICE individual adoptions
        '''
        sum_agent_type_office_ind_cap = 0
        for i in model.schedule.agents:
            if i.bldg_type == 'OFFICE' and i.adopt_ind == 1:
                sum_agent_type_office_ind_cap = sum_agent_type_office_ind_cap + i.pv_size
        return sum_agent_type_office_ind_cap

#-----PARKING-----    
    def agent_type_parking(model):
        '''
        to find total number of PARKING individual adoptions
        '''
        sum_agent_type_parking_ind = 0
        for i in model.schedule.agents:
            if i.bldg_type == 'PARKING':
                sum_agent_type_parking_ind = sum_agent_type_parking_ind + i.adopt_ind
        return sum_agent_type_parking_ind
    
    def agent_type_parking_CAP(model):
        '''
        to find total CAPACITY of PARKING individual adoptions
        '''
        sum_agent_type_parking_ind_cap = 0
        for i in model.schedule.agents:
            if i.bldg_type == 'PARKING' and i.adopt_ind == 1:
                sum_agent_type_parking_ind_cap = sum_agent_type_parking_ind_cap + i.pv_size
        return sum_agent_type_parking_ind_cap

#-----RESTAURANT-----    
    def agent_type_restaurant(model):
        '''
        to find total number of RESTAURANT individual adoptions
        '''
        sum_agent_type_restaurant_ind = 0
        for i in model.schedule.agents:
            if i.bldg_type == 'RESTAURANT':
                sum_agent_type_restaurant_ind = sum_agent_type_restaurant_ind + i.adopt_ind
        return sum_agent_type_restaurant_ind
    
    def agent_type_restaurant_CAP(model):
        '''
        to find total CAPACITY of RESTAURANT individual adoptions
        '''
        sum_agent_type_restaurant_ind_cap = 0
        for i in model.schedule.agents:
            if i.bldg_type == 'RESTAURANT' and i.adopt_ind == 1:
                sum_agent_type_restaurant_ind_cap = sum_agent_type_restaurant_ind_cap + i.pv_size
        return sum_agent_type_restaurant_ind_cap

#-----RETAIL-----   
    def agent_type_retail(model):
        '''
        to find total number of RETAIL individual adoptions
        '''
        sum_agent_type_retail_ind = 0
        for i in model.schedule.agents:
            if i.bldg_type == 'RETAIL':
                sum_agent_type_retail_ind = sum_agent_type_retail_ind + i.adopt_ind
        return sum_agent_type_retail_ind
    
    def agent_type_retail_CAP(model):
        '''
        to find total CAPACITY of RETAIL individual adoptions
        '''
        sum_agent_type_retail_ind_cap = 0
        for i in model.schedule.agents:
            if i.bldg_type == 'RETAIL' and i.adopt_ind == 1:
                sum_agent_type_retail_ind_cap = sum_agent_type_retail_ind_cap + i.pv_size
        return sum_agent_type_retail_ind_cap

#-----SCHOOL-----   
    def agent_type_school(model):
        '''
        to find total number of SCHOOL individual adoptions
        '''
        sum_agent_type_school_ind = 0
        for i in model.schedule.agents:
            if i.bldg_type == 'SCHOOL':
                sum_agent_type_school_ind = sum_agent_type_school_ind + i.adopt_ind
        return sum_agent_type_school_ind
    
    def agent_type_school_CAP(model):
        '''
        to find total CAPACITY of SCHOOL individual adoptions
        '''
        sum_agent_type_school_ind_cap = 0
        for i in model.schedule.agents:
            if i.bldg_type == 'SCHOOL' and i.adopt_ind == 1:
                sum_agent_type_school_ind_cap = sum_agent_type_school_ind_cap + i.pv_size
        return sum_agent_type_school_ind_cap

#-----SINGLE_RES-----
    def agent_type_single_res(model):
        '''
        to find total number of SINGLE_RES individual adoptions
        '''
        sum_agent_type_single_res_ind = 0
        for i in model.schedule.agents:
            if i.bldg_type == 'SINGLE_RES':
                sum_agent_type_single_res_ind = sum_agent_type_single_res_ind + i.adopt_ind
        return sum_agent_type_single_res_ind
    
    def agent_type_res_CAP(model):
        '''
        to find total CAPACITY of SINGLE_RES individual adoptions
        '''
        sum_agent_type_single_res_ind_cap = 0
        for i in model.schedule.agents:
            if i.bldg_type == 'SINGLE_RES' and i.adopt_ind == 1:
                sum_agent_type_single_res_ind_cap = sum_agent_type_single_res_ind_cap + i.pv_size
        return sum_agent_type_single_res_ind_cap

#--------------------------
        
    def cumulate_solar_ind(model):
        """
        To find the cumulative INDIVIDUAL installations at the end of every time step
        """
        solar_sum_ind = 0
        for i in model.schedule.agents:
            solar_sum_ind = solar_sum_ind + i.adopt_ind
        return solar_sum_ind
            
    def cumulate_solar_comm(model):
        """
        To find the cumulative COMMUNITY "ALL buildings with installations" at the end of every time step
        """
        solar_sum_comm = 0
        for i in model.schedule.agents:
            solar_sum_comm = solar_sum_comm + i.adopt_comm
        return solar_sum_comm  
    
    def cumulate_solar_champions(model):
        """
        To find the cumulative COMMUNITY installations at the end of every time step
         = total number of communities formed
        """
        solar_sum_champ = 0
        for i in model.schedule.agents:
            solar_sum_champ = solar_sum_champ + i.en_champ
        return solar_sum_champ        
    
    def cumulate_solar_ind_sizes(model):
        """
        To find the cumulative INDIVIDUAL solar capacity at the end of every time step
        """
        solar_sum_sizes = 0
        
        for i in model.schedule.agents:
            if i.adopt_ind == 1:
                solar_sum_sizes = solar_sum_sizes + i.pv_size
        return solar_sum_sizes
     
    def cumulate_solar_comm_sizes(model):
        """
        To find the cumulative COMMUNITY solar capacity at the end of every time step
        """
        solar_comm_sizes = 0
        for i in model.schedule.agents:
            if i.adopt_comm == 1:
                solar_comm_sizes = solar_comm_sizes + i.pv_size
        return solar_comm_sizes
    

class tpb_functions:
    """
    Functions to calculate values for the intention function
    """
 
    def env_attitude(uid):#(self, uid, attitude):
        """
        To change the environmental attitude depending on the steps.
        @Alejandro : use beta-pert function instead of gauss
        """
        uid = uid
        if agents_info.loc[uid]['minergie'] == 1:
            value = 0.95
        else:
            value = gauss(0.698,0.18)
        if value > 1:
            value = 1 #to keep it from going over 1
        return value
    
    
    def econ_attr(self, uid):
        """
        To update the payback ratio every step
        Takes data from the profitability_index pickle
        @Alejandro: Check the Agents_Ind_PPs_Norm dataframe to ensure the values make sense 
        """
        uid = uid#self.unique_id
        try:
            self.pp = Agents_Ind_PPs_Norm#profitability_index.loc[step_ctr][uid]
        except KeyError:
            self.pp = 0
        
        if self.pp < 0:
            self.pp = 0
            
        return self.pp
    
    def check_peers(self,uid):
        """
        Checks how many other agents (or peers) in the SWN have installed solar
        sets agent attribute peer_effect (self.peer_effect) accordingly
        
        Uses:
            Agents_Peers (it is a DataFrame) - list of all people in the SWN of each building
        """
        swn_with_solar = 0
        temp_list = []
        temp_list = []#Agents_Peers[uid].values.tolist() #make a list of all peers of that particular agent with uid
        
        """
        Explanation of the for loops coming ahead:
            each agent considered by referencing its self.unique_id (or uid)
            for z in temp_list() --> z becomes one of the peers of building 'uid' 
                I run the next for loop in the range of all the agents:
                    IF an agent has the same unique_id as z i.e. I have found peer#1 of 'uid'
                    in the list of all agents then:
                        if the intention of that peer#1 is 1, advance count of swn_with_solar.
            Loop runs for all the peers which 'uid' has
            
            THEN - peer_effect is calculated as:
                all peers in the swn with solar/all peers in the swn 
        """
        
        for z in temp_list:
            for y in range(len(agents_objects_list)): #checking all other agents in the simulation 
                if agents_objects_list[y].unique_id == z:
                    #print("uid of peer = ",agents_objects_list[y].unique_id, "adopted solar = ", agents_objects_list[y].adopt_ind)
                    if agents_objects_list[y].adopt_ind == 1 or agents_objects_list[y].adopt_comm == 1: 
                        swn_with_solar = swn_with_solar + 1
        
        
        self.peer_effect = (swn_with_solar/1)#len(Agents_Peers.loc[:,self.unique_id]))
                         
    def check_neighbours_subplots(self,uid):
        """
        Checks how many other agents in the --subplot = NEIGHBOURS-- have INTENTION/IDEA of installing solar
        sets agent attribute neighbor_influence (self.neighbor_influence) accordingly
        **WORKING WITH INTENTION as this is subplot peers**
        Uses:
            agents_subplots (it is a DataFrame) - list of all agents in the subplot of each building
        """
        temp_df_ZEV_members = pd.DataFrame(data = None)
        temp_df_ZEV_members = agents_info.loc[agents_info['bldg_name'] == uid]
        
        """
        Explanation of the for loops coming ahead:
            each agent considered by referencing its self.unique_id (or uid)
            for z in temp_list_subplots() --> z becomes one of the peers of building 'uid' 
                IF a neighbour has positive intention then:
                    advance neighbor_influence_counter.
            Loop runs for all the neighbours which 'uid' has
            
            THEN - neighbor_influence is calculated as:
                    all neighbours with positive intention/all neighoburs
        """
        
        neighbor_influence_counter = 0      #initialize the counter to 0. This is to see how many of subplot neighbouts in total have intented to adopt solar this year
        
        for z in temp_df_ZEV_members.bldg_name:
            for y in range(len(agents_objects_list)):
                if agents_objects_list[y].unique_id == z:
                    #print("bldg considered = ",uid, "intention of neighbour ", z, "is = ",agents_objects_list[y].intention)
                    if (agents_objects_list[y].intention == 1 and agents_objects_list[y].counter == 1):
                        #print("~~bldg considered = ",uid, "intention of neighbour ", z, "is = ",agents_objects_list[y].intention)
                        neighbor_influence_counter += 1
        self.neighbor_influence = (neighbor_influence_counter/len(temp_df_ZEV_members.bldg_name)) 
        

#%%
"""
Making a CLASS for the community information
"""
#class communities_formed_info(self, members, type_comm):
#    pass
    
        
        
 
#%% STAGES of the ABM




def stage1_intention(self, uid, attitude, pp,ratio,neighbor_influence):
    """
    Intention development at the end of stage 1
    Considers the environmental attitude, payback period ratio and the peer_effect
    Final intention at the end of every step saved as self.intention
    """


    #the basic intention function:
    total = w_att*attitude + w_econ*pp + w_swn*ratio + w_subplot*neighbor_influence
    self.total = total
    
    # GOAL OF THIS PART -> include buildings with existing solar in model outputs
    # TO DO -> MOVE TO INITIALIZATION DURING AGENT CREATION
    # --> BUILDINGS WITH PV IN REALITY, CREATE AS ALREADY ADOPTED
    # --> perhaps get it from the agents_info file itself...
    #checking for buildings with already installed solar and initializing their intention to 1 regardless of the total
    if scenario == "ZEV" or scenario == "no_ZEV":
        if self.unique_id in list_installed_solar_bldgs_100MWh:
            print(self.unique_id)
            total = 1
            self.total = total
            self.intention = 1
    elif scenario == "TOP4_no100MWh_retail" or scenario == "TOP4_no100MWh_wholesale":
        if self.unique_id in list_installed_solar_bldgs_ALL:
            print(self.unique_id)
            total = 1
            self.total = total
            self.intention = 1
        
        
    if total > threshold:
        intention = 1
        self.counter = self.counter + 1
        self.intention = intention
        self.intention_yr = intention
    else:
        intention = 0
        self.intention = intention
        self.intention_yr = intention
        
        # TO DO -> MAKE SURE DATA COLLECTION WORKS REGARDLESS OF AGENT GETTING
        # INTO STAGE 2 OR NOT
        #since in the last year if the intention is 0, stage 2 will not be entered. Results won't be written, hence write them here.
        if step_ctr == 17:
            agents_info.update(pd.Series([0], name  = 'Adopt_IND', index = [self.unique_id]))
            agents_info.update(pd.Series([self.total], name  = 'intention', index = [self.unique_id]))
            agents_info.update(pd.Series(["Intention<threshold"], name  = 'Reason', index = [self.unique_id]))

    

def stage2_decision(self,uid,idea):
    """
    Decision made here after passing the intention stage
    """
    
    #only entered if intention is 1! Modify if necessary
    if self.intention == 1:
        temp_plot_id = agents_info.loc[self.unique_id]['plot_id']
        same_plot_agents = agents_info[agents_info['plot_id']==temp_plot_id]
        same_plot_agents_positive_intention = same_plot_agents[same_plot_agents['intention'] == 1 or same_plot_agents['adoption'] == 1] #available to form community
        #only agents without solar will have the intention variable as '1'. If an agent has individual/community PV then intention is always '0', but adoption will be '1'
        
        #check for community formation 
        from community_combos import community_combinations
        Combos_Info, NPV_Combos, df_solar_combos_possible, df_demand_combos_possible, comm_name = community_combinations(agents_info, same_plot_agents_positive_intention,
                                                                                                                         distances, df_solar, df_demand, df_solar_combos,
                                                                                                                         df_demand_combos, Combos_formed_Info,
                                                                                                                         self.unique_id, step_ctr)
        
        #keeping info of the community formed as it is needed later in case an agent wants to join a particular community
        temp_comm_name                                  = comm_name                                 #'C_' + comm_name - CHECK if this needs to be done. comm_name itself sends back a name like: 'C_B123456_B789101112'
        df_solar_combos[temp_comm_name]                 = df_solar_combos_possible[comm_name]       #add a new column  
        df_demand_combos[temp_comm_name]                = df_demand_combos_possible[comm_name]      #add a new column  
        Combos_formed_Info.loc[Combos_Info.index[0]]    = Combos_Info.iloc[0]                       #copying the only row in Combos_Info to Combos_formed_Info
        
        
        if len(Combos_Info.index) != 0: #meaning that some community is formed
            #here compare with individual NPVs
            
            if Agents_Ind_NPVs.loc[self.unique_id]['npv'] < Combos_Info.loc[temp_comm_name]['npv_share_en_champ'] and Combos_Info.loc[temp_comm_name]['npv_share_en_champ'] > 0:
                #form a community
                #set the adoption as 1 for all the constituent  buildings
                #set some variable which indicates whether it is a community or an individual PV system
                agents_adopting_comm = Combos_Info.combos_bldg_names
                for g in agents_adopting_comm:
                    if g == self.unique_id:
                        # ACTIVATED AGENT : ENERGY CHAMPION 
                        self.en_champ = 1                                                                               #setting the agent which is the energy champion - the first agent
                        agents_info.update(pd.Series([self.en_champ], name  = 'En_Champ', index = [self.unique_id]))
                    for h in range(len(agents_objects_list)):
                        if g == agents_objects_list[h].unique_id:
                            agents_objects_list[h].adopt_comm = 1                                                       #setting community adoption as 1 for all agents involved
                            self.adopt_year = 2018 + step_ctr
                            agents_objects_list[h].intention = 0                                                        #setting intention as 0 for all agents involved
                            agents_info.update(pd.Series([1],               name  = 'Adopt_COMM',   index = [g]))
                            agents_info.update(pd.Series([2018+step_ctr],   name  = 'Year',         index = [g]))
                            agents_info.update(pd.Series([temp_comm_name],  name  = 'Community_ID', index = [g]))
                            agents_info.update(pd.Series([self.total],      name  = 'intention',    index = [g]))
                            #agents_info.update(pd.Series([share_npv],       name  = 'Comm_NPV',     index = [g])) #npv share of each building to be saved here, NOT COMPLETE AS ON 10 DEC
                            #agents_info.update(pd.Series([ind_npv],         name  = 'Ind_NPV',      index = [g])) #npv share of each building to be saved here, NOT COMPLETE AS ON 10 DEC
                            agents_info.update(pd.Series(["Comm>Ind"],      name  = 'Reason',       index = [g]))
                            #agents_info.update(pd.Series([ind_scr],         name  = 'Ind_SCR',      index = [g])) #scr of each building to be saved here, NOT COMPLETE AS ON 10 DEC
                            #agents_info.update(pd.Series([comm_scr],        name  = 'Comm_SCR',     index = [g]))
            
            elif Agents_Ind_NPVs.loc[self.unique_id]['npv'] >= Combos_Info.loc[temp_comm_name]['npv_share_en_champ'] and Agents_Ind_NPVs.loc[self.unique_id]['npv'] > 0:
                #adopt individual
                self.adopt_ind  = 1
                self.adopt_year = 2018 + step_ctr
                ind_npv = Agents_Ind_NPVs.loc[step_ctr][self.unique_id]
                agents_info.update(pd.Series([1],               name  = 'Adopt_IND',    index = [self.unique_id]))
                agents_info.update(pd.Series([2018+step_ctr],   name  = 'Year',         index = [self.unique_id]))
                agents_info.update(pd.Series([self.total],      name  = 'intention',    index = [self.unique_id]))
                agents_info.update(pd.Series([ind_npv],         name  = 'Ind_NPV',      index = [self.unique_id]))
                agents_info.update(pd.Series(["Only_Ind"],      name  = 'Reason',       index = [self.unique_id]))
                self.intention  = 0
                self.adopt_comm = 0
            
        elif len(Combos_Info.index) == 0:
        #meaning that NO community is formed, hence go for individual PV adoption
            if Agents_Ind_NPVs.loc[self.unique_id]['npv'] >=0:
                #adopt individual
                #set adoption as 1 for an individual PV formation
                self.adopt_ind  = 1
                self.adopt_year = 2018 + step_ctr
                ind_npv         = Agents_Ind_NPVs.loc[step_ctr][self.unique_id]
                agents_info.update(pd.Series([1],               name  = 'Adopt_IND',    index = [self.unique_id]))
                agents_info.update(pd.Series([2018+step_ctr],   name  = 'Year',         index = [self.unique_id]))
                agents_info.update(pd.Series([self.total],      name  = 'intention',    index = [self.unique_id]))
                agents_info.update(pd.Series([ind_npv],         name  = 'Ind_NPV',      index = [self.unique_id]))
                agents_info.update(pd.Series(["Only_Ind"],      name  = 'Reason',       index = [self.unique_id]))
                # TO DO - CHANGE IN ORDER TO ALLOW THEM TO CHANGE AFTER INDIVIDUAL
                # ADOPTION
                # WHY? BECAUSE .INTENTION IS USED TO LET IN/OUT AGENTS INTO 
                # INTENTION STAGE
                self.intention  = 0
                self.adopt_comm = 0
            
        
        
    
    