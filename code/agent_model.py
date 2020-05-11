# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 17:58:04 2019

@author: Prakhar Mehta
"""

#%%
"""
Initial libraries and data import from the run_adoption_loop file

__main__ is the main script

"""

#%%
"""
useful packages to work with
"""
from __main__ import *
#NPV information of individual agents
from __main__ import Agents_NPVs as Agents_Ind_NPVs
from __main__ import Agents_SCRs as Agents_Ind_SCRs
from __main__ import Agents_Investment_Costs as Agents_Ind_Investment_Costs 
from __main__ import Agents_Peer_Network as Agents_Peers
from __main__ import Agents_PPs_Norm as Agents_Ind_PPs_Norm                    #payback periods for individual agents normalized

#from community_combos import community_combinations

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
from scheduler import StagedActivation_random  #slightly modified StagedActivation to have control over the random seed
    
from mesa.datacollection import DataCollector                                  #for data collection, of course

from Tools import rank_combos, npv_combo, dc_functions
#%%
'''
AGENT INFORMATION
'''
path = r'C:\Users\no23sane\Dropbox (Personal)\Com_Paper\\'
#path = r'C:\Users\prakh\Dropbox\Com_Paper\\'
#path = r'C:\Users\anunezji\Dropbox\Com_Paper\\'

#CHECK - these might change!
df_demand   = pd.read_pickle(path + r'\05_Data\01_CEA_Disaggregated\00_Demand_Disagg\CEA_Disaggregated_TOTAL_FINAL_3Dec.pickle')
df_solar    = pd.read_pickle(path + r'\05_Data\01_CEA_Disaggregated\01_PV_Disagg\CEA_Disaggregated_SolarPV_3Dec.pickle')
df_solar    = df_solar.copy()*0.97 #converting solar PV DC output to AC 

df_solar_combos     = pd.DataFrame(data = None)     #holds PV potential of the communities formed
df_demand_combos    = pd.DataFrame(data = None)     #holds demands      of the communities formed
Combos_formed_Info  = pd.DataFrame(data = None)     #holds information  of the  communities formed

list_agents = list(agents_info.bldg_name)

#existing PV installations
PV_already_installed = agents_info.loc[agents_info['pv_already_installed_size_kW']>0]
list_installed_solar_bldgs = PV_already_installed.bldg_name

#creating new columns to hold information on the agents
agents_info['intention']    = 0#[ 1 if i%2 == 1 else 0 for i in range(len(agents_info.index))]# 0
agents_info['Comm_NPV']     = 0
agents_info['Ind_NPV']      = 0
agents_info['Reason']       = ""
agents_info['Ind_SCR']      = 0
agents_info['Comm_SCR']     = 0
agents_info['Adopt_IND']    = 0         #saves 1 if INDIVIDUAL adoption occurs, else stays 0
agents_info['Adopt_COMM']   = 0         #saves 1 if COMMUNITY  adoption occurs, else stays 0
agents_info['En_Champ']     = 0         #saves who is the energy champion of that community
agents_info['Adopt_Year']   = 0         #saves year of adoption
agents_info['Community_ID'] = ""        #community ID of the community formed
agents_info['Individual_ID'] = ""       #individual ID of the individual PV formed. Eg = PV_B123456 etc...
agents_info['bldg_names']   = ""

agents_info['bldg_names']   = agents_info['bldg_name']
agents_info = agents_info.set_index('bldg_name', drop = False)

#agents_subplots = subplots_final

#%% AGENT code

#c = 0
step_ctr = 0

class tpb_agent(Agent):
    """Class for the agents. Agents are initialised in init, and the step_idea
    and step_decision methods execute what the agents are supposed to do
    """
    #def tpb_agent(self):
    #    self.adopt_ind
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
        self.pp                 = tpb_functions.econ_attr(self, self.unique_id)#pp                    #perceived profitability
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
        #only those agents whose intention is HIGH can go to decision making,
        # or those who have already adopted individual or community PV
        if self.intention == 1 or self.adopt_ind == 1 or self.adopt_comm == 1:
            global r
            r = 0
            stage2_decision(self,self.unique_id,self.intention) #call the decision making function i.e. stage 2 of the ABM simulation
            
        
#%% MODEL code
        
class tpb(Model):
    '''
    Model setup. The main ABM runs from here.
    Called the FIRST time when the execution happens. Agents are initiated
    '''    
    
    def __init__(self,N,randomseed):
        #print("--TPB MODEL CALLED--")
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
                          0,
                          #tpb_functions.econ_attr(self, self.unique_id),
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
        self.datacollector = DataCollector(model_reporters = {"Ind_solar_number"        :dc_functions.functions.cumulate_solar_ind,
                                                              "Ind_PV_Installed_CAP"    :dc_functions.functions.cumulate_solar_ind_sizes,
                                                              "Comm_solar_number"       :dc_functions.functions.cumulate_solar_comm,
                                                              "Num_of_Comms"            :dc_functions.functions.cumulate_solar_champions,
                                                              "Comm_PV_Installed_CAP"   :dc_functions.functions.cumulate_solar_comm_sizes,
                                                              "GYM_PV_CAP"              :dc_functions.functions.agent_type_gym_CAP,
                                                              "HOSPITAL_PV_CAP"         :dc_functions.functions.agent_type_hospital_CAP,
                                                              "HOTEL_PV_CAP"            :dc_functions.functions.agent_type_hotel_CAP,
                                                              "INDUSTRIAL_PV_CAP"       :dc_functions.functions.agent_type_industrial_CAP,
                                                              "LIBRARY_PV_CAP"          :dc_functions.functions.agent_type_library_CAP,
                                                              "MULTI_RES_PV_CAP"        :dc_functions.functions.agent_type_multi_res_CAP,
                                                              "OFFICE_PV_CAP"           :dc_functions.functions.agent_type_office_CAP,
                                                              "PARKING_PV_CAP"          :dc_functions.functions.agent_type_parking_CAP,
                                                              "SCHOOL_PV_CAP"           :dc_functions.functions.agent_type_school_CAP,
                                                              "SINGLE_RES_PV_CAP"       :dc_functions.functions.agent_type_single_res_CAP,
                                                              "Num_GYM"                 :dc_functions.functions.agent_type_gym,
                                                              "Num_HOSPITAL"            :dc_functions.functions.agent_type_hospital,
                                                              "Num_HOTEL"               :dc_functions.functions.agent_type_hotel,
                                                              "Num_INDUSTRIAL"          :dc_functions.functions.agent_type_industrial,
                                                              "Num_LIBRARY"             :dc_functions.functions.agent_type_library,
                                                              "Num_MULTI_RES"           :dc_functions.functions.agent_type_multi_res,
                                                              "Num_OFFICE"              :dc_functions.functions.agent_type_office,
                                                              "Num_PARKING"             :dc_functions.functions.agent_type_parking,
                                                              "Num_SCHOOL"              :dc_functions.functions.agent_type_school,
                                                              "Num_SINGLE_RES"          :dc_functions.functions.agent_type_single_res},
                                            agent_reporters = {"Building_ID"        :"unique_id",
                                                               "Building_Type"      :"bldg_type",
                                                               #"Part_Comm"          :"part_comm",
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
        
 #%%   
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
            self.pp = Agents_Ind_PPs_Norm[uid][step_ctr]#profitability_index.loc[step_ctr][uid]
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
        
        
        self.peer_effect = (swn_with_solar/len(Agents_Peers.loc[:,self.unique_id]))
                         
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
    
    if total > threshold:
        intention = 1
        self.counter = self.counter + 1
        self.intention = intention
        self.intention_yr = intention
        agents_info.at[self.unique_id,'intention'] = intention
    else:
        intention = 0
        self.intention = intention
        self.intention_yr = intention
        
        # TO DO -> MAKE SURE DATA COLLECTION WORKS REGARDLESS OF AGENT GETTING
        # INTO STAGE 2 OR NOT
        #since in the last year if the intention is 0, stage 2 will not be entered.
        #Results won't be written, hence write them here.
        if step_ctr == 17:
            agents_info.at[self.unique_id,'Adopt_IND'] = 0
            agents_info.at[self.unique_id,'intention'] = self.total
            agents_info.at[self.unique_id,'Reason'] = "Intention"
            
    

def stage2_decision(self,uid,idea):
    """
    Decision made here after passing the intention stage
    """
    
    #only entered if intention is 1! Modify if necessary
    if self.intention == 1: #or self.adopt_comm ==1 or self.adopt_ind == 1:
        temp_plot_id = agents_info.loc[self.unique_id]['plot_id']
        same_plot_agents = agents_info[agents_info['plot_id']==temp_plot_id]
        same_plot_agents_positive_intention = same_plot_agents[(same_plot_agents['intention'] >0) | (same_plot_agents['Adopt_IND'] >0)] #or (same_plot_agents.adoption == 1)] #available to form community
        #only agents without solar will have the intention variable as '1'.
        #If an agent has individual/community PV then intention is always '0',
        #but adoption will be '1'
            
        #this should only be called if same_plot_agents_positive_intention is not empty,
        # and if ZEV is 1 meaning community formation is allowed
        if len(same_plot_agents_positive_intention.index) > 1 and (ZEV == 1):
            
            Combos_Info, NPV_Combos, df_solar_combos_possible, df_demand_combos_possible, comm_name = comm_combos.community_combinations(agents_info, same_plot_agents_positive_intention,         
                                                                                                                         distances, df_solar, df_demand, df_solar_combos,
                                                                                                                         df_demand_combos, Combos_formed_Info,
                                                                                                                         self.unique_id, agents_info.loc[self.unique_id]['zone_id'],
                                                                                                                         no_closest_neighbors_consider,step_ctr,Agents_Ind_NPVs,
                                                                                                                         disc_rate, fit_high, fit_low, ewz_high_large,ewz_low_large,
                                                                                                                         ewz_high_small, ewz_low_small,ewz_solarsplit_fee,
                                                                                                                         PV_lifetime, PV_degradation, OM_Cost_rate,npv_combo,rank_combos,
                                                                                                                         PV_price_projection,list_hours, daylist,diff_prices)
                                                                                                                         
            if len(Combos_Info.index) != 0: #meaning that some community is formed
                #here compare with individual NPVs
                
                #keeping info of the community formed as it is needed later in case an agent wants to join a particular community
                temp_comm_name                                  = comm_name                                 #'C_' + comm_name - CHECK if this needs to be done. comm_name itself sends back a name like: 'C_B123456_B789101112'
                #temp_comm_name = 'C_' + comm_name
                df_solar_combos[temp_comm_name]                 = df_solar_combos_possible[temp_comm_name]       #add a new column  
                df_demand_combos[temp_comm_name]                = df_demand_combos_possible[comm_name]      #add a new column  
                row_info                                        = Combos_Info.iloc[0]
                Combos_formed_Info[temp_comm_name]              = row_info                       #copying the only row in Combos_Info to Combos_formed_Info
                
                temp_comm_name = 'C_' + comm_name
                if Agents_Ind_NPVs.at[step_ctr,self.unique_id] < Combos_Info.at[temp_comm_name,'npv_share_en_champ'] and Combos_Info.at[temp_comm_name,'npv_share_en_champ'] > reduction*Agents_Ind_Investment_Costs.at[step_ctr,self.unique_id]:
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
                                
                                #I used to set intention to zero after adoption, but not doing it anymore
                                #every agent calculates intention at every step inthe ABM for now
                                #agents_objects_list[h].intention = 0                                                        #setting intention as 0 for all agents involved
                                
                                #writing to dataframe all constituent buildings who adopt community PV
                                
                                #consider using agents_info.at[g,'Adopt_COMM'] = 1
                                #agents_info.update(pd.Series([1],               name  = 'Adopt_COMM',   index = [g]))
                                #agents_info.update(pd.Series([2018+step_ctr],   name  = 'Year',         index = [g]))
                                #agents_info.update(pd.Series([temp_comm_name],  name  = 'Community_ID', index = [g]))
                                #agents_info.update(pd.Series([self.total],      name  = 'intention',    index = [g]))
                                agents_info.at[g,'Year'] = 2018 + step_ctr
                                agents_info.at[g,'Adopt_COMM'] = 1
                                agents_info.at[g,'Community_ID'] = temp_comm_name
                                agents_info.at[g,'intention'] = self.total
                                agents_info.at[g,'Reason'] = "Comm>Ind"
                                #npv share of each building to be saved here, NOT COMPLETE - do not calculate the info for this!
                                #agents_info.update(pd.Series([share_npv],       name  = 'Comm_NPV',     index = [g])) 
                                #agents_info.update(pd.Series(["Comm>Ind"],      name  = 'Reason',       index = [g]))
                                
                                #scr of each building to be saved here, NOT COMPLETE - do not calculate the info for this! 
                                #agents_info.update(pd.Series([comm_scr],        name  = 'Comm_SCR',     index = [g]))
                
                elif Agents_Ind_NPVs.at[step_ctr,self.unique_id] > Combos_Info.at[temp_comm_name,'npv_share_en_champ'] and Agents_Ind_NPVs.at[step_ctr,self.unique_id] > reduction*Agents_Ind_Investment_Costs.at[step_ctr,self.unique_id]:
                    #adopt individual
                    self.adopt_ind  = 1
                    self.adopt_year = 2018 + step_ctr
                    ind_npv = Agents_Ind_NPVs.at[step_ctr,self.unique_id]
                    #agents_info.update(pd.Series([1],               name  = 'Adopt_IND',    index = [self.unique_id]))
                    #agents_info.update(pd.Series([2018+step_ctr],   name  = 'Year',         index = [self.unique_id]))
                    #agents_info.update(pd.Series(['PV_' + self.unique_id],  name  = 'Individual_ID', index = [self.unique_id]))
                    #agents_info.update(pd.Series([self.total],      name  = 'intention',    index = [self.unique_id]))
                    #agents_info.update(pd.Series([ind_npv],         name  = 'Ind_NPV',      index = [self.unique_id]))
                    #agents_info.update(pd.Series(["Only_Ind"],      name  = 'Reason',       index = [self.unique_id]))
                    
                    agents_info.at[self.unique_id,'Year'] = 2018 + step_ctr
                    agents_info.at[self.unique_id,'Adopt_IND'] = 1
                    agents_info.at[self.unique_id,'Individual_ID'] = 'PV_' + self.unique_id
                    agents_info.at[self.unique_id,'intention'] = self.total
                    agents_info.at[self.unique_id,'Reason'] = "Only_Ind"
                    agents_info.at[self.unique_id,'Ind_NPV'] = ind_npv
                                
                    #do not set this back to 0 anymore as agents continue to be in the ABM even if they have installed PV
                    #self.intention  = 0
                    #self.adopt_comm = 0
                
            elif len(Combos_Info.index) == 0:
            #meaning that NO community is formed, hence go for individual PV adoption
                if Agents_Ind_NPVs.at[step_ctr,self.unique_id] >= reduction*Agents_Ind_Investment_Costs.at[step_ctr, self.unique_id]:
                    #adopt individual
                    #set adoption as 1 for an individual PV formation
                    self.adopt_ind  = 1
                    self.adopt_year = 2018 + step_ctr
                    ind_npv         = Agents_Ind_NPVs.at[step_ctr,self.unique_id]
                    #agents_info.update(pd.Series([1],               name  = 'Adopt_IND',    index = [self.unique_id]))
                    #agents_info.update(pd.Series([2018+step_ctr],   name  = 'Year',         index = [self.unique_id]))
                    #agents_info.update(pd.Series(['PV_' + self.unique_id],  name  = 'Individual_ID', index = [self.unique_id]))
                    #agents_info.update(pd.Series([self.total],      name  = 'intention',    index = [self.unique_id]))
                    #agents_info.update(pd.Series([ind_npv],         name  = 'Ind_NPV',      index = [self.unique_id]))
                    #agents_info.update(pd.Series(["Only_Ind"],      name  = 'Reason',       index = [self.unique_id]))
                    agents_info.at[self.unique_id,'Year'] = 2018 + step_ctr
                    agents_info.at[self.unique_id,'Adopt_IND'] = 1
                    agents_info.at[self.unique_id,'Individual_ID'] = 'PV_' + self.unique_id
                    agents_info.at[self.unique_id,'intention'] = self.total
                    agents_info.at[self.unique_id,'Reason'] = "Only_Ind"
                    agents_info.at[self.unique_id,'Ind_NPV'] = ind_npv
                    
                    
        elif len(same_plot_agents_positive_intention.index) == 0 and self.adopt_ind != 1:
            #meaning that NO community is formed, hence go for individual PV adoption
            #only if PV is not previously installed
            if Agents_Ind_NPVs.at[step_ctr,self.unique_id] >= reduction*Agents_Ind_Investment_Costs.at[step_ctr,self.unique_id]:
                #adopt individual
                #set adoption as 1 for an individual PV formation
                self.adopt_ind  = 1
                self.adopt_year = 2018 + step_ctr
                ind_npv         = Agents_Ind_NPVs.at[step_ctr,self.unique_id]
                #agents_info.update(pd.Series([1],               name  = 'Adopt_IND',    index = [self.unique_id]))
                #agents_info.update(pd.Series([2018+step_ctr],   name  = 'Year',         index = [self.unique_id]))
                #agents_info.update(pd.Series(['PV_' + self.unique_id],  name  = 'Individual_ID', index = [self.unique_id]))
                #agents_info.update(pd.Series([self.total],      name  = 'intention',    index = [self.unique_id]))
                #agents_info.update(pd.Series([ind_npv],         name  = 'Ind_NPV',      index = [self.unique_id]))
                #agents_info.update(pd.Series(["Only_Ind"],      name  = 'Reason',       index = [self.unique_id]))
                agents_info.at[self.unique_id,'Year'] = 2018 + step_ctr
                agents_info.at[self.unique_id,'Adopt_IND'] = 1
                agents_info.at[self.unique_id,'Individual_ID'] = 'PV_' + self.unique_id
                agents_info.at[self.unique_id,'intention'] = self.total
                agents_info.at[self.unique_id,'Reason'] = "Only_Ind"
                agents_info.at[self.unique_id,'Ind_NPV'] = ind_npv
                    
