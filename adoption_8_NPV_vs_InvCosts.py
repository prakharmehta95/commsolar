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
import json        
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

#INFORMATION on agents and all combinations
agents_info = pd.read_excel(r'C:\Users\iA\Dropbox\Com_Paper\05_Data\01_CEA_Disaggregated\02_Buildings_Info\Bldgs_Info.xlsx')
df_demand = pd.read_pickle(r'C:\Users\iA\Dropbox\Com_Paper\05_Data\01_CEA_Disaggregated\00_Demand_Disagg\CEA_Disaggregated_Demand_TOTAL_FINAL_30Nov.pickle')
df_solar = pd.read_pickle(r'C:\Users\iA\Dropbox\Com_Paper\05_Data\01_CEA_Disaggregated\01_PV_Disagg\CEA_Disaggregated_SolarPV_30Nov.pickle')
df_solar = df_solar*0.97 #converting solar PV DC output to AC 

df_solar_combos = pd.DataFrame(data = None)
df_demand_combos= pd.DataFrame(data = None)
        

#NPV information of individual agents
from __main__ import Agents_NPVs as Agents_Ind_NPVs
from __main__ import Agents_SCRs as Agents_Ind_SCRs
from __main__ import Agents_Investment_Costs as Agents_Ind_Investment_Costs 
from __main__ import Agents_Peer_Network as Agents_Peers


#payback periods for individual agents normalized
from __main__ import Agents_PPs_Norm as Agents_Ind_PPs_Norm


list_agents = list(agents_info.bldg_name)

PV_already_installed = agents_info[agents_info['pv_already_installed_size_kW']>0]
list_installed_solar_bldgs = PV_already_installed.bldg_name

#creating new columns to hold information on the agents
agents_info['intention'] = 0
agents_info['Comm_NPV'] = 0
agents_info['Ind_NPV'] = 0
agents_info['Reason'] = ""
agents_info['Ind_SCR'] = 0
agents_info['Comm_SCR'] = 0
agents_info['Adopt_IND'] = 0        #saves 1 if INDIVIDUAL adoption occurs, else stays 0
agents_info['Adopt_COMM'] = 0       #saves 1 if COMMUNITY  adoption occurs, else stays 0
agents_info['En_Champ'] = 0         #saves who is the energy champion of that community
agents_info['Adopt_Year'] = 0             #saves year of adoption
agents_info['Community_ID'] = ""    #community ID of the community formed
agents_info['bldg_names'] = ""
agents_info['bldg_names'] = agents_info['bldg_name']
agents_info = agents_info.set_index('bldg_names')

#agents_subplots = subplots_final

#%% AGENT code

c = 0
step_ctr = 0
#agents_info['intention'] = ""
#agents_info.set_index('bldg_id')

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
            peer_effect   = ratio of people in small world network who install solar [0,1]
            total       = sum of the intention, from the stage1_intention function [0,1]
            counter     = to know number of times the agent crosses intention. Initialize to 0, if 1 then go to stage2,else do not go to stage2
                            also - if 0 or >1, then do not count in subplot effects!
            adopt_ind   = if agent adopts individual system BOOLEAN 0|1
            adopt_comm  = if agent adopts community system BOOLEAN 0|1
            en_champ    = if this agent is the energy champion or nor 
        
        '''
        super().__init__(unique_id,model) 
        
        self.bldg_type = bldg_type
        self.bldg_own = bldg_own
        self.bldg_zone = bldg_zone
        self.bldg_plot = bldg_plot
        
        self.attitude = attitude                        #environmental attitude
        self.pp = pp                                    #perceived profitability
        self.peer_effect = peer_effect                  #peer effect - calculated in the step_idea later so just add a placeholder here for agent initialization?
        self.neighbor_influence = neighbor_influence    #stores the neighbor_influence, called with the 'check_neighbours_subplots' function in step_idea(self)
        
        self.total = total                      #sum of the intention function
        self.counter = counter
        self.intention = intention              #stores the intention of the agent (whether or not it passed idea stage)
        self.intention_yr = intention_yr        #stores the intention of the agent every year of the simulation
        self.adopt_ind = adopt_ind
        self.adopt_comm = adopt_comm
        self.adopt_year = adopt_year
        self.en_champ = en_champ
        self.pv_size = pv_size
        self.dem_total = dem_total
        #self.egids = egids
        
          
               
    def step_idea(self):
        '''
        defines what the agent does in his step.
        StagedActivation is used now, this is the first stage.
        IDEA/INTENTION developments happens in this case
        '''
        
        # only run the step_idea if agent has not already adopted!- NEED TO CHANGE THIS TYPE OF REASONING - NOW DEPENDS ON STATE OF THE AGENT
        if self.adopt_comm == 1 or self.adopt_ind == 1:
            self.intention = 0                          #do it again to be safe - so that an agent which adopted is out of the simulation
        else:
            #print("start of new agent uid = ",self.unique_id, "has intention = ",self.intention)
            functions.check_peers(self,self.unique_id)                            #sets the peer_effect
            functions.check_neighbours_subplots(self,self.unique_id)              #sets the neighbor_influence
            self.pp = functions.econ_attr(self)
            
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
            #call the decision making function i.e. stage 2 of the ABM simulation
            stage2_decision(self,self.unique_id,self.intention)
            
        
#%% MODEL code
        
        
class tpb(Model):
    '''
    Model setup. The main ABM runs from here.
    Called the FIRST time when the execution happens. Agents are initiated
    '''    
    
    global list_datacollector
    list_datacollector = []
    c = 0
    
    
    def __init__(self,N,randomseed):
        print("--TPB MODEL CALLED--")
        agents_info['Adopt_IND'] = 0            #saves 1 if INDIVIDUAL adoption occurs, else stays 0
        agents_info['Adopt_COMM'] = 0           #saves 1 if COMMUNITY  adoption occurs, else stays 0
        agents_info['Community_ID'] = ""        #community ID of the community formed
        agents_info['Year'] = 0
        
        #Agents_Possibles_Combos['Adopt'] = 0    #saves 1 if COMMUNITY adoption occurs, else stays 0
        #Agents_Possibles_Combos['Year'] = 0     #saves year of adoption

        
        super().__init__()
        self.num_agents = N
        self.randomseed = randomseed
        self.schedule = StagedActivation_random(self,stage_list = ['step_idea','step_decision'],
                                             shuffle = True, shuffle_between_stages = True,seed = self.randomseed) 
        #only called once so the seed is the same, meaning the agents are shuffled the same way every time
        
        
        global agents_objects_list
        agents_objects_list = []            #list of initialized agents
        
        
        global norm_dist_env_attitude_list
        global step_ctr
        step_ctr = 0
        global i
        for i in list_agents:
            
            global c#, intention
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
            
            #agent creation
            a = tpb_agent(agents_info.loc[i]['bldg_name'],self,agents_info.loc[i]['bldg_type'],agents_info.loc[i]['bldg_owner'],
                          agents_info.loc[i]['zone_id'],agents_info.loc[i]['plot_id'],
                          functions.env_attitude(agents_info.loc[i]['IDs']),functions.econ_attr(self),
                          intention,intention_yr,ratio,neighbor_influence,total,counter,
                          adopt_ind,adopt_comm,adopt_year,en_champ,agents_info.loc[i]['pv_size_kw'],
                          agents_info.loc[i]['demand_yearly_kWh'])
            
            agents_objects_list.append(a)
            self.schedule.add(a)
            c = c + 1
        

        
        #data collection
        self.datacollector = DataCollector(model_reporters = {"Individual_solar":functions.cumulate_solar_ind,"Ind_PV_Installed":functions.cumulate_solar_ind_sizes,
                                                              "Community_solar":functions.cumulate_solar_comm,"Energy_Champions":functions.cumulate_solar_champions,
                                                              "Comm_PV_Installed":functions.cumulate_solar_comm_sizes,
                                                              "Agent_Type_Res":functions.agent_type_res,"Agent_Type_Res_PV_Size":functions.agent_type_res_CAP,
                                                              "Agent_Type_Comm":functions.agent_type_comm,"Agent_Type_Comm_PV_Size":functions.agent_type_comm_CAP,
                                                              "Agent_Type_Pub":functions.agent_type_pub,"Agent_Type_Pub_PV_Size":functions.agent_type_pub_CAP,
                                                              "EGIDs_greater_than_one":functions.zone_egids,"EGIDs_greater_than_one_SIZE":functions.zone_egids_size},
                agent_reporters = {"Building_ID":"unique_id","Building_Type":"bldg_type","Part_Comm":"part_comm",
                                   "PV_Size":"pv_size","Demand_MWhyr":"dem_total","Intention" : "intention",
                                   "Yearly_Intention":"intention_yr","Attitude" : "attitude","Payback Period" : "pp",
                                   "SWN Ratio":"peer_effect","Intention_Sum": "total","Subplot Effect":"neighbor_influence",
                                   "Individual":"adopt_ind","Community":"adopt_comm","Year_Adoption":"adopt_year","Energy_Champion":"en_champ"})
        
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
    """

    def agent_type_res(model):
        '''
        to find total number of RESIDENTIAL individual adoptions
        '''
        sum_agent_type_res_ind = 0
        for i in model.schedule.agents:
            if i.bldg_type == 'Residential':
                sum_agent_type_res_ind = sum_agent_type_res_ind + i.adopt_ind
        return sum_agent_type_res_ind
    
    def agent_type_res_CAP(model):
        '''
        to find total CAPACITY of RESIDENTIAL individual adoptions
        '''
        sum_agent_type_res_ind_cap = 0
        for i in model.schedule.agents:
            if i.bldg_type == 'Residential' and i.adopt_ind == 1:
                sum_agent_type_res_ind_cap = sum_agent_type_res_ind_cap + i.pv_size
        return sum_agent_type_res_ind_cap
        
    
    def agent_type_comm(model):
        '''
        to find total number of COMMERCIAL individual adoptions
        '''
        sum_agent_type_comm_ind = 0
        for i in model.schedule.agents:
            if i.bldg_type == 'Commercial':
                sum_agent_type_comm_ind = sum_agent_type_comm_ind + i.adopt_ind
        return sum_agent_type_comm_ind
    
    
    def agent_type_comm_CAP(model):
        '''
        to find total CAPACITY of COMMERCIAL individual adoptions
        '''
        sum_agent_type_comm_ind_cap = 0
        for i in model.schedule.agents:
            if i.bldg_type == 'Commercial' and i.adopt_ind == 1:
                sum_agent_type_comm_ind_cap = sum_agent_type_comm_ind_cap + i.pv_size
        return sum_agent_type_comm_ind_cap
    
    def agent_type_pub(model):
        '''
        to find total number of PUBLIC individual adoptions
        '''
        sum_agent_type_pub_ind = 0
        for i in model.schedule.agents:
            if i.bldg_type == 'Public':
                sum_agent_type_pub_ind = sum_agent_type_pub_ind + i.adopt_ind
        return sum_agent_type_pub_ind
    
    def agent_type_pub_CAP(model):
        '''
        to find total CAPACITY of PUBLIC individual adoptions
        '''
        sum_agent_type_pub_ind_cap = 0
        for i in model.schedule.agents:
            if i.bldg_type == 'Public' and i.adopt_ind == 1:
                sum_agent_type_pub_ind_cap = sum_agent_type_pub_ind_cap + i.pv_size
        return sum_agent_type_pub_ind_cap
    
    def zone_egids(model):
        '''
        to know if an individual adoption is actually like a community because of the CEA ZONES
        '''
        sum_EGID = 0
        for i in model.schedule.agents:
            if i.egids > 1:
                sum_EGID = sum_EGID + i.adopt_ind
        return sum_EGID
    
    def zone_egids_size(model):
        '''
        to know if an individual adoption is actually like a community because of the CEA ZONES, getting SIZES
        '''
        sum_EGID = 0
        for i in model.schedule.agents:
            if i.egids > 1 and i.adopt_ind == 1:
                sum_EGID = sum_EGID + i.pv_size
        return sum_EGID
    
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
        """
        solar_sum_champ = 0
        for i in model.schedule.agents:
            solar_sum_champ = solar_sum_champ + i.en_champ
        return solar_sum_champ        
    
    def cumulate_solar_ind_sizes(model):
        """
        To find the cumulative COMMUNITY solar capacity at the end of every time step
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
    
    
    def env_attitude(uid):#(self, uid, attitude):
        """
        To change the environmental attitude depending on the steps.
        NOT TO BE CHANGED IN MODEL v1.0!
        """
        uid = uid
        if agents_info.loc[uid]['MINERGIE'] == 1:
            value = 0.95
        else:
            value = gauss(0.698,0.18)
        if value > 1:
            value = 1 #to keep it from going over 1
        return value
    
    
    def econ_attr(self):
        """
        To update the payback ratio every step
        Takes data from the profitability_index pickle
        """
        uid = self.unique_id
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
        temp_list = Agents_Peers[uid].values.tolist() #make a list of all peers of that particular agent with uid
        
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
        
        #since in the last year if the intention is 0, stage 2 will not be entered. Results won't be written, hence write them here.
        if step_ctr == 17:
            agents_info.update(pd.Series([0], name  = 'Adopt_IND', index = [self.unique_id]))
            agents_info.update(pd.Series([self.total], name  = 'intention', index = [self.unique_id]))
            agents_info.update(pd.Series(["Intention<threshold"], name  = 'Reason', index = [self.unique_id]))

    

def stage2_decision(self,uid,idea):
    """
    Decision made here after passing the intention stage
    """
    
    #-------------------------------------------new try here for combinatorics----------------------------------------------------------------
    if self.intention == 1:
        temp_plot_id = agents_info.loc[self.unique_id]['plot_id']
        same_plot_agents = agents_info[agents_info['plot_id']==temp_plot_id]
        same_plot_agents_positive_intention = same_plot_agents[same_plot_agents['intention'] == 1 or same_plot_agents['adoption'] == 1] #available to form community
        #only agents without solar will have the intention variable as '1'. If an agent has individual/community PV then intention is always '0', but adoption will be '1'
        
        #even read the agents which have already formed a community or installed PV - how? take all agents with positive intention AND those which have formed PV...
        #2 types of agents already formed PV - individual and community
        
        
        #check for community formation 
        from community_combos import community_combinations
        Combos_Info, NPV_Combos, df_solar_combos_possible, df_demand_combos_possible, comm_name = community_combinations(data, distances, df_solar, df_demand, df_solar_combos, df_demand_combos)
        
        #keeping info of the community formed as it is needed later in case an agent wants to join a particular community
        temp_comm_name = 'C_' + comm_name
        df_solar_combos[temp_comm_name] = df_solar_combos_possible[comm_name] 
        df_demand_combos= df_demand_combos_possible[comm_name]

        if len(Combos_Info.index) != 0: #meaning that some community is formed
            #here compare with individual NPVs
            if Agents_Ind_NPVs.loc[self.unique_id]['npv'] < Combos_Info.loc[temp_comm_name]['npv_share_en_champ'] and Combos_Info.loc[temp_comm_name]['npv_share_en_champ'] > 0:
                #form a community
                #set the adoption as 1 for all the constituent  buildings
                #set some variable which indicates whether it is a community or an individual PV system
                pass
            elif Agents_Ind_NPVs.loc[self.unique_id]['npv'] >= Combos_Info.loc[temp_comm_name]['npv_share_en_champ'] and Agents_Ind_NPVs.loc[self.unique_id]['npv'] > 0:
                #adopt individual
                pass
        elif len(Combos_Info.index) != 0: #meaning that NO community is formed, hence go for individual PV adoption
            if Agents_Ind_NPVs.loc[self.unique_id]['npv'] >=0:
                #adopt individual
                #set adoption as 1 for an individual PV formation
            pass
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    #-------------------------------------------new try until here for combinatorics----------------------------------------------------------
    
    
    
    #print("STAGE 2 ENTERED...............................................")
    
    #if scenario == "ZEV" or scenario == "no_ZEV":
     #   reduction = -0.05    #how much negative NPV as a percentage of investment is allowed. 5% negative NPV allowed after calibration.
            
    #elif scenario == "TOP4_no100MWh_retail" or scenario == "TOP4_no100MWh_wholesale":
     #   reduction = -0.00    #how much negative NPV as a percentage of investment is allowed. NO NEGATIVE NPV allowed after calibration.
    
    agents_adopting_community_list = []
    ind_npv = Agents_Ind_NPVs.loc[step_ctr][self.unique_id]
    
    if self.intention == 1 and (self.unique_id not in list_installed_solar_bldgs_ALL):
        '''
        let's say building A has passed
        it checks all the possible options
        starting from the best NPV, it checks if the other buildings have positive intention as well
        then it takes best NPV 
        --> if that is individual, then cool, end of case, adopt individual
        --> else if it involves other buildings then check if their ranks are among top 3, if yes then already include them as
            the community adoption.
            WHEN those buildings will be the agents making decisions, then if they have already said yes to a community system,
            then do not do the ranking etc for them, simply skip them... :)
        '''
        
        # to set which building/agent is being considered
        temp_id = self.unique_id
        #find the npv to save it with the agent when he adopts
        
        #for the agents which can actually form communities
        if self.part_comm == 1 and ZEV == 1:
            
            temp_agent_info_list_one = []
            temp_agent_info_list_two = []
            for i in range(len(agents_objects_list)):
                temp_agent_info_list_one.append(agents_objects_list[i].unique_id)
                temp_agent_info_list_two.append(agents_objects_list[i].intention_yr)
            
            temp_agent_info_df = pd.DataFrame(data = None) #temporary dataframe to hold agent information about intention in that year of simulation
            temp_agent_info_df['unique_id'] = temp_agent_info_list_one
            temp_agent_info_df['intention_yr'] = temp_agent_info_list_two
            temp_agent_info_df.set_index('unique_id')
            
            all_positive_intention_agents = temp_agent_info_df[temp_agent_info_df['intention_yr'] == 1] #filter out agents with positive intention
            list_all_positive_intention_agents = all_positive_intention_agents.unique_id.tolist() #create a list of positive intention agents
            pos_intent_comm_members = []
            
            possible_community_members_all = subplots_final[temp_id].dropna().values.tolist() #list of neighbours who can form a ZEV
            
            #drop those which have already adopted solar!
            possible_community_members = []
            for k in possible_community_members_all:
                for j in range(len(agents_objects_list)):
                    if k == agents_objects_list[j].unique_id:
                        if  agents_objects_list[j].adopt_ind == 1 or agents_objects_list[j].adopt_comm == 1:
                            continue #do nothing 
                        elif agents_objects_list[j].adopt_ind == 0 and agents_objects_list[j].adopt_comm == 0:
                            possible_community_members.append(k) #if not adopted already, put in a list
           
            for k in possible_community_members:
                if k in list_all_positive_intention_agents:
                    pos_intent_comm_members.append(k) #community members with positive intentions who can potentially form a community
            
            temp_pos_combos_list = []
            names_possible_combos_list = []
            
            #making the possible combinations with the agents who passed intention and who can form the community 
            for j in range(0,len(pos_intent_comm_members)):
                temp_pos_combos = list(itertools.combinations(pos_intent_comm_members,len(pos_intent_comm_members)-j))
                temp_pos_combos_list.append(temp_pos_combos)
                
                for k in range(len(temp_pos_combos_list)):
                    for m in temp_pos_combos_list[k]:
                        temp_name = temp_id #take name of building in consideration here, example = 'Z0054'
                        for n in m:
                            temp_name = temp_name + '_' +n
                        names_possible_combos_list.append(temp_name) #all possible combos which can be formed this year. Example: ''Z0054_Z0055' 
            
            
            temp_combos_exist_df = pd.DataFrame(data = None,index = range(1)) #dataframe to hold which combos can exist and which have positive NPVs
            
            #for loop to find which of the combos can actually exist and which have positive NPVs
            for g in names_possible_combos_list:
                if g in list(Agents_Community_NPVs.columns):
                    if Agents_Community_NPVs.loc[step_ctr][g] > 0:
                        temp_combos_exist_df[g] = ""
                        temp_combos_exist_df.update(pd.Series([Agents_Community_NPVs.loc[step_ctr][g]], name  = g, index = [0]))
                        
            npvs_list = pd.DataFrame(data = None)
            npvs_list = temp_combos_exist_df.loc[0,:] #take the 1st row of the temporary temp_combos_exist_df dataframe as it contains all NPVs of all possible combinations
            npvs_list2 = pd.to_numeric(npvs_list)#, errors = 'coerce', axis=0)
            
            #check if there are communities which can be formed after all the filtering I have done previously
            if len(npvs_list) != 0:
                max_combo_npv = npvs_list2.idxmax(axis=1) #this is the community with the maximum positive NPV!
                                
                #*****NOW COMES THE ACTUAL COMMUNITY FORMATIONS*****
                
                #ZEV and no_ZEV scenario - only case where a community formation can be considered
                if Agents_Community_NPVs.loc[step_ctr][max_combo_npv] > 0 and comm_limit == 1 and Agents_Possibles_Combos.loc[max_combo_npv]['Demand_Year_MWh'] < 100:#) if scen_11 == 1 else True): #extra if is to neglect or consider community sizes
                    #if the npv is greater than 0 and if the demand sum of the buildings is less than 100 MWh, the self imposed limit
                    #print("Commmunty NPV being considered")
                    #compare with individual NPV and share of NPV here
                    temp_bb = json.loads(Agents_Possibles_Combos.loc[max_combo_npv]['Bldg_share_in_community'])
                    fraction_npv = temp_bb[0]
                    share_npv = fraction_npv*Agents_Community_NPVs.loc[step_ctr][max_combo_npv]
                    ind_npv = Agents_Ind_NPVs.loc[step_ctr][self.unique_id]
                    if scenario == "ZEV":
                        comm_scr = Agents_Community_SCRs.loc[max_combo_npv]['Year0']
                    else:
                        comm_scr = Agents_Community_SCRs.loc['Year0'][max_combo_npv]
                    ind_scr = Agents_Ind_SCRs.loc[self.unique_id]['Year0']
                    #print("Comparison:share =",share_npv, "vs ind = ",ind_npv)
                    if share_npv >= ind_npv: #shared npv better than individual npv so community adoption will happen
                        Agents_Possibles_Combos.update(pd.Series([1], name  = 'Adopt', index = [max_combo_npv]))
                        Agents_Possibles_Combos.update(pd.Series([2018+step_ctr], name  = 'Year', index = [max_combo_npv]))
                        #split into the names of agents
                        agents_adopting_comm = max_combo_npv.split("_")  
                        comm_id = Agents_Possibles_Combos.loc[max_combo_npv]['Comm_ID']
                        
                        #setting adoption as 1 for all the buildings involved in the community formation
                        for g in agents_adopting_comm:
                            if g == self.unique_id:
                                self.en_champ = 1 #setting the agent which is the energy champion - the first agent
                                agents_info.update(pd.Series([self.en_champ], name  = 'En_Champ', index = [self.unique_id]))
                            for h in range(len(agents_objects_list)):
                                if g == agents_objects_list[h].unique_id:
                                    agents_objects_list[h].adopt_comm = 1 #setting community adoption as 1 for all agents involved
                                    self.adopt_year = 2018 + step_ctr
                                    agents_objects_list[h].intention = 0 #setting intention as 0 for all agents involved
                                    agents_info.update(pd.Series([1], name  = 'Adopt_COMM', index = [g]))
                                    agents_info.update(pd.Series([2018+step_ctr], name  = 'Year', index = [g]))
                                    agents_info.update(pd.Series([comm_id], name  = 'Community_ID', index = [g]))
                                    agents_info.update(pd.Series([self.total], name  = 'intention', index = [g]))
                                    agents_info.update(pd.Series([share_npv], name  = 'Comm_NPV', index = [g]))
                                    agents_info.update(pd.Series([ind_npv], name  = 'Ind_NPV', index = [g]))
                                    agents_info.update(pd.Series(["Comm>Ind"], name  = 'Reason', index = [g]))
                                    agents_info.update(pd.Series([ind_scr], name  = 'Ind_SCR', index = [g]))
                                    agents_info.update(pd.Series([comm_scr], name  = 'Comm_SCR', index = [g]))
                        agents_adopting_community_list.append(agents_adopting_comm)
                        
                    
                    elif share_npv < ind_npv and ind_npv>reduction*Agents_Ind_Investments.loc[step_ctr][self.unique_id]:#0: #if the individual NPV is positive and better than shared NPV
                        Agents_Possibles_Combos.update(pd.Series([0], name  = 'Adopt', index = [max_combo_npv]))
                        self.adopt_ind = 1
                        self.adopt_year = 2018 + step_ctr
                        agents_info.update(pd.Series([1], name  = 'Adopt_IND', index = [self.unique_id]))
                        agents_info.update(pd.Series([2018+step_ctr], name  = 'Year', index = [self.unique_id]))
                        agents_info.update(pd.Series([self.total], name  = 'intention', index = [self.unique_id]))
                        agents_info.update(pd.Series([ind_npv], name  = 'Ind_NPV', index = [self.unique_id]))
                        agents_info.update(pd.Series(["Ind>Comm"], name  = 'Reason', index = [self.unique_id]))
                    
                    elif share_npv < ind_npv and ind_npv<reduction*Agents_Ind_Investments.loc[step_ctr][self.unique_id]:#0: #if the individual NPV is negative but better than shared NPV
                        Agents_Possibles_Combos.update(pd.Series([0], name  = 'Adopt', index = [max_combo_npv]))
                        self.adopt_ind = 0
                        agents_info.update(pd.Series([0], name  = 'Adopt_IND', index = [self.unique_id]))
                        agents_info.update(pd.Series([self.total], name  = 'intention', index = [self.unique_id]))
                        agents_info.update(pd.Series([ind_npv], name  = 'Ind_NPV', index = [self.unique_id]))
                        agents_info.update(pd.Series(["Ind_negative>Comm_negative"], name  = 'Reason', index = [self.unique_id]))
                        
                
                #ZEV and no_ZEV scenario but because of high demand, the community cannot be formed
                elif Agents_Community_NPVs.loc[step_ctr][max_combo_npv] > 0 and comm_limit == 1 and Agents_Possibles_Combos.loc[max_combo_npv]['Demand_Year_MWh'] >= 100: #and scen_11 == 1:
                    #individual will adopt as community cannot
                    ind_npv = Agents_Ind_NPVs.loc[step_ctr][self.unique_id]
                    if ind_npv>reduction*Agents_Ind_Investments.loc[step_ctr][self.unique_id]:#0:
                        self.adopt_ind = 1
                        self.adopt_year = 2018 + step_ctr
                        agents_info.update(pd.Series([1], name  = 'Adopt_IND', index = [self.unique_id]))
                        agents_info.update(pd.Series([2018+step_ctr], name  = 'Year', index = [self.unique_id]))
                        agents_info.update(pd.Series([self.total], name  = 'intention', index = [self.unique_id]))
                        agents_info.update(pd.Series([ind_npv], name  = 'Ind_NPV', index = [self.unique_id]))
                        agents_info.update(pd.Series(["Demand_No_Comm_Possible"], name  = 'Reason', index = [self.unique_id]))
                    else:
                        self.adopt_ind = 0
                        agents_info.update(pd.Series([0], name  = 'Adopt_IND', index = [self.unique_id]))
                        agents_info.update(pd.Series([self.total], name  = 'intention', index = [self.unique_id]))
                        agents_info.update(pd.Series([ind_npv], name  = 'Ind_NPV', index = [self.unique_id]))
                        agents_info.update(pd.Series(["Demand_No_Comm_Possible & Ind_Negative"], name  = 'Reason', index = [self.unique_id]))
                        #agents_info.update(pd.Series([2018+step_ctr], name  = 'Year', index = [self.unique_id]))
                    #community will not adopt
                    Agents_Possibles_Combos.update(pd.Series([0], name  = 'Adopt', index = [max_combo_npv]))
                    agents_adopting_comm = max_combo_npv.split("_")
                    comm_id = Agents_Possibles_Combos.loc[max_combo_npv]['Comm_ID']
                    for g in agents_adopting_comm:
                        #print(g)
                        for h in range(len(agents_objects_list)):
                            if g == agents_objects_list[h].unique_id:
                                agents_objects_list[h].adopt_comm = 0
                                agents_info.update(pd.Series([0], name  = 'Adopt_COMM', index = [g]))
                                #agents_info.update(pd.Series([comm_id], name  = 'Community_ID', index = [g]))
                    #Agents_Possibles_Combos.loc[max_combo_npv]['Adopt'] = 0
                
                #TOP4_no_100MWh scenario - no restriction based on 100MWh demand limit
                elif Agents_Community_NPVs.loc[step_ctr][max_combo_npv] > 0 and comm_limit == 0:
                    #if the npv is greater than 0 and if the demand sum of the buildings is less than 100 MWh, the self imposed limit
                    
                    #compare with individual NPV and share of NPV here
                    temp_bb = json.loads(Agents_Possibles_Combos.loc[max_combo_npv]['Bldg_share_in_community'])
                    fraction_npv = temp_bb[0]
                    share_npv = fraction_npv*Agents_Community_NPVs.loc[step_ctr][max_combo_npv]
                    ind_npv = Agents_Ind_NPVs.loc[step_ctr][self.unique_id]
                    #print("Comparison:share =",share_npv, "vs ind = ",ind_npv)
                    if scenario == "ZEV":
                        comm_scr = Agents_Community_SCRs.loc[max_combo_npv]['Year0']
                    else:
                        comm_scr = Agents_Community_SCRs.loc['Year0'][max_combo_npv]
                    ind_scr = Agents_Ind_SCRs.loc[self.unique_id]['Year0']
                    if share_npv >= ind_npv: #shared npv better than individual npv so community adoption will happen
                        Agents_Possibles_Combos.update(pd.Series([1], name  = 'Adopt', index = [max_combo_npv]))
                        Agents_Possibles_Combos.update(pd.Series([2018+step_ctr], name  = 'Year', index = [max_combo_npv]))
                        #split into the names of agents
                        agents_adopting_comm = max_combo_npv.split("_")  
                        comm_id = Agents_Possibles_Combos.loc[max_combo_npv]['Comm_ID']
                        
                        #setting adoption as 1 for all the buildings involved in the community formation
                        for g in agents_adopting_comm:
                            if g == self.unique_id:
                                self.en_champ = 1 #setting the agent which is the energy champion - the first agent
                                agents_info.update(pd.Series([self.en_champ], name  = 'En_Champ', index = [self.unique_id]))
                            for h in range(len(agents_objects_list)):
                                if g == agents_objects_list[h].unique_id:
                                    agents_objects_list[h].adopt_comm = 1 #setting community adoption as 1 for all agents involved
                                    self.adopt_year = 2018 + step_ctr
                                    agents_objects_list[h].intention = 0 #setting intention as 0 for all agents involved
                                    agents_info.update(pd.Series([1], name  = 'Adopt_COMM', index = [g]))
                                    agents_info.update(pd.Series([2018+step_ctr], name  = 'Year', index = [g]))
                                    agents_info.update(pd.Series([comm_id], name  = 'Community_ID', index = [g]))
                                    agents_info.update(pd.Series([self.total], name  = 'intention', index = [g]))
                                    agents_info.update(pd.Series([share_npv], name  = 'Comm_NPV', index = [g]))
                                    agents_info.update(pd.Series([ind_npv], name  = 'Ind_NPV', index = [g]))
                                    agents_info.update(pd.Series(["Comm>Ind"], name  = 'Reason', index = [g]))
                                    agents_info.update(pd.Series([ind_scr], name  = 'Ind_SCR', index = [g]))
                                    agents_info.update(pd.Series([comm_scr], name  = 'Comm_SCR', index = [g]))
                        agents_adopting_community_list.append(agents_adopting_comm)
                        
                    
                    elif share_npv < ind_npv and ind_npv>=reduction*Agents_Ind_Investments.loc[step_ctr][self.unique_id]: #if the individual NPV is positive and better than shared NPV
                        Agents_Possibles_Combos.update(pd.Series([0], name  = 'Adopt', index = [max_combo_npv]))
                        self.adopt_ind = 1
                        self.adopt_year = 2018 + step_ctr
                        agents_info.update(pd.Series([1], name  = 'Adopt_IND', index = [self.unique_id]))
                        agents_info.update(pd.Series([2018+step_ctr], name  = 'Year', index = [self.unique_id]))
                        agents_info.update(pd.Series([self.total], name  = 'intention', index = [self.unique_id]))
                        agents_info.update(pd.Series([ind_npv], name  = 'Ind_NPV', index = [self.unique_id]))
                        agents_info.update(pd.Series(["Ind>Comm"], name  = 'Reason', index = [self.unique_id]))
                    elif share_npv < ind_npv and ind_npv<reduction*Agents_Ind_Investments.loc[step_ctr][self.unique_id]:#0: #if the individual NPV is negative but better than shared NPV
                        Agents_Possibles_Combos.update(pd.Series([0], name  = 'Adopt', index = [max_combo_npv]))
                        self.adopt_ind = 0
                        agents_info.update(pd.Series([0], name  = 'Adopt_IND', index = [self.unique_id]))
                        agents_info.update(pd.Series([self.total], name  = 'intention', index = [self.unique_id]))
                        agents_info.update(pd.Series([ind_npv], name  = 'Ind_NPV', index = [self.unique_id]))
                        agents_info.update(pd.Series(["Ind_negative>Comm_negative"], name  = 'Reason', index = [self.unique_id]))
                
                #because of negative npv, the community cannot be formed
                elif Agents_Community_NPVs.loc[step_ctr][max_combo_npv] < 0:
                    
                    Agents_Possibles_Combos.update(pd.Series([0], name  = 'Adopt', index = [max_combo_npv]))
                    agents_adopting_comm = max_combo_npv.split("_")
                    comm_id = Agents_Possibles_Combos.loc[max_combo_npv]['Comm_ID']
                    
                    #form individual if npv > 0
                    if Agents_Ind_NPVs.loc[step_ctr][self.unique_id] > reduction*Agents_Ind_Investments.loc[step_ctr][self.unique_id]:#0:
                        self.adopt_ind = 1
                        self.adopt_year = 2018 + step_ctr
                        agents_info.update(pd.Series([1], name  = 'Adopt_IND', index = [self.unique_id]))
                        agents_info.update(pd.Series([2018+step_ctr], name  = 'Year', index = [self.unique_id]))
                        agents_info.update(pd.Series([self.total], name  = 'intention', index = [self.unique_id]))
                        agents_info.update(pd.Series([ind_npv], name  = 'Ind_NPV', index = [self.unique_id]))
                        agents_info.update(pd.Series(["Ind>Negative_Comm"], name  = 'Reason', index = [self.unique_id]))
                        self.intention = 0
                        self.adopt_comm = 0
                    elif Agents_Ind_NPVs.loc[step_ctr][self.unique_id] < reduction*Agents_Ind_Investments.loc[step_ctr][self.unique_id]:#0:
                        self.adopt_ind = 0
                        agents_info.update(pd.Series([0], name  = 'Adopt_IND', index = [self.unique_id]))
                        agents_info.update(pd.Series([self.total], name  = 'intention', index = [self.unique_id]))
                        agents_info.update(pd.Series([ind_npv], name  = 'Ind_NPV', index = [self.unique_id]))
                        agents_info.update(pd.Series(["Negative_Ind & Negative_Comm"], name  = 'Reason', index = [self.unique_id]))
                        self.intention = 0
                        self.adopt_comm = 0
                        
# =============================================================================
            
            #no other bldgs can make community so go for individual            
            elif len(npvs_list) == 0:
                if Agents_Ind_NPVs.loc[step_ctr][self.unique_id] > reduction*Agents_Ind_Investments.loc[step_ctr][self.unique_id]: #0:
                    self.adopt_ind = 1
                    self.adopt_year = 2018 + step_ctr
                    agents_info.update(pd.Series([1], name  = 'Adopt_IND', index = [self.unique_id]))
                    agents_info.update(pd.Series([2018+step_ctr], name  = 'Year', index = [self.unique_id]))
                    agents_info.update(pd.Series([self.total], name  = 'intention', index = [self.unique_id]))
                    agents_info.update(pd.Series([ind_npv], name  = 'Ind_NPV', index = [self.unique_id]))
                    agents_info.update(pd.Series(["Positive_Ind & No_Comm_possible"], name  = 'Reason', index = [self.unique_id]))
                    self.intention = 0
                    self.adopt_comm = 0
                else:
                    self.adopt_ind = 0
                    agents_info.update(pd.Series([0], name  = 'Adopt_IND', index = [self.unique_id]))
                    agents_info.update(pd.Series([self.total], name  = 'intention', index = [self.unique_id]))
                    agents_info.update(pd.Series([ind_npv], name  = 'Ind_NPV', index = [self.unique_id]))
                    agents_info.update(pd.Series(["Negative_Ind & No_Comm_possible"], name  = 'Reason', index = [self.unique_id]))
                    self.intention = 0
                    self.adopt_comm = 0
                
            self.intention = 0 #if you adopt your intention has to be zero for the next years
        
        #if only individual adoption is possible
        elif self.part_comm == 0:# or self.part_comm == 1:#ZEV == 0: #self.part_comm == 0 and ZEV == 0:# or self.part_comm == 1: #self.part_comm === 1 is so that there are no community adoptions
            
            #adopt if NPV > 0
            if Agents_Ind_NPVs.loc[step_ctr][self.unique_id] > reduction*Agents_Ind_Investments.loc[step_ctr][self.unique_id]:
                self.adopt_ind = 1
                self.adopt_year = 2018 + step_ctr
                agents_info.update(pd.Series([1], name  = 'Adopt_IND', index = [self.unique_id]))
                agents_info.update(pd.Series([2018+step_ctr], name  = 'Year', index = [self.unique_id]))
                agents_info.update(pd.Series([self.total], name  = 'intention', index = [self.unique_id]))
                agents_info.update(pd.Series([ind_npv], name  = 'Ind_NPV', index = [self.unique_id]))
                agents_info.update(pd.Series(["Only_Ind"], name  = 'Reason', index = [self.unique_id]))
                self.intention = 0
                self.adopt_comm = 0
            else:
                self.adopt_ind = 0
                agents_info.update(pd.Series([0], name  = 'Adopt_IND', index = [self.unique_id]))
                agents_info.update(pd.Series([self.total], name  = 'intention', index = [self.unique_id]))
                agents_info.update(pd.Series([ind_npv], name  = 'Ind_NPV', index = [self.unique_id]))
                agents_info.update(pd.Series(["Negative_Only_Ind"], name  = 'Reason', index = [self.unique_id]))
                self.intention = 0
                self.adopt_comm = 0
                    
    elif self.intention != 1:
        print("*-*-*-*-*-*-*")
        self.adopt_ind = 0
        agents_info.update(pd.Series([0], name  = 'Adopt_IND', index = [self.unique_id]))
        agents_info.update(pd.Series([self.total], name  = 'intention', index = [self.unique_id]))
        #agents_info.update(pd.Series([ind_npv], name  = 'Ind_NPV', index = [self.unique_id]))
        agents_info.update(pd.Series(["Intention<0.5"], name  = 'Reason', index = [self.unique_id]))
        self.intention = 0
        self.adopt_comm = 0         
        
    #to ensure the buildings with already installed solar are always counted as having installed solar at the start of the code
    if scenario == "ZEV" or  scenario == "no_ZEV":
        if self.unique_id in list_installed_solar_bldgs_100MWh:
            print(self.unique_id)
            self.adopt_ind = 1
            self.adopt_year = 2018 + step_ctr 
            self.intention = 0
            agents_info.update(pd.Series([1], name  = 'Adopt_IND', index = [self.unique_id]))
            agents_info.update(pd.Series([2018+step_ctr], name  = 'Year', index = [self.unique_id]))
            agents_info.update(pd.Series([self.total], name  = 'intention', index = [self.unique_id]))
            agents_info.update(pd.Series([ind_npv], name  = 'Ind_NPV', index = [self.unique_id]))
            agents_info.update(pd.Series(["Existing"], name  = 'Reason', index = [self.unique_id]))
    elif scenario == "TOP4_no100MWh_retail" or scenario == "TOP4_no100MWh_wholesale":
        if self.unique_id in list_installed_solar_bldgs_ALL:
            print(self.unique_id)
            self.adopt_ind = 1
            self.adopt_year = 2018 + step_ctr
            self.intention = 0
            agents_info.update(pd.Series([1], name  = 'Adopt_IND', index = [self.unique_id]))
            agents_info.update(pd.Series([2018+step_ctr], name  = 'Year', index = [self.unique_id]))
            agents_info.update(pd.Series([self.total], name  = 'intention', index = [self.unique_id]))
            agents_info.update(pd.Series([ind_npv], name  = 'Ind_NPV', index = [self.unique_id]))
            agents_info.update(pd.Series(["Existing"], name  = 'Reason', index = [self.unique_id]))
        
        
        
        
        
              
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


