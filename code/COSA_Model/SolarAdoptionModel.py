# -*- coding: utf-8 -*-
"""
Current version: June, 2020
@authors: Prakhar Mehta, Alejandro Nuñez-Jimenez
"""
#%% IMPORT PACKAGES

# Import python packages
import sys
import pandas as pd
import numpy as np

# Import classes and functions from python packages
from mesa import Model
from mesa.datacollection import DataCollector

# Import classes and functions from own scripts
from COSA_Tools.scheduler import StagedActivation_random
from COSA_Tools import dc_functions
        
class SolarAdoptionModel(Model):
    '''
    Model of solar photovoltaic adoption for individual agents (i.e. building
    owners) who decide to invest in the technology or not, and whether to form
    solar communities or not.
    '''    
    
    def __init__(self, agent, inputs, ind_npv_outputs, 
        AgentsNetwork, agents_info, distances, solar, demand, seed):
        '''
        This method initializes the instantiation of the model class, and 
        creates the agents in it.

        Inputs:
            agent = Agent model taken as an input (class)
            inputs = all inputs for experiment simulation (dict)
            seed = random seed for the model (int)
            agents_info = data about buildings in the model (df).
            AgentsNetwork = peers of each agent (df)
            distances = distances between all buildings (df)
            solar = hourly generation potential for each building (df)
            demand = hourly electricity consumption for each building (df)

        Return:
            None (it executes when a new model class object is instantiated).
        '''
        ## INITIALIZE MAIN MODEL VARIABLES

        # Make inputs a variable of the model
        self.inputs = inputs

        # Define the small world network of the agents
        self.AgentsNetwork = AgentsNetwork

        ## INITIALIZE SIMULATION PARAMETERS
        
        # Define number of agents
        self.n_agents = min(inputs['simulation_parameters']["n_agents"], len(agents_info.keys()))

        # Define activator method used and steps in the model
        self.schedule = StagedActivation_random(self,
                                    stage_list = ['step_idea','step_decision'],
                                    shuffle = True, 
                                    shuffle_between_stages = True,
                                    seed = seed)
        
        # Define variable for simulation step and initialize to zero
        self.step_ctr = 0

        ## INITIALIZE CALIBRATION / SCENARIO PARAMETERS

        # Determine if communities are allowed
        self.com_allowed = inputs["simulation_parameters"]["ZEV"]

        # Define ideation step weighting parameters and threshold
        self.w_att = inputs["calibration_parameters"]["w_att"]
        self.w_econ = inputs["calibration_parameters"]["w_econ"]
        self.w_swn = inputs["calibration_parameters"]["w_swn"]
        self.w_subplot = inputs["calibration_parameters"]["w_subplot"]
        self.threshold = inputs["calibration_parameters"]["threshold"]

        # Determine threshold of losses agents accept
        self.reduction = inputs["calibration_parameters"]["reduction"]

        # Determine the minimum solar generation ratio to demand for communities
        self.min_ratio_sd = inputs["economic_parameters"]["min_ratio_sd"]
        
        # Determine the number of closest neighbors to consider for communities
        self.n_closest_neighbors = inputs["simulation_parameters"]["n_closest_neighbors"]

        ## DEFINE AGENTS VARIABLES

        # Create a dataframe to store communities formed
        self.combos_formed_info = pd.DataFrame(data=None)

        ## CREATE AGENTS
        for unique_id in list(agents_info.keys())[:self.n_agents]:
            
            # Determine the agent's environmental attitude
            ag_env_aw = self.determine_agent_env_aw(unique_id, agents_info, inputs)

            # Define agents perceived profitability over simulated years

            # Try to read the pp in the pre-calculated data
            try:
                pps_norm_years = ind_npv_outputs["Agents_PPs_Norm"][unique_id]

            # If the agent is not found, then pp is always zero (max_pp)
            except KeyError:
                years = (inputs["simulation_parameters"]["end_year"] - inputs["simulation_parameters"]["start_year"])+1
                pps_norm_years = [0] * years

            # Create instantiation of an agent and provide necessary inptus
            ag = agent(self,
                unique_id = unique_id,
                bldg_type = agents_info[unique_id]['bldg_type'],
                bldg_own = agents_info[unique_id]['bldg_owner'],
                bldg_zone = agents_info[unique_id]['zone_id'],
                bldg_plot = agents_info[unique_id]['plot_id'],
                attitude = ag_env_aw,
                pv_size = agents_info[unique_id]['pv_size_kw'],
                pv_possible = agents_info[unique_id]['can_install_pv'],
                peers= self.AgentsNetwork.loc[:,unique_id],
                n_sm = agents_info[unique_id]['num_smart_meters'],
                solar = np.array(solar[unique_id]) * inputs["economic_parameters"]["AC_conv_eff"],
                demand = np.array(demand[unique_id]),
                npv_ind_years = list(ind_npv_outputs["Agents_NPVs"][unique_id].values),
                inv_ind_years = list(ind_npv_outputs["Agents_Investment_Costs"][unique_id].values),
                pps_norm_years = pps_norm_years,
                ind_scr = ind_npv_outputs["Agents_SCRs"][unique_id].values,
                distances = distances[[unique_id, "dist_"+unique_id]]
                )

            # Add agent to model schedule
            self.schedule.add(ag)

        # Define data collection
        self.datacollector = DataCollector(
            model_reporters = {
                "sim_year":"step_ctr",
                "Ind_solar_number": dc_functions.functions.cumulate_solar_ind,
                "Ind_PV_Installed_CAP": dc_functions.functions.cumulate_solar_ind_sizes,
                "Comm_solar_number": dc_functions.functions.cumulate_solar_comm,
                "Num_of_Comms": dc_functions.functions.cumulate_solar_champions,
                "Comm_PV_Installed_CAP":dc_functions.functions.cumulate_solar_comm_sizes,
                "GYM_PV_CAP":dc_functions.functions.agent_type_gym_CAP,
                "HOSPITAL_PV_CAP":dc_functions.functions.agent_type_hospital_CAP,
                "HOTEL_PV_CAP":dc_functions.functions.agent_type_hotel_CAP,
                "INDUSTRIAL_PV_CAP" :dc_functions.functions.agent_type_industrial_CAP,
                "LIBRARY_PV_CAP" :dc_functions.functions.agent_type_library_CAP,
                "MULTI_RES_PV_CAP" :dc_functions.functions.agent_type_multi_res_CAP,
                "OFFICE_PV_CAP":dc_functions.functions.agent_type_office_CAP,
                "PARKING_PV_CAP":dc_functions.functions.agent_type_parking_CAP,
                "SCHOOL_PV_CAP":dc_functions.functions.agent_type_school_CAP,
                "SINGLE_RES_PV_CAP":dc_functions.functions.agent_type_single_res_CAP,
                "Num_GYM" :dc_functions.functions.agent_type_gym,
                "Num_HOSPITAL":dc_functions.functions.agent_type_hospital,
                "Num_HOTEL":dc_functions.functions.agent_type_hotel,
                "Num_INDUSTRIAL":dc_functions.functions.agent_type_industrial,
                "Num_LIBRARY" :dc_functions.functions.agent_type_library,
                "Num_MULTI_RES":dc_functions.functions.agent_type_multi_res,
                "Num_OFFICE":dc_functions.functions.agent_type_office,
                "Num_PARKING" :dc_functions.functions.agent_type_parking,
                "Num_SCHOOL":dc_functions.functions.agent_type_school,
                "Num_SINGLE_RES" :dc_functions.functions.agent_type_single_res
            },
            
            # Define agent reporters that are not inputs and change over time
            agent_reporters = {
                "building_id":"unique_id",
                "sim_year":"model.step_ctr",
                "intention":"intention",
                "attitude":"attitude",
                "pp":"pp",
                "peer_effect":"peer_effect",
                "ideation_total":"ideation_total",
                "neighbor_influence":"neighbor_influence",
                "adopt_ind":"adopt_ind",
                "adopt_comm":"adopt_comm",
                "adopt_year":"adopt_year",
                "en_champ":"en_champ",
                "reason_adoption":"reason_adoption"
            },
            tables = {
                "communities": ["year", "community_id", "solar", "demand",
                    "SC", "SCR", "pv_size", "pv_size_added", "n_sm",
                    "n_sm_added", "npv"],
            }
            )
                                    
        self.running = True
        
    
    def step(self):
        '''
        This method advances the model one step (i.e. one simulation year).
        '''
        
        # Loop through all agents using the scheduler
        self.schedule.step()
        
        # Collect data at the beginning of the step
        self.datacollector.collect(self)
        
        # Increase time counter by one
        self.step_ctr += 1

        return self.combos_formed_info
    
    def determine_agent_env_aw(self, unique_id, agents_info, inputs):
        """
        This method determines the agent's environemntal awarenes.

        Inputs
            unique_id = agent's name (str)
            agents_info = contains information about buildings (dict)
            inputs = simulation, calibration, and economic parameters (dict)
        Returns
            ag_env_aw = agent's environmental awareness (float)
        """
        # Read parameters from experiment inputs
        a_minergie = inputs["calibration_parameters"]["awareness_minergie"]
        a_mean = inputs["calibration_parameters"]["awareness_mean"]
        a_stdev = inputs["calibration_parameters"]["awareness_stdev"]

        # If the agent has a minergie label set the awareness to 0.95
        if agents_info[unique_id]['minergie'] == 1:
            agent_env_awareness = a_minergie

        # Otherwise, set the awareness following a truncated normal
        # distribution between 0 and 1
        else:
            agent_env_awareness = max(min(self.random.gauss(a_mean,a_stdev),1),0)
        
        return agent_env_awareness