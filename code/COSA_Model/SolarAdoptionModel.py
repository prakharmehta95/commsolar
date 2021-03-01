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
        
class SolarAdoptionModel(Model):
    '''
    Model of solar photovoltaic adoption for individual agents (i.e. building
    owners) who decide to invest in the technology or not, and whether to form
    solar communities or not.
    '''    
    
    def __init__(self, agent, inputs, agents_info, distances, solar, demand, seed):
        '''
        This method initializes the instantiation of the model class, and 
        creates the agents in it.

        Inputs:
            agent = Agent model taken as an input (class)
            inputs = all inputs for experiment simulation (dict)
            seed = random seed for the model (int)
            agents_info = data about buildings in the model (df)
            distances = distances between all buildings (df)
            solar = hourly generation potential for each building (df)
            demand = hourly electricity consumption for each building (df)

        Return:
            None (it executes when a new model class object is instantiated).
        '''
        ## INITIALIZE MAIN MODEL VARIABLES

        # Make inputs a variable of the model
        self.inputs = inputs

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

        # Define start and end years
        self.start_year = inputs["simulation_parameters"]["start_year"]
        self.end_year = inputs["simulation_parameters"]["end_year"]

        # Define initial year as first simulated year
        self.sim_year = self.start_year

        ## INITIALIZE CALIBRATION / SCENARIO PARAMETERS

        # Define ideation step weighting parameters and threshold
        self.w_att = inputs["calibration_parameters"]["w_att"]
        self.w_econ = inputs["calibration_parameters"]["w_econ"]
        self.w_swn = inputs["calibration_parameters"]["w_swn"]
        self.w_subplot = inputs["calibration_parameters"]["w_subplot"]
        self.threshold = inputs["calibration_parameters"]["threshold"]

        # Determine threshold of losses agents accept
        self.reduction = inputs["calibration_parameters"]["reduction"]

        # Determine minimum solar generation ratio to demand for communities
        self.min_ratio_sd = inputs["economic_parameters"]["min_ratio_sd"]
        
        # Determine the number of closest neighbors to consider for communities
        self.n_closest_neighbors = inputs["simulation_parameters"]["n_closest_neighbors"]

        # INITIALIZE POLICY PARAMETERS

        # Initialize model without allowing for solar communities
        self.com_allowed = False

        # Define year communities become allowed
        self.com_allowed_year = inputs["policy_parameters"]["com_year"]

        # Define if joining an existing community is allowed
        self.join_com = True

        # Define if FIT is available
        self.fit_on = True

        # Define the feed-in tariff for high and low electricity price hours
        self.fit_high = inputs["economic_parameters"]["fit_high"]
        self.fit_low = inputs["economic_parameters"]["fit_low"]

        # Define the historical FIT per system size since 2010
        self.hist_fit = inputs["economic_parameters"]["hist_fit"]

        # Define if investment subsidies are available
        self.sub_on = True

        # Define base investment subsidy
        self.base_d = inputs["economic_parameters"]["base_d"]

        # Define volumetric (potence) terms for first 30 kW, 100 kW, etc
        self.pot_30_d = inputs["economic_parameters"]["pot_30_d"]
        self.pot_100_d = inputs["economic_parameters"]["pot_100_d"]
        self.pot_100_plus_d = inputs["economic_parameters"]["pot_100_plus_d"]

        # Initialize policy cost of investment subsidies
        self.pol_cost_sub_ind = 0
        self.pol_cost_sub_com = 0

        # Allow or not direct marketing of self-produced electricity
        self.direct_market = inputs["policy_parameters"]["direct_market"]

        # Establish annual consumption to allow direct marketing
        self.direct_market_th = inputs["policy_parameters"]["direct_market_th"]

        ## INITIALIZE ECONOMIC PARAMETERS
        
        # Define PV lifetime (in years)
        self.pv_lifetime = inputs["economic_parameters"]["PV_lifetime"]

        # Define degradation rate of PV output
        self.deg_rate = inputs["economic_parameters"]["PV_degradation"]

        # Define discount rate
        self.disc_rate = inputs["economic_parameters"]["disc_rate"]

        # Defime maximum payback period
        self.max_pp = inputs["economic_parameters"]["max_payback_period"]

        # Initialize community pp variable
        self.av_pp_com = 0

        # Define scale effects parameters
        self.pv_scale_alpha = inputs["economic_parameters"]["pv_scale_alpha"]
        self.pv_scale_beta = inputs["economic_parameters"]["pv_scale_beta"]

        # Define dictionary of smart meter prices        
        # Key = limit of smart meters for price category (string) (e.g., "12")
        # Value = price per smart meter for than number of smart meters (int)
        self.smp_dict = inputs["economic_parameters"]["smart_meter_prices"]

        # Define TOU hours in a year
        # All hours of the year are "low" except from Mon-Sat from 6-21
        self.hour_price = inputs["economic_parameters"]["hour_price"]

        # Define relation of hourly price to annual average for spot market
        self.hour_to_average = np.array(inputs["economic_parameters"]["hour_to_average"])

        # Define if simple or discounted payback period
        self.discount_pp = inputs["economic_parameters"]["discount_pp"]

        # Define initial PV price
        self.pv_price = inputs["economic_parameters"]["hist_pv_prices"][str(self.start_year)]

        # Define rate of PV price reduction 
        self.pv_price_change = inputs["economic_parameters"]["pv_price_change"]

        # Define Operation & Maintenance costs
        self.om_cost = inputs["economic_parameters"]["OM_Cost_rate"]

        # Define electricity tariff names
        self.el_tariff_names = ["COSA_R1","COSA_R2","COSA_R3","COSA_R4","COSA_R5","COSA_R6","COSA_C1","COSA_C2","COSA_C3","COSA_C4","COSA_C5"]

        # Define initial electricity prices
        # Index is (sim_year - 2010) because historical data starts in 2010
        self.el_price = {tariff:inputs["economic_parameters"]["hist_el_prices"][tariff][self.sim_year-2010] for tariff in self.el_tariff_names}

        self.wholesale_el_price = inputs["economic_parameters"]["hist_wholesale_el_prices"][self.sim_year-2010]

        # Define rate of electricity prices change for tariffs and wholesale
        self.el_price_change = inputs["economic_parameters"]["el_price_change"]

        self.wholesale_el_price_change = inputs["economic_parameters"]["wholesale_el_price_change"]

        # Define demand limits for electricity tariff type
        self.el_tariff_demands = inputs["economic_parameters"]["el_tariff_demand"]

        # Define ratio between high and low TOU electricity price
        self.ratio_high_low = inputs["economic_parameters"]["ratio_high_low"]

        # Define surcharge for smart metering within communities
        self.solar_split_fee = inputs["economic_parameters"]["ewz_solarsplit_fee"]

        ## DEFINE AGENTS VARIABLES

        # Create a dataframe to store communities formed
        self.combos_formed_info = pd.DataFrame(data=None)

        ## CREATE AGENTS
        for unique_id in list(agents_info.keys())[:self.n_agents]:
            
            # Determine the agent's environmental attitude
            ag_env_aw = self.determine_agent_env_aw(unique_id, agents_info, inputs)

            # Assign an electricity tariff to the agent
            ag_tariff = self.assign_electricity_tariff(unique_id, demand,agents_info, self.el_tariff_demands, inputs["economic_parameters"]["av_hh_size"])

            # Create agent's social network **NOT USED FOR THESIS SIMULATIONS**
            ag_peers = self.assign_social_network(unique_id, distances[[unique_id, "dist_"+unique_id]], inputs["simulation_parameters"]["n_peers"], inputs["simulation_parameters"]["p_rewire"])

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
                peers = ag_peers,
                n_sm = agents_info[unique_id]['num_smart_meters'],
                solar = np.array(solar[unique_id]) * inputs["economic_parameters"]["AC_conv_eff"],
                demand = np.array(demand[unique_id]),
                distances = distances[[unique_id, "dist_"+unique_id]],
                tariff = ag_tariff
                )

            # Add agent to model schedule
            self.schedule.add(ag)

        # Define data collection
        self.datacollector = DataCollector(
            model_reporters = {
                "sim_step":"step_ctr",
                "sim_year":"sim_year",
                "pv_price":"pv_price",
                "n_ind": lambda m: np.sum([ag.adopt_ind for ag in m.schedule.agents]),
                "inst_cum_ind": lambda m: np.sum([ag.pv_size for ag in m.schedule.agents if ag.adopt_ind == 1]),
                "n_com": lambda m: np.sum([ag.adopt_com for ag in m.schedule.agents]),
                "inst_cum_com":lambda m: np.sum([ag.pv_size for ag in m.schedule.agents if ag.adopt_com == 1]),
                "pol_cost_sub_ind":"pol_cost_sub_ind",
                "pol_cost_sub_com":"pol_cost_sub_com",
                "direct_market_th":"direct_market_th",
                "direct_market":"direct_market",
                "com_year":"com_allowed_year"
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
                "adopt_com":"adopt_com",
                "adopt_year":"adopt_year",
                "en_champ":"en_champ",
                "reason_adoption":"reason_adoption",
                "ind_inv":"ind_inv",
                "ind_scr":"ind_scr",
                "ind_npv":"ind_npv",
            },
            tables = {
                "communities": ["year", "community_id", "solar", "demand",
                    "SC", "SCR", "pv_size", "pv_size_added", "n_sm",
                    "n_sm_added", "npv", "tariff", "pv_sub", "inv_new", "inv_old", "pp_com"],
            }
            )
                                    
        self.running = True
    
    def step(self):
        '''
        This method advances the model one step (i.e. one simulation year).
        '''
        # Allow communities or not depending on year
        if self.sim_year >= self.com_allowed_year:
            self.com_allowed = True

        # Update PV price
        if self.sim_year < 2020 :
            self.pv_price = self.inputs["economic_parameters"]["hist_pv_prices"][str(self.sim_year)]
        else:
            self.pv_price *= (1 + self.pv_price_change)

        # Update electricity price
        if self.sim_year < 2020:
            # Historical data always starts in 2010
            self.el_price = {tariff:self.inputs["economic_parameters"]["hist_el_prices"][tariff][self.sim_year-2010] for tariff in self.el_tariff_names}

            self.wholesale_el_price = self.inputs["economic_parameters"]["hist_wholesale_el_prices"][self.sim_year-2010]
        else:
            for tariff in self.el_tariff_names:
                self.el_price[tariff] *= (1 + self.el_price_change)
            
            self.wholesale_el_price *= (1 + self.wholesale_el_price_change)

        # Update base investment subsidy
        if self.sim_year > 2020:
            for var in [self.base_d, self.pot_30_d, self.pot_100_d, self.   pot_100_plus_d]:
                var[str(self.sim_year)] = var[str(self.sim_year - 1)] * (1 + self.pv_price_change)
        
        # Loop through all agents using the scheduler
        self.schedule.step()
        
        # Collect data at the beginning of the step
        self.datacollector.collect(self)
        
        # Increase time counter by one
        self.step_ctr += 1

        # Update simulation year
        self.sim_year += 1

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
    
    def assign_electricity_tariff(self, unique_id, demand, agents_info, el_tariff_demands, av_hh_size):
        """
        This method assigns an electricity tariff to each agent depending on (1) the type of use of the building (residential, commercial), and the
        (2) annual electricity consunption.

        Inputs:
            unique_id = name of the building (str)
            demand = list of hourly demand for each building over a year (d)
            agents_info = dictionary with info on buildings (d)
            el_tariff_demands = list of max annual demand for each tariff (d)
            av_hh_size = average household size in Switzerland (float)
        Returns:
            el_tariff = name of electricity tariff (str)
        """

        # Read building type
        building_type = agents_info[unique_id]["bldg_type"]
        

        # Classify building into commercial or residential eletricity tariffs
        if (building_type != "MULTI_RES") or (building_type != "SINGLE_RES"):
            
            # Define type of tariff
            t_type = "commercial"

            # Read demand for building
            demand_yr = np.sum(demand[unique_id])


        else:
            
            # Define type of tariff
            t_type = "residential"

            # Average household 2.2 (2018)
            # https://www.bfs.admin.ch/bfs/en/home/statistics/regional-statistics/regional-portraits-key-figures/cantons/zurich.html

            # Compute the number of households in the building
            n_households = agents_info[unique_id]["total_persons"] / av_hh_size

            # Define average consumption of households in building
            if n_households > 1:
                demand_yr = np.sum(demand[unique_id]) / n_households
            else:
                demand_yr = np.sum(demand[unique_id])
        
        # List max demands for each tariff category
        t_ds = sorted(list(el_tariff_demands[t_type].values()))

        # Find the index of the category whose demnad limit is higher than the annual demand of the building
        try:
            t_ix = next(ix for ix,v in enumerate(t_ds) if v > demand_yr)
        except:
            t_ix = len(t_ds)

        # Read the label of the tariff and return it
        # Note: we can only do this because demand limits and tariff names can be sorted alphabetically and by value, otherwise this is wrong!
        ag_tariff = sorted(list(el_tariff_demands[t_type].keys()))[t_ix]

        return ag_tariff

    def assign_social_network(self, unique_id, distances, n_peers, p_rewire):
        """
        This method creates a list of peers with whom the agent is connected.

        Inputs:
            unique_id = agent's identifier (str)
            distances = distance to other agents (df) (two columns: "unique_id" contains agents ids, "dist_"+unique_id contains distance to agent)
            n_peers = number of connections per agent (int)
            p_rewire = probability connection rewiring to distant agent (float)
        
        Returns:
            peers_list = list of agents ids connected to agent (list)
        """

        # Order dataframe with distances and pick n_peers first
        all_near_peers = list(distances.sort_values("dist_"+unique_id)[unique_id])[:n_peers]

        # Loop through all connections, remove at random, and rewire later
        near_peers = [all_near_peers[i] for i in range(n_peers) if self.random.uniform(0,1) > p_rewire]

        # If any of the connections have been removed, connect to other agent
        if len(near_peers) < n_peers:
            far_ags = list(distances.sort_values("dist_"+unique_id)[n_peers:][unique_id])
            far_peers = self.random.sample(far_ags, n_peers - len(near_peers))
        else:
            far_peers = []

        # Make list of connections
        peers_list = near_peers + far_peers

        # ALTERNATIVE - PURE RANDOM CHOICE AMONG ALL AGENTS
        #peers_list = self.random.sample(list(distances[unique_id]), k = n_peers)
        
        return peers_list