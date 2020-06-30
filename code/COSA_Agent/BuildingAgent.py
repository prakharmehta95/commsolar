#%% AGENT code
import pandas as pd
import numpy as np
import itertools

from mesa import Agent

from COSA_Tools.npv_ind import calculate_com_npv

class BuildingAgent(Agent):
    """
    Creates a building owner agent.
    """

    def __init__(self, model, unique_id, bldg_type, bldg_own, bldg_zone, 
        bldg_plot, attitude, pv_size, pv_possible, peers, n_sm,
        solar, demand, npv_ind_years, inv_ind_years, pps_norm_years, ind_scr,
        distances):
        '''
        Initializes the attributes of the agent (i.e. building owner).

        Inputs:
            model = model as input
            unique_id   = Agent unique identification (B140907,...)
            bldg_type   = Type of the building 
            attitude    = Environmental attitude [0,1]
            pv_size     = size of the pv system
            peers = list of unique_ids of agents connected to this agent
        
        '''
        super().__init__(unique_id, model) 
        
        # INITIALIZE AGENT VARIABLES FROM MODEL INITIALIZATION

        # Assign a unique_id to this agent from the model object
        self.unique_id = unique_id 

        # Assign a building type to this agent
        self.bldg_type = bldg_type

        # Determine who is the building owner
        self.bldg_own = bldg_own

        # Assign a zone to this agent
        self.bldg_zone = bldg_zone

        # Assign a plot to this agent
        self.bldg_plot = bldg_plot

        # Determine the attitude (environmental awareness) of the agent
        self.attitude = attitude

        # Assign a PV size to the agent
        self.pv_size = pv_size

        # Assign a social network to the agent
        self.peers = list(peers)

        # Define number of smart meters in the building
        self.n_sm = n_sm
        
        # Define the agent's solar generation potential (in AC)
        self.solar = solar

        # Define the agent's hourly electricity demand
        self.demand = demand

        # Define if the agent has the possibility to install solar PV
        if pv_possible == 1:
            self.pv_possible = True
        else:
            self.pv_possible = False

        # Define the individual NPV per year
        self.npv_ind_years = npv_ind_years

        # Define the individual investment costs per years
        self.inv_ind_years = inv_ind_years

        # Define the agent's perceived profitability normalizes per years
        self.pps_norm_years = pps_norm_years

        # Define self-consumption rate for individual adoption
        self.ind_scr = ind_scr

        # Define dataframe with two columns for distance to each building
        self.distances = distances

        # INITIALIZE AGENT VARIABLES *NOT* FROM MODEL INITIALIZATION

        # Initialize the intention to adopt to zero
        self.intention = 0

        # Initialize the influence of peer-effects to zero
        self.peer_effect  = 0

        # Initialize the influence of neighbors to zero
        self.neighbor_influence = 0

        # Initialize the ideation total to zero (weighted sum of ideation variable)
        self.ideation_total = 0

        # Initialize the counter of how many times agent exceeds idea threshold
        self.intention_counter = 0

        # Initialize boolean attribute for individual adoption
        self.adopt_ind = 0

        # initialize boolean attribute for community adoption
        self.adopt_comm = 0

        # initialize adoption year to zero
        self.adopt_year = 0

        # initialize attribute as energy chamption to zero
        self.en_champ = 0

        # Define the name (str) of the community the agent is part of
        self.com_name = None

        # Define the agent's perceived profitability
        self.pp = self.pps_norm_years[self.model.step_ctr]

        # Establish reason for adoption
        self.reason_adoption = None
        
    def step_idea(self):
        '''
        Determines if the agent develops the idea of adopting solar PV or not.
        Input:
            None (based on values of agent attributes)
        Output:
            None (it modifies agent attribute: self.intention)
        '''
        
        # If the agent has already adopted solar PV, whether individually or
        # as part of a community, set the intention to 0
        if self.adopt_comm == 1 or self.adopt_ind == 1:
            self.intention = 0

        # If the agent has not adopted solar PV, update its attributes and 
        # evaluate whether it develops the intention in this step                        
        else:

            # Evaluate the influence of peers
            self.peer_effect = self.check_peers()

            # Evaluate the persuasion effect from neighbors
            self.neighbor_influence = self.check_neighbors_influence()

            # Update the perceived profitability of solar PV
            self.pp = self.pps_norm_years[self.model.step_ctr]
            
            # Update ideation_total as weighted sum of ideation variables
            self.ideation_total = (self.model.w_att * self.attitude 
                            + self.model.w_econ * self.pp
                            + self.model.w_swn * self.peer_effect
                            + self.model.w_subplot * self.neighbor_influence)
            
            # If self.ideation_total is greater than the ideation threshold
            if self.ideation_total > self.model.threshold:

                # Then the agent develops the intention to adopt solar
                self.intention = 1

                # Count how many years the agent develops the intention
                self.intention_counter += 1
            
            # Else, remain without the idea to adopt solar
            else:
                self.intention = 0   
    
    def check_peers(self):
        """
        Determines the impact of peer effects in the agent.

        Checks the number of peers (agents in the network of current agent)
        that have installed solar and updates agent attribute peer_effect
        (self.peer_effect) accordingly
        
        Inputs:
            None
        Returns:
            peer_effect = fraction of contacts with solar installed (float)
        """
        
        # Initialize output variable for peer-effects
        peer_effect = 0

        # Determine the number of contacts in the network of current agent
        n_peers = len(self.peers)

        if n_peers == 0:
            print(self.unique_id + " zero peers")

        # List all agents in the model
        all_agents = self.model.schedule.agents

        # If the agent has any peers, then count how many have solar
        if n_peers != 0:

            # Create a list of agents that are peers of this one
            peers_list = [ag for ag in all_agents if ag.unique_id in self.peers]

            # Sum up the number of peers with solar
            n_peers_solar = sum([1 if (
                (ag.adopt_ind == 1) or (ag.adopt_comm == 1)) else 0 for ag in peers_list])
            
            # Determine peer effects
            peer_effect = (n_peers_solar/n_peers)

        return peer_effect

    def check_neighbors_influence(self):
        """
        Determines the impact of neighbors wanting to adopt on the agent.

        Checks how many other agents in the building block (plot) have the
        idea of installing solar.

        Inputs:
            None
        Returns:
            neighbor_influence = impact of neighbors (float)
        """

        # Initialize local variable
        neighbor_influence = 0

        # List all agents in the model
        all_agents = self.model.schedule.agents

        # Collect list of neighbors
        neighbors_list = [ag for ag in all_agents if ag.bldg_plot == self.bldg_plot]

        # Remove current agent from neighbors list
        neighbors_list = [ag for ag in neighbors_list if ag.unique_id != self.unique_id]

        # If there are any neighbors (own agent counts as 1)
        if len(neighbors_list) > 0:
            
            # Compute the number of neighbors with intention to adopt
            neighbors_idea = sum([1 if ag.intention == 1 else 0 for ag in neighbors_list])

            # Compute the neighbors persuation influence
            neighbor_influence = neighbors_idea / len(neighbors_list)

        return neighbor_influence

    def step_decision(self):
        """
        This method determines if the agent adopts solar or not, whether
        individually or as part of a community.

        The decision is based on the economic evaluation through NPV of all 
        the alternatives available to the agent (e.g., individual adoption, 
        and possible solar communities).

        The agent takes the option with the best NPV, and adopts only if it
        is positive or if the economic lossses are acceptable.
        """
           
        # Only agents with the intention and possibility to adopt enter this
        # step, who are not year part of a solar community
        if self.intention == 1 and self.pv_possible and self.adopt_comm == 0:
            
            # Create a list containing all the agents in the plot with the
            # idea to adopt PV or that already have PV, who are not in community
            potential_partners = [ag for ag in self.model.schedule.agents if (
                                    (ag.bldg_plot == self.bldg_plot) and (
                                    (ag.intention == 1) or (ag.adopt_ind == 1))
                                    and (ag.adopt_comm == 0))]
                
            # If agents in plot with idea to install and communities allowed
            if len(potential_partners) > 1 and self.model.com_allowed:

                # Define what available communities the agent considers
                # Important: neighbors in communities are discarded as potential
                # partners, which means an agent cannot join existing community
                partners_to_consider = self.define_partners_to_consider(
                            potential_partners, self.model.n_closest_neighbors)

                # Define what possible communities could be formed
                combinations_dict = self.define_possible_communities(
                                                        partners_to_consider)

                # Evaluate the characteristics of each possible community
                self.update_combinations_available(self.model, combinations_dict)

                # Remove combinations that do not meet the min criteria of
                # solar generation potential to electricity demand
                self.remove_communities_below_min_ratio_sd(
                                    self.model.min_ratio_sd, combinations_dict)
                
                # Compare the possibility to join a community with individual
                # adoption of solar (if there is any combination to consider)                                                                        
                if len(combinations_dict) > 0:

                    # Pick the community with highest NPV among all possible
                    c_max_npv = self.pick_community_highest_npv(combinations_dict)

                    # Compute the  shares of NPV of each member
                    npv_sh_d =  self.compute_community_npv_shares(
                                                combinations_dict[c_max_npv])
                    
                    # Store the shares in the dictionary of best community
                    combinations_dict[c_max_npv]["npv_shares"] = npv_sh_d

                    # Determine the NPV for the active agent of best option
                    npv_com = combinations_dict[c_max_npv]["npv_shares"][self.unique_id]

                    # Read NPVs for individual adoption
                    npv_ind = self.npv_ind_years[self.model.step_ctr]

                    # If NPV community is larger than NPV individual
                    if npv_ind < npv_com:
                        
                        # Go for a solar community
                        self.consider_community_adoption(c_max_npv,
                            combinations_dict[c_max_npv], self.model)                        
                    
                    # If NPV of individual adoption is greater than the
                    # NPV of the solar community alternative
                    elif npv_ind > npv_com:
                    
                        # Go for individual adoption
                        self.consider_individual_adoption(self.model)
                
                # If there are no possible solar communities
                else:

                    # Go for individual adoption
                    self.consider_individual_adoption(self.model)
                        
            
            # If no agents in plot with intention and no individual solar yet
            elif len(potential_partners) == 0 and self.adopt_ind != 1:

                # Go for individual adoption
                self.consider_individual_adoption(self.model)

    
    def consider_individual_adoption(self, model):
        """
        This method determines if the agent adopts solar PV as an individual.

        Inputs
            self = agent (obj)
            model = model (obj)
                Note: I pass the model as input to have access to all its
                variables, and call it model to avoid confusion and shorten
                the code.
        Returns
            None (agent attributes are updated directly)
        """

        # Read the individual adoption NPV for this year to this agent
        ind_npv_yr_ag = self.npv_ind_years[model.step_ctr]

        # Read the investment cost for individual adoption this year this agent
        ind_inv_yr_ag = self.inv_ind_years[model.step_ctr]

        # Read the maximum acceptance for negative NPV values
        neg_npv_fraction = model.inputs["calibration_parameters"]["reduction"]

        # If the NPV of individual adoption is greater or equal to the maximum
        # losses that the agent is willing to tolerate (expressed as a fraction
        # of the investment cost), then adopt
        if ind_npv_yr_ag >= neg_npv_fraction * ind_inv_yr_ag:

            # Then, adopt individually and update the agent's attributes

            # Define agent as individual adopter of solar PV
            self.adopt_ind  = 1

            # Record the year of installation
            self.adopt_year = 2018 + model.step_ctr

            # Record agent's reason for adoption
            self.reason_adoption = "only_ind"

    def consider_community_adoption(self, c_max_npv, c_max_npv_dict, model):
        """
        This method determines if the agent joins a solar community, by first
        checking if all members of the community agree.

        Inputs:
            c_max_npv = name of community with largest NPV (str)
            c_max_npv_dict = dict of community variables for community
                with highest NPV among all possible (dict)
            model = model as an input (obj)

        Returns:
            None
        """

        # Store decision of each agent in list
        members_decision = []

        # Compute the threshold of each agent in the community
        for ag in c_max_npv_dict["members"]:

            # Read individual investment cost for this year for the agent
            inv_ag = ag.inv_ind_years[model.step_ctr]
            
            # Compute threshold of losses tolerated by this agnet this year
            th_ag = (-1) * model.reduction * inv_ag

            # Determine agent decision
            if c_max_npv_dict["npv_shares"][ag.unique_id] > th_ag:
                members_decision.append(True)
            else:
                members_decision.append(False)

        # Adopt if all npv shares are acceptable for all agents
        if all(members_decision):

            # Form a community and update the attributes of all agents in it

            # Update this agent's attributes as the energy chamption
            self.en_champ = 1
            self.adopt_comm = 1
            self.adopt_ind = 0

            # Loop through all the agents in the community
            for com_ag in c_max_npv_dict["members"]:
                
                #manually setting adopt_ind to zero to avoid double counting
                com_ag.adopt_ind = 0
                
                #setting community adoption as 1 for all agents involved
                com_ag.adopt_comm = 1
                        
                # Record the year of installation
                com_ag.adopt_year = 2018 + self.model.step_ctr
                
                # Record agent's variables in info dataframe
                self.reason_adoption = "Comm>Ind"

            # Save data about formed community
            self.save_formed_community(c_max_npv, c_max_npv_dict)              
        
    def define_partners_to_consider(self, potential_partners, n_closest_neighbors):
        """
        Create a list of agents with whom a community could be formed taking the
        n_closest_neighbors potential partners based on distance to agent.

        Inputs:
            self = active agent (obj)
            potential_partners = agents in same plot and not in a community, with
                the idea to adopt solar OR individual solar (list of objects)
            n_closest_neighbors = max number of potential partners to consider (int)
        
        Returns:
            partners_to_consider = agents to consider forming a community (list objs)
        """

        # Read the active agent unique identification
        uid = self.unique_id

        # Check how many agents in the building's plot have intention to adopt
        if len(potential_partners) > n_closest_neighbors:

            # If there are more than n_closests_neighbors, then choose those
            # closest to the agent up to n_closest_neighbors

            # Reduce list of distances to only buildings in plot with ideas
            d_df = self.distances[self.distances[uid].isin(potential_partners)]
            # Note - column name for list of buildings in distance dataframe
            # is the unique_id of the active agent

            # Short buildings by distance to active agent
            d_df.sort_values(by = ['dist_' + uid])

            # List the n_closest_neighbors as the closest potential partners
            closest_pps = list(d_df[uid].iloc[:(n_closest_neighbors + 1)].values)

            # Create a list of agents (objs) containing the closest_pps
            partners_to_consider = [ag for ag in potential_partners 
                                                if ag.unique_id in closest_pps]

        # If there are fewer than n_closest_neighbors, then take all agents
        else:
            partners_to_consider = potential_partners

        return partners_to_consider

    def define_possible_communities(self, partners_to_consider):
        """
        Creates a dictionary of all possible communities of all possible sizes, 
        where the key is the name of the community and the value is a dictionary
        of the community characteristics.

        Inputs:
            self = active agent (obj)
            partners_to_consider = available agents to form community (list objs)

        Returns:
            combinations_dict = dictionary of all possible communities with
                key = community name (all unique_id of agents joined with "_") and
                value = dictionary of community properties (the only item is:
                key = "members", value = list of agents in community (list objs))
        
        Note: in future versions, this will be easier with a new object class.
        """

        # Create empty list to store all possible communities
        combinations_list = []

        # Create empty dictionary to store the possible communities by name
        combinations_dict = {}

        # Loop through all possible community sizes (i.e. number of members)
        for com_size in np.arange(1,len(partners_to_consider)+1):

            # Create a list of tuples, each one a combination possible
            combinations_list.extend(list(itertools.combinations(
                                            partners_to_consider, com_size)))
        
        # Transform the tuples into lists (to later add active agent)
        combinations_list = [list(c) for c in combinations_list]

        # Add active agent to each potential community in the list
        for item in combinations_list:
            item.append(self)

        # Create dictionary of combinations
        for community_members in combinations_list:

            # Create the community name by joining with "_" in between the list of
            # unique identifications of each agent in the community
            community_name = '_'.join([ag.unique_id for ag in community_members])

            # Create a dictionary for storing this community parameters
            combinations_dict[community_name] = {}

            # Store community in dictionary
            combinations_dict[community_name]["members"] = community_members
            
        return combinations_dict

    def update_combinations_available(self, model, combinations_dict):
        """
        This method completes the characteristics of all possible communities
        by directly updating the combinations_dict.
        """

        # Loop through combinations_dict to fill-in community attributes
        for c_name, c_dict in combinations_dict.items():

            # Create list of agents members of the community
            members = c_dict["members"]

            # Compute solar generation potential of community by summing along
            # the rows the columns in solar of the agents in community
            c_dict["solar"] = np.nansum([ag.solar for ag in members], axis=0)

            # Compute demand of community
            c_dict["demand"] = np.nansum([ag.demand for ag in members], axis=0)

            # IMPORTANT: if there is any agent with PV already:
            # (1) PV investment is the sum of agents with no prior PV
            # (2) Smart meter investmetns is sum of agents with no prior PV
            # IMPORTANT 2: PV and smart meter investment needs to be 
            # computed for the whole community to realize scale effects on prices

            # Compute total community size PV
            c_dict["pv_size"] = sum([ag.pv_size for ag in members])

            # Compute added PV (in case any agent has already PV)
            c_dict["pv_size_added"] = sum(
                [ag.pv_size for ag in members if ag.adopt_ind == 0])
            
            # Compute total number of smart meters in the community
            c_dict["n_sm"] = sum([ag.n_sm for ag in members])

            # Compute added smart meters (in case any agent has already PV)
            c_dict["n_sm_added"] = sum(
                [ag.n_sm for ag in members if ag.adopt_ind == 0])

            # Compute NPV of the community
            c_dict["npv"] = calculate_com_npv(model.inputs, c_dict, 
                                                model.step_ctr)
            
            """
            TO BE SOLVED: 
            (1) What happens with installed PV capacity?
            ---> At the moment, no PV inv and no SM costs of already installed
            """
        
    def pick_community_highest_npv(self, combinations_dict):
        """
        Finds the community with highest NPV among all possible.

        Inputs
            combinations_dict = all possible communities (dict)
        
        Returns
            c_max_npv = name of community with highest NPV (str)
                (it is a string with buildings id of all members joined
                with "_")
        """
        
        # Pick the community with the best NPV:
        max_c_npv = max([c["npv"] for c in combinations_dict.values()])

        # Read the name of the community with max_com_npv
        for c, c_d in combinations_dict.items():
            if c_d["npv"] == max_c_npv:
                c_max_npv = c
        
        return c_max_npv


    def compute_community_npv_shares(self, c_max_npv_dict):
        """
        This method computes the NPV share for each agent in the community
        with the best NPV of all the community available and stores it
        by updating combinations_dict.

        Inputs
            c_max_npv_dict = dict of community variables for community
                with highest NPV among all possible (dict)

        Returns
            npv_sh_d = a dictionary with key = building id and value =    
                NPV share for community with best NPV of all possible
        
        PROBLEM: 
            There are 230 buildings (of 4919) with zero yearly demand, and
            12 buildings (of 4919) with zero solar potential.

            There are several buildings that cannot install PV 
            (ag.pv_possible == False).

            This is not solved now. For example, a building with no demand
            would see an ag_d_share = 0. This results in an npv_sh_d = 0, 
            which means the agent will always say yes to form a community.
            This behavior is preserved because the agent would only export and
            not self-consume, we assume it goes with the community if others
            accept. HOWEVER, the cost of PV may exceed the revenues from 
            exporting above the acceptable losses.
        """
        
        # Create a dictionary of NPV share per agent for the c_max_npv
        npv_sh_d = {}

        # Loop through the community members
        for ag in c_max_npv_dict["members"]:

            # Compute the fraction of the community demand by each agent
            ag_d_share = sum(ag.demand) / sum(c_max_npv_dict["demand"])

            # Compute and store the fraction of NPV in temporary dict
            npv_sh_d[ag.unique_id] = ag_d_share * c_max_npv_dict["npv"]            

        return npv_sh_d
    
    def remove_communities_below_min_ratio_sd(self, min_ratio_sd, 
                                                            combinations_dict):
        """
        This method removes communities that do not meet the minium solar
        generation potential requirement from consideration.
        
        Inputs
            min_ratio_sd = minimum ratio of annual solar generation and 
                annual electricity demand (float)
            combinations_dict = all possible communities (dict)

        Returns
            None = it updates directly the combinations_dict 

        Note: this code also prevents the formation of communities with no
        solar generation or no demand.           
        """
        # List communities to delete
        coms_below_ratio = []

        # Remove communities that do not meet minimum criteria
        # of solar production to demand ratio
        for c, c_d in combinations_dict.items():

            if np.nansum(c_d["demand"]) != 0:
                # Divide annual solar generation over annual demand
                c_ratio_sd = np.nansum(c_d["solar"]) / np.nansum(c_d["demand"])
            
            # If the community has zero demand, it cannot be formed
            else:
                c_ratio_sd = 0

            # Compare the community's ratio
            if ((c_ratio_sd < min_ratio_sd) or np.isnan(c_ratio_sd) 
                                                    or np.isinf(c_ratio_sd)):

                # If ratio too small, put in list to remove
                coms_below_ratio.append(c)
        
        # Loop through the communities below ratio and delete them from dict
        for c in coms_below_ratio:
            del combinations_dict[c]
    
    def save_formed_community(self, c_name, c_dict):
        """
        This method exports the data about the formed community to the model
        """

        print("Community formed!")

        # Create an empty dictionary containing the community details
        c_export_dict = {}

        # Define year of community formation
        c_export_dict["year"] = self.model.step_ctr

        # Define community name
        c_export_dict["community_id"] = c_name

        # Define community solar generation potential
        c_export_dict["solar"] = np.nansum(c_dict["solar"])

        # Define community electricity demand
        c_export_dict["demand"] = np.nansum(c_dict["demand"])

        # Define community self-consumed electricity
        c_s = np.array(c_dict["solar"])
        c_d = np.array(c_dict["demand"])
        c_export_dict["SC"] = np.nansum([min(c_s[i], c_d[i])
                                     if c_s[i] > 0 else 0 for i in range(8760)])

        # Define community self-consumption ratio
        c_export_dict["SCR"] = c_export_dict["SC"] / c_export_dict["solar"]

        # Define PV size and profitability values
        for v in ["pv_size", "pv_size_added", "n_sm", "n_sm_added", "npv"]:
            c_export_dict[v] = c_dict[v]
        
        self.model.datacollector.add_table_row("communities", c_export_dict)