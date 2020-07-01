#%% AGENT code
import pandas as pd
import numpy as np
import itertools

from mesa import Agent

class BuildingAgent(Agent):
    """
    Creates a building owner agent.
    """

    def __init__(self, model, unique_id, bldg_type, bldg_own, bldg_zone, 
        bldg_plot, attitude, pv_size, pv_possible, peers, n_sm,
        solar, demand, distances, tariff):
        '''
        Initializes the attributes of the agent (i.e. building owner).

        Inputs:
            model = model as input
            unique_id   = Agent unique identification (B140907,...)
            bldg_type   = Type of the building 
            attitude    = Environmental attitude [0,1]
            pv_size     = size of the pv system
            peers = list of unique_ids of agents connected to this agent
            tariff = name of electricity tariff assigned to agent
        
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

        # Define dataframe with two columns for distance to each building
        self.distances = distances

        # INITIALIZE AGENT VARIABLES *NOT* FROM MODEL INITIALIZATION

        # Initialize the intention to adopt to zero
        self.intention = 0

        # Initialize the influence of peer-effects to zero
        self.peer_effect  = 0

        # Initialize the influence of neighbors to zero
        self.neighbor_influence = 0

        # Initialize ideation total to zero (weighted sum of ideation variable)
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
        # Initialize to maximum payback period
        self.pp = 1

        # Establish reason for adoption
        self.reason_adoption = None

        # Compute the agent's lifetime load profile
        self.lifetime_load_profile = self.compute_lifetime_load_profile(self.solar, self.demand, self.model.pv_lifetime,self.model.deg_rate, self.model.hour_price)

        # Compute the agent's self-consumption ratio for ind adoption
        self.ind_scr = self.lifetime_load_profile["SCR"][0]

        # Compute the agent's smart meters investment
        self.sm_inv = self.compute_smart_meters_inv(self.n_sm, self.model.smp_dict)

        # Define the agent's electricity tariff
        self.el_tariff = tariff

        # Initialize individual investment to zero
        self.ind_inv = 0

        # Initialize individual npv
        self.ind_npv = 0
        
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

        # If the agent has not adopted solar PV and not developed the intention before, then, update its attributes and evaluate whether it develops the intention in this step                        
        elif (self.adopt_comm == 0) and (self.adopt_ind == 0) and (self.intention != 1):

            # Evaluate the influence of peers
            self.peer_effect = self.check_peers()

            # Evaluate the persuasion effect from neighbors
            self.neighbor_influence = self.check_neighbors_influence()

            # Update investment cost for this year
            self.ind_inv = self.sm_inv + self.compute_pv_inv(self.pv_size, self.model.pv_price, self.model.pv_scale_alpha, self.model.pv_scale_beta, self.model.sim_year, self.model.base_d, self.model.pot_30_d, self.model.pot_100_d, self.model.pot_100_plus_d)

            # Update lifetime cashflows for this year
            self.lifetime_cashflows = self.compute_lifetime_cashflows(self.el_tariff, self.pv_size, self.lifetime_load_profile, self.model.deg_rate, self.model.pv_lifetime, self.model.sim_year, self.model.el_price, self.model.ratio_high_low, self.model.fit_high, self.model.fit_low, self.model.hist_fit, self.model.solar_split_fee,self.model.om_cost, sys="ind")

            # Update the perceived profitability of solar PV
            self.pp = self.check_econ_attractiveness(self.ind_inv, self.lifetime_cashflows, self.model.discount_pp, self.model.max_pp, self.model.disc_rate)
            
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
    
    def check_econ_attractiveness(self, ind_inv, lifetime_cashflows, discount_pp, max_pp, disc_rate):

        # Depending on the simulation specifications, use simple or discounted
        # payback period calculation:
        if discount_pp == True:

            # Compute discounted payback period per year simulated
            pp = self.compute_discounted_pp(ind_inv, lifetime_cashflows,
                                            max_pp, disc_rate)

        else:
            # Compute simple payback period per year simulated
            pp = self.compute_simple_pp(ind_inv, lifetime_cashflows, max_pp)

        # Compute normalized payback periods
        pp_norm = 1 - (pp / max_pp)

        return pp_norm

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
                    npv_ind = self.compute_npv(self.ind_inv,self.lifetime_cashflows, self.model.disc_rate)

                    # Store value
                    self.ind_npv = npv_ind

                    # If NPV community is larger than NPV individual
                    if npv_ind < npv_com:
                        
                        # Go for a solar community
                        self.consider_community_adoption(c_max_npv,
                            combinations_dict[c_max_npv], self.model.reduction, self.model.sim_year)                        
                    
                    # If NPV of individual adoption is greater than the
                    # NPV of the solar community alternative
                    elif npv_ind > npv_com:
                    
                        # Go for individual adoption
                        self.consider_individual_adoption(self.ind_inv, npv_ind, self.model.reduction, self.model.sim_year)
                
                # If there are no possible solar communities
                else:

                    # Calculate individual adoption NPV 
                    npv_ind = self.compute_npv(self.ind_inv, self.lifetime_cashflows, self.model.disc_rate)

                    # Store value
                    self.ind_npv = npv_ind

                    # Go for individual adoption
                    self.consider_individual_adoption(self.ind_inv, npv_ind, self.model.reduction, self.model.sim_year)
                        
            
            # If no agents in plot with intention and no individual solar yet
            elif ((len(potential_partners) == 0) or (self.model.com_allowed == False)) and self.adopt_ind == 0:

                # Calculate individual adoption NPV 
                npv_ind = self.compute_npv(self.ind_inv, self.lifetime_cashflows, self.model.disc_rate)

                # Store value
                self.ind_npv = npv_ind

                # Go for individual adoption
                self.consider_individual_adoption(self.ind_inv, npv_ind, self.model.reduction, self.model.sim_year)

    
    def consider_individual_adoption(self, ind_inv, npv_ind, reduction, sim_year):
        """
        This method determines if the agent adopts solar PV as an individual.

        Inputs
            self = agent (obj)
            ind_inv = investment for individual adoption (float)
            npv_ind = net-present value of individual adoption (float)
            reduction = fraction of inv agent tolerates to lose (float)
            sim_year = year in the simualtion (int)
        Returns
            None (agent attributes are updated directly)
        """

        # If the NPV of individual adoption is greater or equal to the maximum
        # losses that the agent is willing to tolerate (expressed as a fraction
        # of the investment cost), then adopt
        if npv_ind >= reduction * ind_inv:

            # Then, adopt individually and update the agent's attributes

            # Define agent as individual adopter of solar PV
            self.adopt_ind  = 1

            # Record the year of installation
            self.adopt_year = sim_year

            # Record agent's reason for adoption
            self.reason_adoption = "only_ind"

    def consider_community_adoption(self, c_max_npv, c_max_npv_dict, reduction,sim_year):
        """
        This method determines if the agent joins a solar community, by first
        checking if all members of the community agree.

        Inputs:
            c_max_npv = name of community with largest NPV (str)
            c_max_npv_dict = dict of community variables for community
                with highest NPV among all possible (dict)
            reduction = fraction of inv agent tolerates to lose (float)
            sim_year = year in the simualtion (int)

        Returns:
            None
        """

        # Store decision of each agent in list
        members_decision = []

        # Compute the threshold of each agent in the community
        for ag in c_max_npv_dict["members"]:
            
            # Compute threshold of losses tolerated by this agnet this year
            th_ag = (-1) * reduction * ag.ind_inv

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
                com_ag.adopt_year = sim_year
                
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
            closest_pps = np.array(d_df[uid].iloc[:(n_closest_neighbors + 1)].values)

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

            # Determine the community's electricity tariff
            # Following this paragraph: "As a rule, [the tariff used to compute the price charged to members of the solar community] does not correspond to the external electricity product that the community actually purchases (Art. 16 (1) (b) EnV), since the community, as the larger consumer, is no longer considered a household customer." (EnergieSchweiz, Leitfaden Eigenverbrauch, 2019, p.18). We consider communities enter the commercial tariffs and we assign them one depending on their size. This forces us to use two electricity prices -> one to compute the cost of the electricity bought from the grid *as a community* and one to compute the savings with the electricity tariff the agents had *as individual consumers*.
            c_dict["tariff"] = self.assign_community_tariff(np.sum(c_dict["demand"]), model.el_tariff_demands)

            # Compute NPV of the community
            c_dict["npv"] = self.calculate_com_npv(model.inputs, c_dict, 
                                                model.sim_year)
            
            """
            TO BE SOLVED: 
            (1) What happens with installed PV capacity?
            ---> At the moment, no PV inv and no SM costs of already installed
            """
    def assign_community_tariff(self, demand_yr, el_tariff_demands):
        """
        This method assigns the community a commercial electricity tariff based on the annual demand of the community.

        Inputs:
            demand_yr = annual electricity consumption of community (float)
            el_tariff_demands = dictionary with limits of demand for the different electricity tariffs (dict)
        Returns:
            el_tariff = name of tariff (str)
        """
        
        # List max demands for each tariff category
        t_ds = sorted(list(el_tariff_demands["commercial"].values()))

        # Find the index of the category whose demnad limit is higher than the annual demand of the building
        try:
            t_ix = next(ix for ix,v in enumerate(t_ds) if v > demand_yr)
        except:
            t_ix = len(t_ds)

        # Read the label of the tariff and return it
        com_tariff = sorted(list(el_tariff_demands["commercial"].keys()))[t_ix]

        return com_tariff
        
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

        More info: this method is based on the Art. 15 par 1 of the RS 730.01 which says "Grouping in the context of own consumption is permitted, provided that the production power of the installation or installations is at least 10% of the connection power of the grouping" https://www.admin.ch/opc/fr/official-compilation/2019/913.pdf         
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
        for v in ["pv_size", "pv_size_added", "n_sm", "n_sm_added", "npv", "tariff"]:
            c_export_dict[v] = c_dict[v]
        
        self.model.datacollector.add_table_row("communities", c_export_dict)
    
    def compute_lifetime_load_profile(self,solar_building, demand_ag, PV_lifetime, deg_rate, hour_price):
        """
        Inputs
            solar_outputs = hourly electricity output of PV system in the building for first year of its operational lifetime (list of 8760 items)
            demand_ag = hourly electricity demand of the building (list)
            PV_lifetime = years of operational life of installation (integer)
            deg_rate  = degression rate of PV output (float)
            hour_price = price level for each hour of the year (list)
        Returns
            lifetime_load_profile = description of annual energy balances over the operational lifetime of the PV installation of the buildign
                (dataframe with index = year of lifetime, columns = energy balances)
        """
        if deg_rate != 0:

            # Compute the hourly solar output AC for each operational year of PV
            solar_outputs = [solar_building * ((1 - deg_rate) ** y) for y in range(PV_lifetime)]

            for yr in range(PV_lifetime):

                # Create a dataframe with one row per hour of the year and one
                # column per building
                load_profile = pd.DataFrame(data = None, index = range(8760))

                # Create a dictionary to contain the annual energy balances
                load_profile_year = {} 
                
                # Define hourly solar system output for this building and hourly demand
                load_profile["solar"] = solar_outputs[yr]
                load_profile["demand"] = demand_ag

                # Define price of electricity per hour of the day
                load_profile["hour_price"] = hour_price

                # Compute hourly net demand from grid and hourly excess solar
                load_profile["net_demand"] = load_profile.demand - load_profile.solar
                load_profile["excess_solar"] = load_profile.solar - load_profile.demand

                # Remove negative values by making them zero
                load_profile["net_demand"] = np.array(
                    [x if x > 0 else 0 for x in load_profile["net_demand"]])
                load_profile["excess_solar"] = np.array(
                    [x if x > 0 else 0 for x in load_profile["excess_solar"]])

                # Compute hourly self-consumed electricity
                # For the hours of the year with solar generation: self-consume all
                # solar generation if less than demand (s) or up to demand (d)
                s = np.array(solar_outputs[yr])
                d = np.array(demand_ag)
                load_profile["sc"] = [min(s[i], d[i]) if s[i] > 0 else 0 for i in range(8760)]
                
                # Compute annual energy balances regardless of hour prices
                for bal in ["solar", "demand", "net_demand", "excess_solar", "sc"]:
                    load_profile_year[bal] = sum(load_profile[bal])
                
                # Compute annual energy balances for high and low price hours
                for bal in ["solar", "demand", "excess_solar", "net_demand", "sc"]:
                    for pl in ["high", "low"]:
                        cond = (load_profile["hour_price"] == pl)
                        load_profile_year[bal+'_'+pl] = sum(load_profile[bal].loc[cond])

                # Compute year self-consumption rate
                load_profile_year["SCR"] = 0
                if load_profile_year["sc"] > 0:
                    load_profile_year["SCR"] = load_profile_year["sc"] / load_profile_year["solar"]

                # Store results in return dataframe
                if yr == 0:
                    # If it is the first year, then create the dataframe
                    lifetime_load_profile = pd.DataFrame(load_profile_year, index=[0])
                else:
                    # Append the dictionary containing the results for this year
                    lifetime_load_profile = lifetime_load_profile.append(
                                                load_profile_year, ignore_index=True)
            
        # No degradation
        else:

            # Set solar output as output first year of lifetime
            solar_outputs = solar_building

            # Create a dataframe with one row per hour of the year and one
            # column per building
            load_profile = pd.DataFrame(data = None, index = range(8760))

            # Create a dictionary to contain the annual energy balances
            load_profile_year = {} 
            
            # Define hourly solar system output for this building and hourly demand
            load_profile["solar"] = solar_outputs
            load_profile["demand"] = demand_ag

            # Define price of electricity per hour of the day
            load_profile["hour_price"] = hour_price

            # Compute hourly net demand from grid and hourly excess solar
            load_profile["net_demand"] = load_profile.demand - load_profile.solar
            load_profile["excess_solar"] = load_profile.solar - load_profile.demand

            # Remove negative values by making them zero
            load_profile["net_demand"] = np.array(
                [x if x > 0 else 0 for x in load_profile["net_demand"]])
            load_profile["excess_solar"] = np.array(
                [x if x > 0 else 0 for x in load_profile["excess_solar"]])

            # Compute hourly self-consumed electricity
            # For the hours of the year with solar generation: self-consume all
            # solar generation if less than demand (s) or up to demand (d)
            s = solar_outputs
            d = demand_ag
            load_profile["sc"] = np.array([min(s[i], d[i]) 
                                        if s[i] > 0 else 0 for i in range(8760)])
            
            # Compute annual energy balances regardless of hour prices
            for bal in ["solar", "demand", "net_demand", "excess_solar", "sc"]:
                load_profile_year[bal] = sum(load_profile[bal])
            
            # Compute annual energy balances for high and low price hours
            for bal in ["solar", "demand", "excess_solar", "net_demand", "sc"]:
                for pl in ["high", "low"]:
                    cond = (load_profile["hour_price"] == pl)
                    load_profile_year[bal+'_'+pl] = sum(load_profile[bal].loc[cond])

            # Compute year self-consumption rate
            load_profile_year["SCR"] = 0
            if load_profile_year["sc"] > 0:
                load_profile_year["SCR"] = np.divide(load_profile_year["sc"], 
                                                        load_profile_year["solar"])

            # Store results in return dataframe
            lifetime_load_profile = pd.DataFrame(load_profile_year, index=[0])

            # Make results the same for all lifetime
            lifetime_load_profile = pd.concat([lifetime_load_profile] * PV_lifetime,ignore_index=True)

        return lifetime_load_profile

    def compute_lifetime_cashflows(self, ind_tariff, pv_size, lifetime_load_profile, deg_rate, PV_lifetime, sim_year, el_price, ratio_high_low, fit_high, fit_low, hist_fit, solar_split_fee, om_cost, sys = "ind", com_tariff = None):
        """
        This function computes the annual cashflows over the operational lifetime of the PV system in the building.

        Inputs
            lifetime_load_profile = description of annual energy balances over the operational lifetime of the PV installation of the buildign(dataframe with index = year of lifetime, columns = energy balances)
            sys = type of system (str) -> "ind" or "com"
            com_tariff = name of electricity tariff for community (str)

        Returns
            lifetime_cashflows = monetary flows into and out of the project for each year of its operational lifetime (dataframe, index = yr,columns = cashflow category)
        """
        # FEED-IN TARIFF

        # Before 2017 (end of FIT funds), we assume all go for federal FIT
        if sim_year > 2016:

            # Later, therere is only the EWZ feed-in tariff available
            fit_h = fit_high
            fit_l = fit_low

        # Use historical feed-in tariff per system size
        elif sim_year >= 2010:

            # Optimize this later
            if pv_size < 10:
                fit_h = hist_fit["10"][(sim_year - 2010)]
            elif pv_size < 30:
                fit_h = hist_fit["30"][(sim_year - 2010)]
            elif pv_size < 100:
                fit_h = hist_fit["100"][(sim_year - 2010)]
            elif pv_size < 1000:
                fit_h = hist_fit["1000"][(sim_year - 2010)]
            elif pv_size < 50000:
                fit_h = hist_fit["50000"][(sim_year - 2010)]
            else:
                fit_h = 0
            
            # No hour distinction in historic feed-in tariff
            fit_l = fit_h
        
        # ELECTRICITY PRICE

        # Define average electricity price for individual adoption
        el_p_ind = el_price[ind_tariff]

        # Set high and low electricity prices for individual adoption
        el_price_l = 0.25 + 0.75 * ratio_high_low * el_p_ind
        el_price_h = ratio_high_low * el_price_l
        
        # If this is for a community NPV, assign the electricity when in com
        if sys == "com":

            # Define av com electricity price
            el_p_com = el_price[com_tariff]

            # Set high and low individual electricity prices
            el_p_com_l = 0.25 + 0.75 * ratio_high_low * el_p_com
            el_p_com_h = ratio_high_low * el_p_com_l

        # ANNUAL CASHFLOW CALCULATION

        # Create empty dictionary to store annual cashflows
        cf_y = {}

        # Check if load profile changes over the lifetime or not
        if deg_rate != 0:

            # Loop through the years of operational life of the system
            for y in range(PV_lifetime):

                # Read year excess solar
                ex_pv_h = np.array(lifetime_load_profile["excess_solar_high"][y])
                ex_pv_l = np.array(lifetime_load_profile["excess_solar_low"][y])
                
                # Compute the revenues from feeding solar electricity to the grid
                cf_y["FIT"] = ex_pv_h * fit_h + ex_pv_l * fit_l

                # Read avoided consumption from the grid (i.e. self-consumption)
                sc_h = np.array(lifetime_load_profile["sc_high"][y])
                sc_l = np.array(lifetime_load_profile["sc_low"][y])

                # Compute the savings from self-consuming solar electricity
                if sys == "ind":

                    # Savings only from avoided consumption from grid
                    cf_y["savings"] = sc_h * el_price_h + sc_l * el_price_l

                elif sys == "com":

                    # Savings from avoided consumption from grid *with old tariff* and from moving to a cheaper electricity tariff because now a single bigger consumer
                    cf_y["savings"] = sc_h * el_price_h + sc_l * el_price_l + np.array(lifetime_load_profile["net_demand_high"][y]) * (el_price_h - el_p_com_h) + np.array(lifetime_load_profile["net_demand_high"][y]) * (el_price_l - el_p_com_l)
                
                    # Compute the cost of individual metering
                    cf_y["split"] = (sc_h + sc_l) * solar_split_fee

                # Compute O&M costs
                cf_y["O&M"] = np.array(lifetime_load_profile["solar"][y]) * om_cost

                # Compute net cashflows to the agent
                if sys == "ind":
                    cf_y["net_cf"] = (cf_y["FIT"] + cf_y["savings"] - cf_y["O&M"])

                elif sys == "com":
                    cf_y["net_cf"] = (cf_y["FIT"] + cf_y["savings"] - cf_y["split"]- cf_y["O&M"])

                # Store results in return dataframe
                if y == 0:
                    # If it is the first year, then create the dataframe
                    lifetime_cashflows = pd.DataFrame(cf_y, index=[0])
                else:
                    # Append the dictionary containing the results for this year
                    lifetime_cashflows = lifetime_cashflows.append(cf_y, ignore_index=True)
            
        else:

            # Without degradation, all years have the same profile so we just take the first one and copy the results over the lifetime of the system.

            # Read year excess solar
            ex_pv_h = np.array(lifetime_load_profile["excess_solar_high"][0])
            ex_pv_l = np.array(lifetime_load_profile["excess_solar_low"][0])
            
            # Compute revenues from feeding solar electricity to the grid
            cf_y["FIT"] = ex_pv_h * fit_h + ex_pv_l * fit_l

            # Read avoided consumption from grid (i.e. self-consumption)
            sc_h = np.array(lifetime_load_profile["sc_high"][0])
            sc_l = np.array(lifetime_load_profile["sc_low"][0])

            # Compute the savings from self-consuming solar electricity
            if sys == "ind":

                # Savings only from avoided consumption from grid
                cf_y["savings"] = sc_h * el_price_h + sc_l * el_price_l

            elif sys == "com":

                # Savings from avoided consumption from grid *with old tariff* and from moving to a cheaper electricity tariff because now a single bigger consumer
                cf_y["savings"] = sc_h * el_price_h + sc_l * el_price_l + np.array(lifetime_load_profile["net_demand_high"][0]) * (el_price_h - el_p_com_h) + np.array(lifetime_load_profile["net_demand_high"][0]) * (el_price_l - el_p_com_l)
            
                # Compute the cost of individual metering
                cf_y["split"] = (sc_h + sc_l) * solar_split_fee

            # Compute O&M costs
            cf_y["O&M"] = np.array(lifetime_load_profile["solar"][0]) * om_cost

            # Compute net cashflows to the agent
            if sys == "ind":
                cf_y["net_cf"] = (cf_y["FIT"] + cf_y["savings"] - cf_y["O&M"])

            elif sys == "com":
                cf_y["net_cf"] = (cf_y["FIT"] + cf_y["savings"] - cf_y["split"]- cf_y["O&M"])

            # Store results in return dataframe
            lifetime_cashflows = pd.DataFrame(cf_y, index=[0])

            # Make results the same for all lifetime
            lifetime_cashflows = lifetime_cashflows.append([cf_y] * PV_lifetime, ignore_index=True)
    
        return lifetime_cashflows

    def compute_smart_meters_inv(self, n_sm, smp_dict):
        """
        This function takes the number of smart meters and their prices per 
        number in the installation of the building and provides the investment
        cost for the system.

        Inputs
            n_sm = number of smart meters in the building (integer)
            smp_dict = price per smart meter depending on number of sm (dict)
        
        Returns
            sm_inv = investment cost of smart meters (float)
        """
        
        # Convert the keys in the smp_dict into a list of integers that
        # indicate the maximum number of meters to receive that price
        smp_cats = [int(x) for x in list(smp_dict.keys())]

        # Try to find a number of smart meters in the price categories that
        # is larger then the n_sm. If you don't find any, then
        # use the lowest price category (i.e. for more than 50 meters)
        try:
            smp_ix = next(ix for ix, v in enumerate(smp_cats) if v > n_sm)
        except StopIteration:
            smp_ix = len(smp_cats)

        # Estimate investment cost of smart meters
        sm_inv = n_sm * smp_dict[str(smp_cats[smp_ix - 1])]

        return sm_inv

    def compute_pv_inv(self, pv_size, pv_price, pv_scale_alpha, pv_scale_beta, sim_year, base_d, pot_30_d, pot_100_d, pot_100_plus_d):
        """
        This function calculates the investment cost of the PV system based on 
        its size, and the price for that size category for the current year.

        Inputs
            pv_size = size of the installation (float)
            pv_price = PV price for ref size (float)
            pv_scale_alpha, pv_scale_beta = parameters of the scale effects in the price of PV systems, following the relation calculated from empirical data in Switzerland: alpha * pv_size ^ beta
            sim_year = year in the simulation (int)

        Returns
            pv_inv = investment cost of PV system (float)
        """
        # Calculate the scale effects for this agent's PV system
        if pv_size > 1:
            scale_effect = pv_scale_alpha * pv_size ** pv_scale_beta
        else:
            scale_effect = 2

        # Compute the price for this system based on projection for ref size
        pv_price_size = pv_price * scale_effect 

        # Estimate investment cost of PV system
        pv_inv = pv_size * pv_price_size

        # Compute investment subsidy for the agent
        pv_sub = self.compute_pv_sub(pv_size, sim_year, base_d, pot_30_d, pot_100_d, pot_100_plus_d)

        # Apply subsidy for installation
        pv_inv = pv_inv - pv_sub
        
        if pv_inv < 0:
            print("PROBLEMO WITH SUBSIDIES")

        return pv_inv

    def compute_npv(self, inv, lifetime_cashflows, disc_rate):
        """
        This function provides the net-present value of installation for one
        building or community for simulation year "yr".

        Inputs
            inv = investment cost of the installation (float)
            lifetime_cashflows = annual cashflows over lifetime of the system (df)
                (index = year of lifetime, column = cashflow)
            disc_rate = time value of money for the agent (float)

        Returns
            npv = NPV for this installation and year of the simulation (float)
        """

        # Start a list with cashflows with negative inv cost
        cf = [- inv]

        # Add the net cashflows for the operational life of the syst to list
        cf.extend(list(lifetime_cashflows["net_cf"].values))

        # Compute NPV if installation occurs this year
        npv = np.npv(disc_rate, cf)

        return npv

    def compute_simple_pp(self, inv, lifetime_cashflows, max_pp):
        """
        This function computes the simple payback period (without time-discounted values) for the year for this building.
        
        Inputs
            inv = investment cost of the installation (floa)
            lifetime_cashflows = annual cashflows over the lifetime of the system (df) (index = year of lifetime, column = cashflow)
            max_pp = maximum payback period considered by agents (integer)

        Returns
            pp = simple payback period (float)
        """

        # Sum up the cashflows over the lifetime of the installation
        cf = lifetime_cashflows["net_cf"][0]

        # Compute payback period
        if cf > 0:
            pp = min(inv / cf, max_pp)
        else:
            pp = max_pp

        return pp

    def compute_discounted_pp(self, inv, lifetime_cashflows, max_pp, disc_rate):
        """
        This function computes the simple payback period (without time-discounted values) for current year for this building.
        
        Inputs
            inv = investment cost of the installation (float)
            lifetime_cashflows = annual cashflows over the lifetime of the system (df) (index = year of lifetime, column = cashflow)
            max_pp = maximum payback period considered by agents (integer)
            disc_rate = time value of money for the agent (float)

        Returns
            dpp = discounted payback period (float)
        """
        # Start a list with cashflows for this year with negative inv cost
        dcf = [- inv]

        # Add the net cashflows for the operational life of the syst to list
        dcf.extend(list(lifetime_cashflows["net_cf"].values))

        # Discount cashflows
        dcf = [dcf[y] / (1 + disc_rate) ** y for y in range(len(dcf))]

        # Cumulate cashflows
        cum_dcf = [dcf[i] + sum(dcf[:i]) for i in range(len(dcf))]

        # Compute discounted cash-flow by looking when cum_dcf changes sign
        try:
            dpp = next(ix for ix, v in enumerate(cum_dcf) if v > 0)

        # If it nevers become positive, assign max_pp
        except:
            dpp = max_pp

        return dpp

    def calculate_com_npv(self, inputs, c_dict, sim_year):
        '''
        This function computes the economic parameters for the evaluation of
        the formation of a solar community.

        Inputs
            inputs = simulation parameters (dict)
            c_dict = community parameters (dict)
            year = simulation year (int)

        Returns
            com_npv_outputs = contains a dictioanry of dataframes with results:
                "Agents_NPVs" = NPV per agent per year of simulation (df)
                "Agents_Investment_Costs" = total inv cost per sim year per agent (df) 
                "Agents_PPs_Norm" = normalized pp per sim year per agent (df)
                "Agents_SCRs" = self-consumption rate per agent per operation year
                    of PV system in the building (df)       
        '''
        # DEFINE ECONOMIC PARAMETERS

        # COMMUNITY SPECIFIC

        # Define the newly installed PV capacity
        pv_size = c_dict["pv_size_added"]

        # Define the number of newly installed smart meters
        n_sm = c_dict["n_sm_added"]

        # Define community's generation potential (already AC)
        solar = c_dict["solar"]

        # Define community's aggregated consumption
        demand = c_dict["demand"]

        # Define the community's electricity tariff
        com_tariff = c_dict["tariff"]

        # NON-COMMUNITY SPECIFIC

        # Compute the load profile of the installation throughout its lifetime
        lifetime_load_profile = self.compute_lifetime_load_profile(solar, demand, self.model.pv_lifetime, self.model.deg_rate, self.model.hour_price)

        # Compute hte cashflows of the installation throughout its lifetime
        lifetime_cashflows = self.compute_lifetime_cashflows(self.el_tariff, pv_size, lifetime_load_profile, self.model.deg_rate,self.model.pv_lifetime, self.model.sim_year, self.model.el_price, self.model.ratio_high_low, self.model.fit_high, self.model.fit_low, self.model.hist_fit, self.model.solar_split_fee,self.model.om_cost, sys="com", com_tariff=com_tariff)
        
        # Compute the investment cost of the PV system
        pv_inv = self.compute_pv_inv(pv_size, self.model.pv_price, self.model.pv_scale_alpha, self.model.pv_scale_beta, self.model.sim_year, self.model.base_d, self.model.pot_30_d, self.model.pot_100_d, self.model.pot_100_plus_d)
        
        # Compute the investment cost of smart meters
        sm_inv = self.compute_smart_meters_inv(n_sm, self.model.smp_dict)
        
        # Compute the cooperation costs
        coop_cost = 0
        
        # Estimate the total investment cost
        inv = pv_inv + sm_inv + coop_cost

        # Compute the community's NPV
        npv_com = self.compute_npv(inv, lifetime_cashflows, self.model.disc_rate)

        return npv_com
    
    def compute_pv_sub(self, pv_size, sim_year, base_d, pot_30_d, pot_100_d, pot_100_plus_d):
        """
        This method computes the investment subsidy applicable to the installation.

        Inputs
            pv_size = size of the PV system (float)
            sim_year = year in the simulation (int)
        Returns
            pv_sub = lump-sum subsidy in CHF (float)
        
        More info: since 2013, the Swiss government made available a one-time investment subsidy for new PV installations. The introduction of this measure was motivated by the long waiting list to obtain a feed-in tariff subsidy. The investment subsidy was made available to systems of different sizes at different points in times, and it also applied to registered installtions waiting for the feed-in tariff (which makes it ambiguous to pinpoint the exact time the policy began since installtions prior to the entry into force of the law could apply for it). We simplify by taking one single value per year, and assuming only installations taking place after the entry in force of the legislation can apply to it.
        """
        # 2014 -> (<30kW) base 1400, vol 850 730.01, App 1.8, 3.1
        # Systems <10 kW forced to take investment subsidy, no more FIT
        # Systems 10-30 kW can choose to get investment subsidy or FIT
        # https://www.admin.ch/opc/fr/classified-compilation/19983391/201404010000/730.01.pdf

        # DATA FROM https://pronovo.ch/fr/financement/systeme-de-retribution-de-linjection-sri/retribution/

        # In law: https://www.admin.ch/opc/fr/classified-compilation/20162947/index.html#app6ahref3 
        # RS 730.03 Annexe 2.1

        # Initialize subsidy to zero
        pv_sub = 0

        if (sim_year > 2013) and (sim_year <= 2030):

            # Chapter 4 art 36 https://www.admin.ch/opc/fr/classified-compilation/20162947/index.html
            if (pv_size > 2) and (pv_size < 30):

                try:
                    pv_sub =  base_d[str(sim_year)] + pot_30_d[str(sim_year)] * pv_size
                except:
                    pv_sub = min(base_d.values()) + min(pot_30_d.values()) * pv_size

            elif 30 <= pv_size < 100:

                try:
                    pv_sub =  base_d[str(sim_year)] + pot_30_d[str(sim_year)] * 30 + pot_100_d[str(sim_year)] * (pv_size - 30)
                except:
                    pv_sub = min(base_d.values()) + min(pot_30_d.values()) * 30 + min(pot_100_d.values()) * (pv_size - 30)

            elif (pv_size >= 100) and (pv_size < 50000):
                
                # Based on https://pronovo.ch/fr/financement/systeme-de-retribution-de-linjection-sri/
                if (sim_year > 2018):
                    try:
                        pv_sub =  base_d[str(sim_year)] + pot_30_d[str(sim_year)] * 30 + pot_100_d[str(sim_year)] * 70 + pot_100_plus_d[str(sim_year)] * (pv_size - 100)
                    except:
                        pv_sub = min(base_d.values()) + min(pot_30_d.values()) * 30 + min(pot_100_d.values()) * 70 + min(pot_100_plus_d.values()) * (pv_size - 100)
                else:
                    pv_sub = 0

        return pv_sub