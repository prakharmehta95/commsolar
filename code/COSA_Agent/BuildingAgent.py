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

        # Initialize PV system tracker
        self.pv_installation = False

        # Initialize year of pv installation
        self.pv_installation_year = 0

        # Initialize pv installation investment cost at time of adoption
        self.pv_installation_cost = 0

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

        # Initialize individual investment subsidy to zero
        self.ind_inv_sub = 0

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
            self.peer_effect = self.check_peers(self.peers, self.model.schedule.agents)

            # Evaluate the persuasion effect from neighbors
            self.neighbor_influence = self.check_neighbors_influence()

            # Evaluate investment subsidy
            self.ind_inv_sub = self.compute_pv_sub(self.pv_size, self.model.sim_year, self.model.base_d, self.model.pot_30_d, self.model.pot_100_d, self.model.pot_100_plus_d)

            # Update investment cost for this year
            self.ind_inv = self.sm_inv + self.compute_pv_inv(self.pv_size, self.ind_inv_sub, self.model.pv_price, self.model.pv_scale_alpha, self.model.pv_scale_beta)

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

    def check_peers(self, peers, all_agents):
        """
        Determines the impact of peer effects in the agent.

        Checks the number of peers (agents in the network of current agent)
        that have installed solar and updates agent attribute peer_effect
        (self.peer_effect) accordingly
        
        Inputs:
            peers = list of agent ids connected to this agent (list of str)
            all_agents = list of agents in the model (list of objs)
        Returns:
            peer_effect = fraction of contacts with solar installed (float)
        """
        
        # Initialize output variable for peer-effects
        peer_effect = 0

        # Determine the number of contacts in the network of current agent
        n_peers = len(peers)

        # If the agent has any peers, then count how many have solar
        if n_peers != 0:

            # Create a list of agents that are peers of this one
            peers_list = [ag for ag in all_agents if ag.unique_id in peers]

            # Sum up the number of peers with solar
            n_peers_solar = sum([1 if (ag.pv_installation == True) else 0 for ag in peers_list])
            
            # Determine peer effects
            peer_effect = n_peers_solar / n_peers

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
        neighbors_list = np.array([ag for ag in all_agents if (ag.bldg_plot == self.bldg_plot) and (ag.unique_id != self.unique_id)])

        # If there are any neighbors (own agent counts as 1)
        if len(neighbors_list) > 0:
            
            # Compute the number of neighbors with intention to adopt
            neighbors_idea = np.sum([1 if ((ag.intention == 1) or (ag.adopt_comm == 1)) else 0 for ag in neighbors_list])

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

            # Create a list containing all the agents in the plot with the idea to adopt PV or that already have PV, who are not in community
            potential_partners = [ag for ag in self.model.schedule.agents if (
                                    (ag.bldg_plot == self.bldg_plot) and (
                                    (ag.intention == 1) or (ag.adopt_ind == 1))
                                    and (ag.adopt_comm == 0) and (ag.unique_id != self.unique_id))]

            # Create a list of formed communities the agent could join
            potential_communities = list(set([ag.com_name for ag in self.model.schedule.agents if ((ag.bldg_plot==self.bldg_plot) and (ag.adopt_comm==1) and (ag.unique_id != self.unique_id))]))

            # Compute the number of community options
            if self.model.join_com == True:
                n_com_options = len(potential_partners) + len(potential_communities)
            else:
                n_com_options = len(potential_partners)
                
            # If agents in plot with idea to install and communities allowed
            if (n_com_options > 1) and (self.model.com_allowed == True):

                # Define what community options the agent considers
                agents_to_consider = self.define_agents_to_consider(potential_partners, potential_communities, self.unique_id, self.distances, self.model.n_closest_neighbors)

                # Define what possible communities could be formed
                combinations_dict = self.define_possible_coms(agents_to_consider)

                # Evaluate the characteristics of each possible community
                self.update_combinations_available(self.model, combinations_dict)
                
                # Compare community vs individual (if there are combinations)
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
                    npv_ind = self.compute_npv(self.ind_inv,self.lifetime_cashflows, self.model.disc_rate, self.model.sim_year)

                    # Store value
                    self.ind_npv = npv_ind

                    # If NPV community is larger than NPV individual
                    if npv_ind < npv_com:
                        
                        # Go for a solar community
                        self.consider_community_adoption(c_max_npv,combinations_dict[c_max_npv], self.model.reduction, self.model.sim_year)                        
                    
                    # If NPV of individual adoption is greater than the
                    # NPV of the solar community alternative
                    elif npv_ind > npv_com:
                    
                        # Go for individual adoption
                        self.consider_individual_adoption(self.ind_inv, npv_ind, self.ind_inv_sub, self.model.reduction, self.model.sim_year)
                
                # If there are no possible solar communities
                else:

                    # Calculate individual adoption NPV 
                    npv_ind = self.compute_npv(self.ind_inv, self.lifetime_cashflows, self.model.disc_rate, self.model.sim_year)

                    # Store value
                    self.ind_npv = npv_ind

                    # Go for individual adoption
                    self.consider_individual_adoption(self.ind_inv, npv_ind, self.ind_inv_sub, self.model.reduction, self.model.sim_year)
                        
            
            # If no agents in plot with intention and no individual solar yet
            elif ((len(potential_partners) == 0) or (self.model.com_allowed == False)) and self.adopt_ind == 0:

                # Calculate individual adoption NPV 
                npv_ind = self.compute_npv(self.ind_inv, self.lifetime_cashflows, self.model.disc_rate, self.model.sim_year)

                # Store value
                self.ind_npv = npv_ind

                # Go for individual adoption
                self.consider_individual_adoption(self.ind_inv, npv_ind, self.ind_inv_sub, self.model.reduction, self.model.sim_year)

    
    def consider_individual_adoption(self, ind_inv, npv_ind, ind_inv_sub, reduction, sim_year):
        """
        This method determines if the agent adopts solar PV as an individual.

        Inputs
            self = agent (obj)
            ind_inv = investment for individual adoption (float)
            npv_ind = net-present value of individual adoption (float)
            ind_inv_sub = investment subsidy for individual adoption (float)
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

            # Increase policy cost
            self.model.pol_cost_sub_ind += ind_inv_sub

            # Define pv_installation as true
            self.pv_installation = True

            # Record installation year
            self.pv_installation_year = sim_year

            # Record the cost of the installation
            self.pv_installation_cost = ind_inv

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

            # Increase policy cost
            self.model.pol_cost_sub_com += c_max_npv_dict["pv_sub"]

            # Loop through all the agents in the community
            for com_ag in c_max_npv_dict["members"]:
                
                #manually setting adopt_ind to zero to avoid double counting
                com_ag.adopt_ind = 0
                
                #setting community adoption as 1 for all agents involved
                com_ag.adopt_comm = 1
                        
                # Record the year of installation
                com_ag.adopt_year = sim_year

                # Record the name of the community
                com_ag.com_name = c_max_npv
                
                # Record agent's variables in info dataframe
                com_ag.reason_adoption = "Comm>Ind"

                # Update PV insallation if necessary
                if com_ag.pv_installation == False:
                    
                    # Set PV installation true
                    com_ag.pv_installation = True

                    # Record year of insallation
                    com_ag.pv_installation_year = sim_year

                    # Set cost of installation
                    com_ag.pv_installation_cost = c_max_npv_dict["inv_new"]

            # Save data about formed community
            self.save_formed_community(c_max_npv, c_max_npv_dict)

        # If other agents don't accept community, fall back to individual
        else:

            self.consider_individual_adoption(self.ind_inv, self.ind_npv, self.ind_inv_sub, self.model.reduction, self.model.sim_year)     
        
    def define_agents_to_consider(self, potential_partners, potential_communities, uid, distances, n_closest_neighbors):
        """
        Defines list of potential communities to consider.

        Inputs

        Returns
            agents_to_consider = list of buildings to consider for forming a community (list of ids)
        """

        # Place all agents that could be potential partners in a list
        agents_to_consider = potential_partners

        # If there is any potential community, add the closest agent of each potential community to the list
        if len(potential_communities) > 0:
            
            # List the agents in each potential community to join
            com_ags = [com.split("_") for com in potential_communities]

            # List the closest agents per community
            for com_ag_ids in com_ags:

                # Create a distances dataframe only of the agents in community
                d_df = distances[distances[uid].isin(com_ag_ids)]

                # Check if the dataframe is empty
                if len(d_df) == 0:
                    
                    # If empty dataframe, pick the first agent
                    agents_to_consider.extend([ag for ag in self.model.schedule.agents if ag.unique_id == com_ag_ids[0]])

                else:

                    # Set the names of the buildings as the index
                    d_df = d_df.set_index(uid)

                    # Find closest agent in community and append to list
                    agents_to_consider.extend([ag for ag in self.model.schedule.agents if ag.unique_id == d_df.idxmin().values[0]])
        
        # Check if more options to consider than limit
        if len(agents_to_consider) > n_closest_neighbors:

            # Create dataframe with only the agents in the list
            d_df = self.distances[self.distances[uid].isin(potential_partners)]

            # List the n_closest_neighbors 
            agents_to_consider = [ag for ag in agents_to_consider if ag.unique_id in d_df.sort_values(by = ['dist_' + uid]).iloc[:n_closest_neighbors].index]

        return agents_to_consider

    def define_possible_coms(self, agents_to_consider):
        """
        Creates a dictionary of all possible communities of all possible sizes, where the key is the name of the community and the value is a dictionary of the community characteristics.

        Inputs:
            self = active agent (obj)
            agents_to_consider = available agents to form community (list objs)

        Returns:
            combinations_dict = dictionary of all possible communities with
                key = community name (all unique_id of agents joined with "_") and value = dictionary of community properties (the only item is: key = "members", value = list of agents in community (list objs))
        """

        # Filter individual buildings and buildings in communities
        ags_alone = [ag for ag in agents_to_consider if ag.adopt_comm == 0]
        ags_in_com = [ag for ag in agents_to_consider if ag.adopt_comm == 1]
        
        # Create empty list to store all possible communities
        coms_list = []

        # Create empty dictionary to store the possible communities by name
        coms_dict = {}

        # Loop through all possible community sizes (i.e. number of members)
        for com_size in np.arange(1,len(ags_alone)+1):

            # Create a list of tuples, each one a combination possible
            coms_list.extend(list(itertools.combinations(ags_alone, com_size)))
        
        # Remove repeated buildings in each combination
        coms_list = [tuple(set(c)) for c in coms_list]

        # Remove repeated combinations
        coms_list = list(set(coms_list))

        # Add active agent to each combination (as tuple)
        coms_list = [c+(self,) for c in coms_list]

        # Create dictionary of combinations
        for com_members in coms_list:

            # Create the community name
            com_name = '_'.join([ag.unique_id for ag in com_members])

            # Create a dictionary for storing this community parameters
            coms_dict[com_name] = {}

            # Store community in dictionary
            coms_dict[com_name]["members"] = com_members
        
        # Add the agent to existing communities available
        for ag_in_com in ags_in_com:

            # Create the name of the new community
            com_name = '_'.join([ag_in_com.com_name,self.unique_id])

            # Create a dictionary for storing this community parameters
            coms_dict[com_name] = {}

            # List members
            com_members = [ag for ag in self.model.schedule.agents if ag.unique_id in com_name.split("_")]

            # Store community in dictionary
            coms_dict[com_name]["members"] = com_members

        return coms_dict

    def update_combinations_available(self, model, combinations_dict):
        """
        This method completes the characteristics of all possible communities
        by directly updating the combinations_dict.
        """
        # IMPORTANT: if there is any agent with PV already:
        # (1) PV investment is the sum of agents with no prior PV
        # (2) Smart meter investmetns is sum of agents with no prior PV
        # IMPORTANT 2: PV and smart meter investment needs to be computed for the whole community to realize scale effects on prices

        # Following this paragraph: "As a rule, [the tariff used to compute the price charged to members of the solar community] does not correspond to the external electricity product that the community actually purchases (Art. 16 (1) (b) EnV), since the community, as the larger consumer, is no longer considered a household customer." (EnergieSchweiz, Leitfaden Eigenverbrauch, 2019, p.18). We consider communities enter the commercial tariffs and we assign them one depending on their size. This forces us to use two electricity prices -> one to compute the cost of the electricity bought from the grid *as a community* and one to compute the savings with the electricity tariff the agents had *as individual consumers*.

        # List communities to delete because insufficient generation
        coms_below_ratio = []

        # Loop through combinations_dict to fill-in community attributes
        for c, c_d in combinations_dict.items():

            # Create list of agents members of the community
            members = np.array(c_d["members"])

            # Compute solar generation potential of community by summing along
            # the rows the columns in solar of the agents in community
            c_d["solar"] = np.nansum([ag.solar for ag in members], axis=0)

            # Compute demand of community
            c_d["demand"] = np.nansum([ag.demand for ag in members], axis=0)

            # Compute ratio of solar generation to demand
            if np.nansum(c_d["demand"]) != 0:

                # Divide annual solar generation over annual demand
                c_sd = np.nansum(c_d["solar"]) / np.nansum(c_d["demand"])
            
            # Note: this code also prevents the formation of communities with no solar generation or no demand.

            # More info: this is based on the Art. 15 par 1 of the RS 730.01 which says "Grouping in the context of own consumption is permitted, provided that the production power of the installation or installations is at least 10% of the connection power of the grouping" https://www.admin.ch/opc/fr/official-compilation/2019/913.pdf

            # If the community has zero demand, it cannot be formed
            else:
                c_sd = 0

            # Compare the community's ratio
            if ((c_sd < model.min_ratio_sd) or np.isnan(c_sd) or np.isinf(c_sd)):

                # If ratio too small, put in list to remove
                coms_below_ratio.append(c)
            
            # If ratio is big enough, then continue estimating parameters
            else:

                # Compute total community size PV
                c_d["pv_size"] = sum([ag.pv_size for ag in members])

                # Compute added PV (in case any agent has already PV)
                c_d["pv_size_added"] = sum([ag.pv_size for ag in members if ag.pv_installation == False])
                
                # Compute total number of smart meters in the community
                c_d["n_sm"] = sum([ag.n_sm for ag in members])

                # Compute added smart meters (in case any agent has already PV)
                c_d["n_sm_added"] = sum([ag.n_sm for ag in members if ag.pv_installation == False])

                # Determine the community's electricity tariff
                c_d["tariff"] = self.assign_community_tariff(np.sum(c_d["demand"]), model.el_tariff_demands)

                # If any member has pre-existing PV:
                if any([ag.pv_installation for ag in members]):

                    # Create list of members with indiviudal pre-existing PV
                    members_ind_pv = [ag for ag in members 
                    if ((ag.pv_installation == True) and (ag.adopt_ind == 1))]
                    
                    # Create list of members with community pre-existing PV
                    members_com_pv = [ag for ag in members 
                    if ((ag.pv_installation == True) and (ag.adopt_com == 1))]

                    # Create a list with each item is the hourly solar production of one year of the pv_lifetime of the community installation taking into account that prior installations are dismantled when they reach their end of life.
                    c_d["solar_prior"] = [np.sum([ag.solar 
                    for ag in members 
                    if ((ag.pv_installation_year + model.pv_lifetime > yr)
                    or (ag.pv_installation == False))], axis=0)
                    for yr in range(model.sim_year, model.sim_year + model.pv_lifetime)]
                    # Note: for each member in the community being evaluated, the solar output throught the year is added only if (a) the agent is installing a new installation, or (b) the prior existing installation is within its operational lifetime.

                    # For agents that join the community now with existing PV, compute the current value of their invesmtent wiht a linear depretiation process and add them up
                    c_d["present_inv_ind"] = np.sum([(
                        ag.pv_installation_cost * (1 - (model.sim_year - ag.pv_installation_year) / self.model.pv_lifetime))
                        for ag in members_ind_pv])
                    
                    # For agents that join the community as part of an existing community, compute the current value of the existing community PV with a linear depretiation process and add them up
                    c_d["present_inv_com"] = np.sum(list(set([(
                        ag.pv_installation_cost * (1 - (model.sim_year - ag.pv_installation_year) / self.model.pv_lifetime))
                        for ag in members_com_pv])))
                    # Explanation: each agent already in a community has stored the year when they installed on their rooftops pv_installation_year and how much they invested in it in pv_installation_cost (which could be individually, if they joined being grid prosumers, or could be the new investment in the commmunity if they joined as grid consumers). For the agents that joined the community as grid consumers, the stored pv_installation_cost is the inv_new (this is, the cost of adding new PV and smart meters) of the whole community. Since these agents store the same pv_installation_year and pv_installation_cost, using set() removes duplicated values and ensures we only count them once.
                    # WARNING: these removes duplicates of pv_installation_costs NOT based on agents part of the same community. There is a risk of under counting if e.g., there are members from two communities who installed the same PV size in the same year.

                    c_d["inv_old"] = c_d["present_inv_ind"] + c_d["present_inv_com"]

                else:
                    c_d["inv_old"] = 0

                # Compute NPV and pv_sub of the community
                npv_c, pv_sub_c, inv_c_new, pp_c = self.calculate_com_npv(model.inputs, c_d, model.sim_year)

                # Store the economic parameters of the community
                c_d["npv"] = npv_c
                c_d["pv_sub"] = pv_sub_c
                c_d["inv_new"] = inv_c_new
                c_d["pp_com"] = pp_c

        # Loop through the communities below ratio and delete them from dict
        for c in coms_below_ratio:
            del combinations_dict[c]

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

        # If the community consumes more than 100 MWh/yr - it can sell/buy electricity in the wholesale electricity market. Then, assign the wholesale price
        if self.model.direct_market == True:

            if demand_yr > self.model.direct_market_th:
                com_tariff = "wholesale"

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

        # Define community self-sufficiency ratio
        c_export_dict["SSR"] = c_export_dict["SC"] / c_export_dict["demand"]

        # Define PV size and profitability values
        for v in ["pv_size", "pv_size_added", "n_sm", "n_sm_added", "npv", "tariff", "npv", "pv_sub", "inv_new", "inv_old", "pp_com"]:
            c_export_dict[v] = c_dict[v]
        
        self.model.datacollector.add_table_row("communities", c_export_dict)
    
    def compute_lifetime_load_profile(self,solar_building, demand_ag, PV_lifetime, deg_rate, hour_price, com_prior_PV=False):
        """
        Inputs
            solar_outputs = hourly electricity output of PV system in the building for first year of its operational lifetime (list of 8760 items)
            demand_ag = hourly electricity demand of the building (list)
            PV_lifetime = years of operational life of installation (integer)
            deg_rate  = degression rate of PV output (float)
            hour_price = price level for each hour of the year (list)
            com_prior_PV = computation for a community with existing PV (boolean, False by default)
        Returns
            lifetime_load_profile = description of annual energy balances over the operational lifetime of the PV installation of the buildign
                (dataframe with index = year of lifetime, columns = energy balances)
        """
        if (deg_rate != 0) or (com_prior_PV == True):

            if deg_rate != 0:
                # Compute hourly solar output AC for each operational year
                solar_outputs = [solar_building * ((1 - deg_rate) ** y) for y in range(PV_lifetime)]
            
            elif com_prior_PV == True:
                # Hourly solar output already a list accounting for prior installations reaching end of life
                solar_outputs = solar_building

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

                # Compute hourly self-consumed electricity. For the hours of the year with solar generation: self-consume all solar generation if less than demand (s) or up to demand (d)
                s = solar_outputs[yr]
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
                    load_profile_year["SCR"] = np.divide(np.sum(load_profile_year["sc"]),np.sum(load_profile_year["solar"]))

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
            
            # Store energy balances regardless of hour prices
            for bal in ["solar", "demand", "net_demand", "excess_solar", "sc"]:
                load_profile_year[bal] = sum(load_profile[bal])
            
            # Compute annual energy balances for high and low price hours
            for bal in ["solar", "demand", "excess_solar", "net_demand", "sc"]:
                for pl in ["high", "low"]:
                    cond = (load_profile["hour_price"] == pl)
                    load_profile_year[bal+'_'+pl] = sum(load_profile[bal].loc[cond])

            # Compute year self-consumption rate
            load_profile_year["SCR"] = 0
            if np.sum(load_profile_year["sc"]) > 0:
                load_profile_year["SCR"] = np.divide(np.sum(load_profile_year["sc"]),np.sum(load_profile_year["solar"]))

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

        WARNING: energy balances with "_high" or "_low" are already summed up
        per year, while energy balances without them (e.g., "demand") are a numpy array with hourly value for each year of the lifetime.

        INCOMPLETE: the version with solar output degradation is NOT complete

        IMPORTANT: Since 2013, systems above 10 kWp could choose between the feet-in tariff and the investment subsidy. Because the feed-in tariff was generally more profitable, we assume all installations opted for the feed-in tariff.

        POLICY CHANGES: Since 2018, installations above 100 kWp that had taken the federal FIT had the obligation to market their electricity directly (turning the FIT into a feed-in premium plus a management fee). Supposedly, all new PV installations >100 kWp that opted for the FIT would have to market directly but (all in all, the revenue would be guaranteed by the feed-in premum). These installations could not take the investment subsidy. But the waiting list is so long that I think no new installations would get the FIT. ASSUMPTION: After 2018, all installations take the investment subsidy.
        """
        # FEED-IN TARIFF

        # Before 2017 (end of FIT funds), we assume all go for federal FIT
        if sim_year > 2017:

            # Later, therere is only the EWZ feed-in tariff available
            fit_h = fit_high
            fit_l = fit_low

        # Use historical feed-in tariff per system size
        elif sim_year >= 2010:

            # Optimize this later
            if pv_size < 10:
                # hist_fit = 0 since 2013 because gov <10 to investment subsidy
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
        # High prices Mon-Sat 06:00 to 22:00 = 6 d/w * 16 h/d = 96 h/w
        # Low prices Mon-Sat 22:00 to 06:00; and Sunday = 6 * 8 + 24 = 72 h/w
        el_p_l = el_p_ind / ((72/168) + (96/168) * ratio_high_low)
        el_p_h = ratio_high_low * el_p_l
        
        # If this is for a community NPV, assign the electricity when in com
        if sys == "com":

            # Check if the community will market the electricity directly
            if com_tariff == "wholesale":

                el_p_com = self.model.hour_to_average * self.model.wholesale_el_price
            
            else:

                # Define av com electricity price
                el_p_com = el_price[com_tariff]

                # Set high and low individual electricity prices
                el_p_com_l = el_p_com / ((72/168) + (96/168) * ratio_high_low)
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
                
                # Compute the revenues from feeding PV electricity to the grid
                cf_y["FIT"] = ex_pv_h * fit_h + ex_pv_l * fit_l

                # Read avoided consumption from grid (i.e. self-consumption)
                sc_h = np.array(lifetime_load_profile["sc_high"][y])
                sc_l = np.array(lifetime_load_profile["sc_low"][y])

                # Compute the savings from self-consuming solar electricity
                if sys == "ind":

                    # Savings only from avoided consumption from grid
                    cf_y["savings"] = sc_h * el_p_h + sc_l * el_p_l

                elif sys == "com":

                    # Savings from avoided consumption from grid *with old tariff* and from moving to a cheaper electricity tariff because now a single bigger consumer
                    cf_y["savings"] = sc_h * el_p_h + sc_l * el_p_l + np.array(lifetime_load_profile["net_demand_high"][y]) * (el_p_h - el_p_com_h) + np.array(lifetime_load_profile["net_demand_high"][y]) * (el_p_l - el_p_com_l)
                
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
                    # Append dictionary containing the results for this year
                    lifetime_cashflows = lifetime_cashflows.append(cf_y, ignore_index=True)
            
        else:

            # Without degradation, all years have the same profile so we just take the first one and copy the results over the lifetime of the system.

            # Read avoided consumption from grid (i.e. self-consumption)
            sc_h = lifetime_load_profile["sc_high"][0]
            sc_l = lifetime_load_profile["sc_low"][0]

            # Read demand from grid before adoption
            d_h = lifetime_load_profile["demand_high"][0]
            d_l = lifetime_load_profile["demand_low"][0]

            if (sys == "com") and (com_tariff == "wholesale"):

                # Compute gains from direct marketing
                cf_y["FIT"] = np.sum(np.multiply(lifetime_load_profile["excess_solar"][0],el_p_com))

                # Compute savings as the difference between electricity bill
                cf_y["savings"] = d_h * el_p_h + d_l * el_p_l - np.sum(np.multiply(lifetime_load_profile["net_demand"][0],el_p_com))

                # Compute the cost of individual metering
                cf_y["split"] = (sc_h + sc_l) * solar_split_fee

            else:

                # Read year excess solar
                ex_pv_h = lifetime_load_profile["excess_solar_high"][0]
                ex_pv_l = lifetime_load_profile["excess_solar_low"][0]
                
                # Compute revenues from feeding solar electricity to the grid
                cf_y["FIT"] = ex_pv_h * fit_h + ex_pv_l * fit_l

                # Compute the savings from self-consuming solar electricity
                if sys == "ind":

                    # Savings only from avoided consumption from grid
                    cf_y["savings"] = sc_h * el_p_h + sc_l * el_p_l

                elif sys == "com":

                    # Compute net-demands in high and low elec prices periods
                    net_d_h = lifetime_load_profile["net_demand_high"][0]
                    net_d_l = lifetime_load_profile["net_demand_high"][0]

                    # Savings from avoided consumption from grid *with old tariff* and from moving to a cheaper electricity tariff because now a single bigger consumer
                    cf_y["savings"] = sc_h * el_p_h + sc_l * el_p_l + net_d_h * (el_p_h - el_p_com_h) + net_d_l * (el_p_l - el_p_com_l)
                
                    # Compute the cost of individual metering
                    cf_y["split"] = (sc_h + sc_l) * solar_split_fee

            # Compute O&M costs
            cf_y["O&M"] = np.sum(lifetime_load_profile["solar"][0]) * om_cost

            # Compute net cashflows to the agent
            if sys == "ind":
                cf_y["net_cf"] = (cf_y["FIT"] + cf_y["savings"] - cf_y["O&M"])
                cf_y["net_cf_nofit"] = (cf_y["savings"] - cf_y["O&M"])

            elif sys == "com":
                cf_y["net_cf"] = (cf_y["FIT"] + cf_y["savings"] - cf_y["split"]- cf_y["O&M"])
                cf_y["net_cf_nofit"] = (cf_y["savings"] - cf_y["split"] - cf_y["O&M"])

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

    def compute_pv_inv(self, pv_size, pv_sub, pv_price, pv_scale_alpha, pv_scale_beta):
        """
        This function calculates the investment cost of the PV system based on 
        its size, and the price for that size category for the current year.

        Inputs
            pv_size = size of the installation (float)
            pv_sub = investment subsidy for the installation (float)
            pv_price = PV price for ref size (float)
            pv_scale_alpha, pv_scale_beta = parameters of the scale effects in the price of PV systems, following the relation calculated from empirical data in Switzerland: alpha * pv_size ^ beta

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

        # Apply subsidy for installation
        pv_inv = pv_inv - pv_sub
        
        if pv_inv < 0:
            print("PROBLEM WITH SUBSIDIES")

        return pv_inv

    def compute_npv(self, inv, lifetime_cashflows, disc_rate, sim_year):
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

        # Until 2013, federal FIT lasted 25 yrs, 2014-2016 only for 20 years
        if (sim_year > 2013) and (sim_year < 2017):
            
            # Add 20 years of cashflows with FIT
            cf.extend(list(lifetime_cashflows["net_cf"].values)[:20])

            # Add 5 years without FIT
            cf.extend(list(lifetime_cashflows["net_cf_nofit"].values)[-5:])

        else:
            # Add net cashflows for the operational life of the syst to list
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
            npv_com = community system NPV (float)
            pv_sub = investment subsidy for installation (float)
        '''
        # DEFINE ECONOMIC PARAMETERS

        # COMMUNITY SPECIFIC

        # Define the newly installed PV capacity
        pv_size = c_dict["pv_size_added"]

        # Define the number of newly installed smart meters
        n_sm = c_dict["n_sm_added"]

        # Define community's aggregated consumption
        demand = c_dict["demand"]

        # Define the community's electricity tariff
        com_tariff = c_dict["tariff"]

        # If pre-existing PV, then solar is different to account for installations reaching the end of their operational life
            
        # No pre-existing PV
        if c_dict["inv_old"] == 0:
            
            # Define community's generation potential (already AC)
            solar = c_dict["solar"]
            com_prior_PV = False

        else:

            # Define community's generation potential
            solar = c_dict["solar_prior"]
            com_prior_PV = True

        # NON-COMMUNITY SPECIFIC

        # Compute the load profile of the installation throughout its lifetime
        lifetime_load_profile = self.compute_lifetime_load_profile(solar, demand, self.model.pv_lifetime, self.model.deg_rate, self.model.hour_price, com_prior_PV)

        # Compute the cashflows of the installation throughout its lifetime
        lifetime_cashflows = self.compute_lifetime_cashflows(self.el_tariff, pv_size, lifetime_load_profile, self.model.deg_rate,self.model.pv_lifetime, self.model.sim_year, self.model.el_price, self.model.ratio_high_low, self.model.fit_high, self.model.fit_low, self.model.hist_fit, self.model.solar_split_fee,self.model.om_cost, sys="com", com_tariff=com_tariff)

        # Compute the investment subsidy
        pv_sub = self.compute_pv_sub(pv_size, self.model.sim_year, self.model.base_d, self.model.pot_30_d, self.model.pot_100_d, self.model.pot_100_plus_d)
        
        # Compute the investment cost of the PV system
        pv_inv = self.compute_pv_inv(pv_size, pv_sub, self.model.pv_price, self.model.pv_scale_alpha, self.model.pv_scale_beta)
        
        # Compute the investment cost of smart meters
        sm_inv = self.compute_smart_meters_inv(n_sm, self.model.smp_dict)
        
        # Compute the cooperation costs
        coop_cost = 0
        
        # Estimate the total investment cost
        inv_new = pv_inv + sm_inv + coop_cost

        # Compute investment for buying old systems
        inv_old = c_dict["present_inv_ind"] + c_dict["present_inv_com"]

        # Total investment
        inv = inv_new + inv_old

        # Compute the community's NPV
        npv_com = self.compute_npv(inv, lifetime_cashflows, self.model.disc_rate, self.model.sim_year)

        # Compute the community's simple pay-back period
        pp_com = self.compute_simple_pp(inv, lifetime_cashflows, self.model.max_pp)

        return npv_com, pv_sub, inv_new, pp_com
    
    def compute_pv_sub(self, pv_size, sim_year, base_d, pot_30_d, pot_100_d, pot_100_plus_d):
        """
        This method computes the investment subsidy applicable to the installation based on the historical policy in Switzerland.

        Inputs
            pv_size = size of the PV system (float)
            sim_year = year in the simulation (int)
        Returns
            pv_sub = lump-sum subsidy in CHF (float)
        
        More info: since 2013, the Swiss government made available a one-time investment subsidy for new PV installations. The investment subsidy was made available to systems of different sizes at different times, and it also applied to registered installations waiting for the feed-in tariff (which makes it ambiguous to pinpoint the exact time the policy began since installtions prior to the entry into force of the law could apply for it). We simplify by taking one single value per year, and assuming only installations taking place after the entry in force of the legislation can apply to it.

        Summary:
        Since 2010      Federal feed-in tariff available to all installations
        Since 2013      Systems < 10 kWp excluded from FIT
                        Systems 10-30 kWp can choose. We assume all take FIT
        Since 2016      Federal feed-in tariff has no more funds for PV
                        All systems take EWZ FIT, except direct marketing 
                        All systems take investment subsidy
        Since 2018      Large installations >30 kWp can take inv subsidy
        Since 2030      No more investment subsidies

        Sources:

        Subsidies agency Pronovo
        https://pronovo.ch/fr/financement/systeme-de-retribution-de-linjection-sri/retribution/

        RS 730.01
        https://www.admin.ch/opc/fr/classified-compilation/19983391/201404010000/730.01.pdf

        RS 730.03 Annexe 2.1
        https://www.admin.ch/opc/fr/classified-compilation/20162947/index.html#app6ahref3 

        Chapter 4 art 36 
        https://www.admin.ch/opc/fr/classified-compilation/20162947/index.html
        """

        # Initialize subsidy to zero
        pv_sub = 0

        # No investment subsidies before 2013 or after 2030
        if (sim_year > 2013) and (sim_year <= 2030):

            # Small installations
            if (pv_size > 2) and (pv_size < 30):
                
                # Compute investmet subsidy
                try:
                    pv_sub =  base_d[str(sim_year)] + pot_30_d[str(sim_year)] * pv_size
                except:
                    pv_sub = min(base_d.values()) + min(pot_30_d.values()) * pv_size

                # All installations 10-30 kWp opted for FIT until 2016
                if (pv_size > 10) and (sim_year < 2017):
                    pv_sub = 0

            # Medium installations
            elif 30 <= pv_size < 100:

                # Since 2018, >30 kWp can access investment subsidy
                if (sim_year > 2017):

                    try:
                        pv_sub =  base_d[str(sim_year)] + pot_30_d[str(sim_year)] * 30 + pot_100_d[str(sim_year)] * (pv_size - 30)
                    except:
                        pv_sub = min(base_d.values()) + min(pot_30_d.values()) * 30 + min(pot_100_d.values()) * (pv_size - 30)
                else:
                    pv_sub = 0

            # Large installations
            elif (pv_size >= 100) and (pv_size < 50000):
                
                # Since 2018, >100k kWp can access investment subsidy
                if (sim_year > 2017):
                    try:
                        pv_sub =  base_d[str(sim_year)] + pot_30_d[str(sim_year)] * 30 + pot_100_d[str(sim_year)] * 70 + pot_100_plus_d[str(sim_year)] * (pv_size - 100)
                    except:
                        pv_sub = min(base_d.values()) + min(pot_30_d.values()) * 30 + min(pot_100_d.values()) * 70 + min(pot_100_plus_d.values()) * (pv_size - 100)
                else:
                    pv_sub = 0

        return pv_sub