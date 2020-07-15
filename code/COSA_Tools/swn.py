# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 17:47:48 2019

@author: iA
"""
import random 
import pandas as pd

def make_swn(distances, list_agents, n_peers, seed):
    '''
    This function makes 
    to make random groups of (NOT small world networks) but random members
    
    Inputs
        distances = distances from each agent to its nearest 200 agents
            (200 agents because dataframe size is reasonable and
            still satisfies all criteria for the ABM conceptual model)
        list_agents = list of all agents' unique identifiers (list)
        seed = random seed so we get the same peers for each agent (int)
    
    Return
        AgentsNetwork = list of peers for each agent (df)
    '''
    # Determine the random seed
    random.seed(seed)

    # Create an empty dataframe
    AgentsNetwork = pd.DataFrame(data = None, columns = list_agents)

    # Loop through all the agents
    for agent_id in list_agents:

        # Assign the agent a random list of k peers from 200 closests agents
        AgentsNetwork[agent_id] = random.sample(list(distances[agent_id]), 
                                                k = n_peers)
    
    return AgentsNetwork   