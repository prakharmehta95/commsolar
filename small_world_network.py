# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 17:47:48 2019

@author: iA
"""

#%%

def make_swn(distances, agents_info, seed):
    '''
    to make random groups of (NOT small world networks) but random members
    
    distances   = Holds the distances from each agent to its nearest 200 agents (200 agents because dataframe size is reasonable but still satisfies all criteria for the ABM conceptual model)
    agents_info = All agent information 
    seed        = To set the random seed so that every time we get the same peers for each agent 
    '''
    
    #print('swn entered')
    import random 
    import pandas as pd
    
    random.seed(seed)
    list_agents = agents_info.bldg_name
    temp_df = pd.DataFrame(data = None, columns = list_agents)
    for i in list_agents:
        temp_df[i] = random.sample(list(distances[i]), k = 20)
    
    return temp_df

#%%    
    