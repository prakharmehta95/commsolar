# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 16:31:13 2020

@author: prakh
"""
#%%
def make_swn(distances, agents_info, seed):
    '''
    to make random groups of (NOT small world networks) but random members
    
    distances   = Holds the distances from each agent to its nearest 200 agents (200 agents because dataframe size is reasonable but still satisfies all criteria for the ABM conceptual model)
    agents_info = All agent information 
    seed        = To set the random seed so that every time we get the same peers for each agent 
    '''
    
    print('swn entered')
    import random 
    import pandas as pd
    #print(random.choices([1,2,3,4,5,6], k =2))
    random.seed(1)
    list_agents = agents_info.bldg_name
    temp_df = pd.DataFrame(data = None, columns = list_agents)
    for i in list_agents:
        temp_df[i] = random.sample(list(distances[i]), k = 20)
    
    
    return temp_df


#%%
    
"""
other ideas I did not try...
"""
    for i in ['B146906']:#list_agents:
        temp_dist_name          = 'dist_' + i
        temp_df[i]              = ""
        temp_df[i]              = distances[i].copy()
        temp_df[temp_dist_name] = ""
        temp_df[temp_dist_name] = distances[temp_dist_name].copy()
        temp_owner              = agents_info.loc[i]['bldg_owner']
        for j in list(temp_df[i]):
            
        
    
    
    
    
    
# =============================================================================
#   old way using the watts_strogatz function
            #G = nx.watts_strogatz_graph(716,10,0.5,2)       
#     for i in range(716):
#         l = list(G.adj[i])
#         if len(l) < 10:
#             for j in range(10-len(l)):
#                 l.append(np.nan)
#         temp_df[i] = pd.Series(list(G.adj[i]))
#     
# =============================================================================
    
# =============================================================================
#     OLD way, no need for it now
#     swn_ref_Z0003 = pd.read_csv(r'C:\Users\iA\OneDrive - ETHZ\Thesis\PM\Codes\ABM\MasterThesis_PM\masterthesis\TPB\SWN_List_less_100MWh.csv') #key of changing numbers to building names/IDs
#     di = swn_ref_Z0003.Circular_List.to_dict()              #dictionary to replace numbers of the watts-stratogatz function with actual building names
#     temp_df = temp_df.rename(columns = di)
#     swn = pd.DataFrame(data = None)                         #holds all swns for all agents
#     swn = temp_df
# 
#     for i in swn.columns:
#         swn[i] = temp_df[i].map(di)
#     
# =============================================================================
    return swn

#%%
import networkx as nx
import pandas as pd

thisdict = {
  0: "Ford",
  1: "Audi",
  2: "Chevy",
  3: "Lambo",
  4: "Ferrari"
}
#%%
n = 5
k = 2
p = 0.01
swn = nx.generators.random_graphs.watts_strogatz_graph(n,k,p)
ll = [n for n in swn.neighbors(2)]

l = list(swn.adj[2])

temp_df = pd.DataFrame(data = None)
temp_df['swn'] = ""
temp_df['swn'] = l

temp_df = temp_df.rename(columns = thisdict)

swn_name = pd.DataFrame(data = None)
swn_name = temp_df
swn_name = temp_df['swn'].map(thisdict)


#%%

from collections import defaultdict
inputdict = thisdict
reverse = defaultdict(list)
for key, value in inputdict.items():
    reverse[value].append(key)
    
    
for key in reverse[value]:
    inputdict[key] = 999
    
    
    #%%
    
 #  old way using the watts_strogatz function
#G = nx.watts_strogatz_graph(716,10,0.5,2)   
temp_df = pd.DataFrame(data = None)
n = 5
k = 2
p = 0.01
G = nx.generators.random_graphs.watts_strogatz_graph(n,k,p)    
for i in range(n):
    l = list(G.adj[i])
    #if len(l) < 10:
    #    for j in range(10-len(l)):
    #        l.append(np.nan)
    temp_df[i] = pd.Series(list(G.adj[i]))

 #=============================================================================
 #  
 #=============================================================================
  #   OLD way, no need for it now
#swn_ref_Z0003 = pd.read_csv(r'C:\Users\prakh\OneDrive - ETHZ\Thesis\PM\Codes\ABM\MasterThesis_PM\masterthesis\TPB\SWN_List_less_100MWh.csv') #key of changing numbers to building names/IDs

di = thisdict#swn_ref_Z0003.Circular_List.to_dict()              #dictionary to replace numbers of the watts-stratogatz function with actual building names
temp_df = temp_df.rename(columns = di)
swn = pd.DataFrame(data = None)                         #holds all swns for all agents
swn = temp_df
#%% 
for i in swn.columns:
    swn[i] = temp_df[i].map(di)