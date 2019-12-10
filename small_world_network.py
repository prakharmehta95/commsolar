# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 17:47:48 2019

@author: iA
"""

#%%

def make_swn(distances, agents_info):
    '''
    to make random groups of small world networks 
    
    distances   = Holds the distances from each agent to its nearest 200 agents (200 agents because dataframe size is reasonable but still satisfies all criteria for the ABM conceptual model)
    agents_info = All agent information 
    '''
    agents_info = agents_info.set_index('bldg_name', drop = False)
    list_agents = agents_info.bldg_name
    list_agents = data.bldg_name
    temp_df = pd.DataFrame(data = None)
    
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

