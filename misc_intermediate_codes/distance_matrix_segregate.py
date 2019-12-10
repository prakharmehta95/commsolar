# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 10:33:52 2019

@author: prakh
"""

#%%

import pandas as pd

dist = pd.read_csv(r'C:\Users\iA\Dropbox\Com_Paper\07_GIS\DataVisualization_newData\Distance_Matrix_Nearest200.csv')

list_egids = dist.InputID

mylist = list_egids
mylist = list(dict.fromkeys(mylist))

#%%

df_dist = pd.DataFrame(data = None)
c = 0
for i in mylist:
    c = c + 1
    print(c, "and i = ",i)
    a = dist.loc[dist['InputID'] == i]
    b = list(a.TargetID)
    b_v1 = ['B' + str(i) for i in b]
    d = list(a.Distance)
    try:
        temp_bldgname = 'B' + str(i)
        df_dist[temp_bldgname] = 0
        df_dist[temp_bldgname] = b_v1
        temp_name = 'dist_B' + str(i)
        df_dist[temp_name] = ""
        df_dist[temp_name] = d
    except ValueError:
        temp_bldgname = 'B' + str(i)
        del b_v1[200]
        del d[200]
        df_dist[temp_bldgname] = 0
        df_dist[temp_bldgname] = b_v1
        temp_name = 'dist_B' + str(i)
        df_dist[temp_name] = ""
        df_dist[temp_name] = d
        #continue

df_dist.to_csv(r'C:\Users\iA\Dropbox\Com_Paper\07_GIS\DataVisualization_newData\distances_nearest_200bldgs_v1.csv')        

#%% SWN try

import networkx as nx
import numpy as np
temp_df = pd.DataFrame(data = None, index = range(20),columns = range(4919))

G = nx.watts_strogatz_graph(4919,20,0,2)       
for i in range(200):
    l = list(G.adj[i])
    if len(l) < 21:
        for j in range(21-len(l)):
            l.append(np.nan)
    temp_df[i] = pd.Series(list(G.adj[i]))

#%%
"""
SWN 
may need to use connected watts_strogatz graph if we need all agents to have same number of peers in their network
Also - need to try for preferential attachement - 70% chance of connecting to same type of agent, and 10% to other agent types
- maybe random networks are better
"""

swn = pd.DataFrame(data = None)                         #holds all swns for all agents
temp_df = pd.DataFrame(data = None, index = range(20))
for k in mylist:
    print(k)
    G = nx.watts_strogatz_graph(200,20,0.5,2) #
    temp_df["main"] = pd.Series(list(G.adj[0])) #adj of 0 is taken as 0 represents the node in question, always
    swn_ref = df_dist.loc[:,k]
    di = swn_ref.to_dict()              #dictionary to replace numbers of the watts-stratogatz function with actual building names
    df_x = temp_df.rename(columns = di)
    df_x = df_x.replace({'main':di})
    #swn = pd.DataFrame(data = None)                         #holds all swns for all agents
    swn[k] = ""
    swn[k] = list(df_x.main)
    