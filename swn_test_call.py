# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 16:31:37 2020

@author: prakh
"""
#%%
import time
start = time.time()
import pandas as pd


distances = pd.read_csv(path + r'07_GIS\DataVisualization_newData\distances_nearest_200bldgs_v1.csv') #all the distances to each building 
path = r'C:\Users\prakh\Dropbox\Com_Paper\\'
agents_info = pd.read_excel(path + r'05_Data\01_CEA_Disaggregated\02_Buildings_Info\Bldgs_Info.xlsx')


from small_world_network import  make_swn
seed = 1
Agents_Peer_Network = make_swn(distances, agents_info, seed) #calls swn function

end = time.time()

durn = end - start
print(durn)