# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 11:23:43 2019

@author: iA
"""
#%%
import os
import pandas as pd
os.chdir(r'C:\Users\iA\Desktop\Runs_10-11_Oct')
a = os.listdir()
df = pd.DataFrame(data = None)
df['name'] = ""
df['name'] = a
df.to_csv(r'C:\Users\iA\Desktop\Runs_10-11_Oct\names_11_Oct.csv')
