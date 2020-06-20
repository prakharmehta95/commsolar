# -*- coding: utf-8 -*-
"""
Current version: June, 2020
@authors: Alejandro Nuñez-Jimenez
"""
#%% IMPORT REQUIRED PACKAGES

import sys, os, glob, json

from os.path import join
from matplotlib import colors

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

#%% IMPORT DATA
files_dir = os.path.dirname(__file__)

# Set directory with data files
data_subfolder = 'code\\COSA_Outputs\\'
data_dir = files_dir[:files_dir.rfind('code')] + data_subfolder

# Create container dictionaries to import simulation results
model_files = {}
com_files = {}
ag_files = {}

# Create a dictionary of file name and container dict
out_dict = {"model_vars_":model_files, "com_formed_":com_files,"agent_vars_":ag_files}

# Loop through container dictionaries
for key, val in out_dict.items():
    for input_file in glob.glob(join(data_dir,'*'+key+'.csv')):

        # Create a label from the name of the experiment that created the data
        label = "_".join(input_file.split("_")[4:7])
        
        # Read data and store it in the container dictionary
        with open(input_file, "r") as mydata:
            val[label] = pd.read_csv(input_file, sep=';')

# List of parameters included in the scenario labels
sim_pars = ["runs", "start_year", "end_year", "n_agents", "n_peers", "n_closest_neighbors", "ZEV", "n_cores"]
cal_pars = ["w_econ","w_swn","w_att","w_subplot","threshold","reduction","awareness_mean","awareness_stdev","awareness_minergie"]
eco_pars = ["diff_prices","min_ratio_sd","fit_high","fit_low","ewz_high_large","ewz_low_large","ewz_high_small","ewz_low_small","ewz_solarsplit_fee","PV_lifetime","PV_degradation","OM_Cost_rate","disc_rate","discount_pp","AC_conv_eff","max_payback_period","demand_price_threshold","n_communities_npv"]

pars_d = {"sim_label": sim_pars, "cal_label":cal_pars, "eco_label":eco_pars}

# Rename second column in analysed files to "variable"
for key, df in out_dict["model_vars_"].items():
    df.rename(columns={"Unnamed: 0":'sim_year'}, inplace=True)

# Rename first column in summaries of calibration results
for dictionary in out_dict.values():
    for key, df in dictionary.items():

        # Create new columns with values of scenario parameters
        for pars_label, pars_list in pars_d.items():
            for ix in range(len(pars_list)):
                df[pars_list[ix]] = df[pars_label].str.split('_').str[ix]

            # Make numerical parameters float
            for par in pars_list:
                try: 
                    df[par] = pd.to_numeric(df[par])
                except:
                    pass

        # Create column to identify experiment
        df["experiment"] = key

# Put all data frames into one
model_df = pd.concat(out_dict["model_vars_"])
communities_df = pd.concat(out_dict["com_formed_"])
agents_df = pd.concat(out_dict["agent_vars_"])

#%% PLOT INSTALLED CAPACITIES
fig_inst, ax_inst = plt.subplots(1,1)

for run in list(set(model_df["run"])):
    
    run_df = model_df.loc[model_df["run"]==run]

    y = run_df["Comm_PV_Installed_CAP"].values
    ax_inst.plot(y, color='blue')

    y = run_df["Ind_PV_Installed_CAP"].values
    ax_inst.plot(y, color='red')

    # Set vertical axis label
    ax_inst.set_ylabel("Installed capacity [kWp]")

    # Set horizontal axis tick labels
    xlabels= np.arange(2017,2036, 2)
    ax_inst.set_xticks(np.arange(0,19, 2))
    ax_inst.set_xticklabels(xlabels)

    # Add legend
    ax_inst.legend(["Community adoption", "Individual adoption"])