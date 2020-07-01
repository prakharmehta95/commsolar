# -*- coding: utf-8 -*-
"""
Current version: June, 2020
@authors: Alejandro Nuñez-Jimenez
"""
#%% IMPORT REQUIRED PACKAGES

import sys, os, glob, json
import feather

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
input_subfolder = 'code\\COSA_Data\\'
data_dir = files_dir[:files_dir.rfind('code')] + data_subfolder
input_dir = files_dir[:files_dir.rfind('code')] + input_subfolder

# Create container dictionaries to import simulation results
model_files = {}
com_files = {}
ag_files = {}

# Create a dictionary of file name and container dict
out_dict = {"model_vars_":model_files, "com_formed_":com_files,#"agent_vars_":ag_files
}

# Loop through container dictionaries
for key, val in out_dict.items():
    for input_file in glob.glob(join(data_dir,'*'+key+'.feather')):

        # Create a label from the name of the experiment that created the data
        label = "_".join(input_file.split("_")[4:7])
        
        # Read data and store it in the container dictionary
        with open(input_file, "r") as mydata:
            # val[label] = pd.read_csv(input_file, sep=';')
            val[label] = feather.read_dataframe(input_file)

# List of parameters included in the scenario labels
cal_pars = ["w_econ","w_swn","w_att","w_subplot","threshold","reduction","awareness_mean","awareness_stdev","awareness_minergie"]

pars_d = {"cal_label":cal_pars}

# Rename second column in analysed files to "variable"
for key, df in out_dict["model_vars_"].items():
    df.rename(columns={"Unnamed: 0":'sim_year'}, inplace=True)

# Put all data frames into one
model_df = pd.concat(out_dict["model_vars_"])
communities_df = pd.concat(out_dict["com_formed_"])
#agents_df = pd.concat(out_dict["agent_vars_"])

# Rename first column in summaries of calibration results
#for df in [model_df, communities_df, agents_df]:
for df in [model_df, communities_df]:

    # Create new columns with values of scenario parameters
    for ix in range(len(cal_pars)):

        df[cal_pars[ix]] = df["cal_label"].str.split('_').str[ix]

        # Make numerical parameters float
        df[cal_pars[ix]] = pd.to_numeric(df[cal_pars[ix]])

# Import agents info
agents_info = pd.read_csv(input_dir+'buildings_info.csv', sep=",")

# Make agent id the index
agents_info = agents_info.set_index("bldg_name")

# Import calibration data
with open(input_dir+"cal_data.json", "r") as cal_file:
    cal_data = json.load(cal_file)

#%% ANALYSE MODEL RESULTS PER VARIABLE

vars = ['n_ind', 'inst_cum_ind', 'n_com', 'n_champions', 'inst_cum_com']

n_ticks =  len(set(model_df["sim_year"]))
n_runs = len(set(model_df["run"]))

sc_results = {}
for cal_lab in list(set(model_df["cal_label"])):
    sc_results[cal_lab] = {}

    for var in vars:
        sc_results[cal_lab][var] = pd.DataFrame(np.zeros((n_ticks, n_runs)),
        index=range(n_ticks), columns=range(n_runs))
        
        for run in range(n_runs):
            cond = (model_df["cal_label"]==cal_lab) & (model_df["run"]==run)
            
            sc_results[cal_lab][var][run] = model_df[var].loc[cond].values

            sc_results[cal_lab][var][run] = pd.to_numeric(sc_results[cal_lab][var][run])
        
        # Compute percentiles
        p5 = sc_results[cal_lab][var].quantile(q=0.05, axis="columns")
        p50 = sc_results[cal_lab][var].quantile(q=0.5, axis="columns")
        p95 = sc_results[cal_lab][var].quantile(q=0.95, axis="columns")

        # Store percentiles
        sc_results[cal_lab][var]["p05"] = p5
        sc_results[cal_lab][var]["p50"] = p50
        sc_results[cal_lab][var]["p95"] = p95

        # Compute and store 90% confidence interval
        sc_results[cal_lab][var]["CI90"] = sc_results[cal_lab][var]["p95"] - sc_results[cal_lab][var]["p05"]

    sc_results[cal_lab] = pd.concat(sc_results[cal_lab].values(), 
                            keys=sc_results[cal_lab].keys())

    sc_results[cal_lab] = sc_results[cal_lab].reset_index()

    sc_results[cal_lab].rename(columns={"level_0":"variable", "level_1":"sim_year"}, inplace=True)

sc_results_analysed = pd.concat(sc_results.values(), keys=sc_results.keys(), names=["scenario"]).reset_index(level=1, drop=True)

#%% PLOT INSTALLED CAPACITIES

color_d = {"inst_cum_ind":"blue", "inst_cum_com":"red"}

fig_inst, ax_inst = plt.subplots(1,1, figsize=(6.5,4))

for sc in set(sc_results_analysed.index):

    # Select data
    plot_df = sc_results_analysed.loc[sc,:]

    x = range(len(set(plot_df["sim_year"])))

    print(sc)

    for var in color_d.keys():

        cond_var = plot_df["variable"]==var

        ax_inst.fill_between(x, plot_df["p05"].loc[cond_var].values / 1000, plot_df["p95"].loc[cond_var].values / 1000, alpha=0.25, color=color_d[var])

        for run in set(model_df["run"]):
            ax_inst.plot(plot_df[run].loc[cond_var].values, color=color_d[var], alpha=0.5)

        ax_inst.plot(plot_df["p50"].loc[cond_var].values / 1000, color=color_d[var])

        print(plot_df["p50"].loc[cond_var].values[0])

ax_inst.plot(cal_data['inst_cum_ZH_wiedikon_cal'], color="k", linestyle="--")

ax_inst.set_ylabel("Installed capacity [kWp]")
ax_inst.set_xticks(np.arange(0,len(x),2))
ax_inst.set_xticklabels(np.arange(min(model_df["sim_year"]),max(model_df["sim_year"])+1,2))

#%% PLOT NUMBER OF INSTALLATIONS

color_d = {"n_ind":"blue", "n_com":"red"}

fig_inst, ax_inst = plt.subplots(1,1, figsize=(6.5,4))

for sc in set(sc_results_analysed.index):

    # Select data
    plot_df = sc_results_analysed.loc[sc,:]

    x = range(len(set(plot_df["sim_year"])))

    print(sc)

    for var in color_d.keys():

        cond_var = plot_df["variable"]==var

        ax_inst.fill_between(x, plot_df["p05"].loc[cond_var].values / 1000, plot_df["p95"].loc[cond_var].values / 1000, alpha=0.25, color=color_d[var])

        for run in set(model_df["run"]):
            ax_inst.plot(plot_df[run].loc[cond_var].values, color=color_d[var], alpha=0.5)

        ax_inst.plot(plot_df["p50"].loc[cond_var].values / 1000, color=color_d[var])

        print(plot_df["p50"].loc[cond_var].values[0])

ax_inst.set_ylabel("Installed capacity [kWp]")
ax_inst.set_xticks(np.arange(0,len(x),2))
ax_inst.set_xticklabels(np.arange(min(model_df["sim_year"]),max(model_df["sim_year"])+1,2))