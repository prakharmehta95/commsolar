# -*- coding: utf-8 -*-
"""
Current version: June, 2020
@authors: Alejandro Nuñez-Jimenez
"""
#%% IMPORT REQUIRED PACKAGES

import sys, os, glob, json
import feather

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

from os.path import join
from matplotlib import colors
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

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
out_dict = {"model_vars_":model_files, "com_formed_":com_files, "agent_vars_":ag_files
}

# Loop through container dictionaries
for key, val in out_dict.items():
    for input_file in glob.glob(join(data_dir,'*'+key+'.feather')):

        # Create a label from the name of the experiment that created the data
        label = "_".join(input_file.split("_")[4:7])
        
        # Read data and store it in the container dictionary
        with open(input_file, "r") as mydata:
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
agents_df = pd.concat(out_dict["agent_vars_"])

# Rename first column in summaries of calibration results
for df in [model_df, communities_df, agents_df]:
#for df in [model_df, communities_df]:

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

vars = ['n_ind', 'inst_cum_ind', 'n_com', 'n_champions', 'inst_cum_com', 'pol_cost_sub_ind', 'pol_cost_sub_com']

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

            """
            Take care of duplicated runs
            """

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
#%% PLOT INSTALLATIONS

# Define vars to plot
plotvars = ["inst_cum_ind", "inst_cum_com"]

color_d = {"inst_cum_ind":"blue", "inst_cum_com":"red"}

labels_d = {"inst_cum_ind":"Individual installations", "inst_cum_com":"Commnuity installations"}

# Create figure
fig_inst, ax_inst = plt.subplots(1,1, figsize=(6.5,4))

# Select data
plot_df = sc_results_analysed

# Define time variable
x = range(len(set(plot_df["sim_year"])))

# Loop through variables
for var in plotvars:

    # Define variable to plot
    cond_var = plot_df["variable"]==var

    # Plot median
    ax_inst.plot(plot_df["p50"].loc[cond_var].values, color=color_d[var], label=labels_d[var])

    # Plot confidence interval
    ax_inst.fill_between(x,plot_df["p05"].loc[cond_var].values,plot_df["p95"].loc[cond_var].values, color=color_d[var], alpha=0.5)

    # Set Y-axis label
    ax_inst.set_ylabel("Cumulative installed capacity [kWp]")

    # Set X-axis labels
    d_yrs = 3
    ax_inst.set_xticks(np.arange(0,len(x),d_yrs))
    ax_inst.set_xticklabels(np.arange(min(model_df["sim_year"]),max(model_df["sim_year"])+1,d_yrs))

# Add legend
ax_inst.legend(loc="upper left")

#%% EXPORT FIGURE
fig_inst.savefig(files_dir +"\\fig_inst.svg", format="svg")
fig_inst.savefig(files_dir +"\\fig_inst.png", format="png", bbox_inches="tight", dpi=210)
#%% PLOT POLICY COSTS

# Define vars to plot
plotvars = ["pol_cost_sub_ind", "pol_cost_sub_com"]

color_d = {"pol_cost_sub_ind":"blue", "pol_cost_sub_com":"red"}

labels_d = {"pol_cost_sub_ind":"Individual installations", "pol_cost_sub_com":"Commnuity installations"}

# Create figure
fig_inst, ax_inst = plt.subplots(1,1, figsize=(6.5,4))

# Select data
plot_df = sc_results_analysed

# Define time variable
x = range(len(set(plot_df["sim_year"])))

# Loop through variables
for var in plotvars:

    # Define variable to plot
    cond_var = plot_df["variable"]==var

    # Plot median
    ax_inst.plot(plot_df["p50"].loc[cond_var].values/1000, color=color_d[var], label=labels_d[var])

    # Plot confidence interval
    ax_inst.fill_between(x,plot_df["p05"].loc[cond_var].values/1000,plot_df["p95"].loc[cond_var].values/1000, color=color_d[var], alpha=0.5)

    # Set Y-axis label
    ax_inst.set_ylabel("Cumulative policy costs [kCHF]")

    # Set X-axis labels
    d_yrs = 3
    ax_inst.set_xticks(np.arange(0,len(x),d_yrs))
    ax_inst.set_xticklabels(np.arange(min(model_df["sim_year"]),max(model_df["sim_year"])+1,d_yrs))

# Add legend
ax_inst.legend(loc="upper left")

#%% EXPORT FIGURE
fig.savefig(files_dir +"\\fig.svg", format="svg")
fig.savefig(files_dir +"\\fig.png", format="png", bbox_inches="tight", dpi=210)

#%% PLOT ENVIRONMENTAL AWARENESS OF ONE RUN

# Create figure and axes
fig_att, ax = plt.subplots(1,1,figsize=(6.5,5))

# Define condition to find the right data to plot
cond = ((agents_df["run"]==0) & (agents_df["sim_year"]==0))

# Plot histogram of environmental attitudes
ax.hist(agents_df["attitude"].loc[cond].values, bins=100, edgecolor="k")

# Set X-axis limits
ax.set_xlim(0,1)

# Set axes labels
ax.set_xlabel("Environmental awareness [-]")
ax.set_ylabel("Absolute frequency [number of agents]")
#%% EXPORT FIGURE
fig_att.savefig(files_dir +"\\fig_att.svg", format="svg")
fig_att.savefig(files_dir +"\\fig_att.png", format="png", bbox_inches="tight", dpi=210)
#%%

var = "peer_effect"

run = 0

fig, ax = plt.subplots(1,1,figsize=(6.5,5), sharex=True)

cm = plt.cm.get_cmap('viridis')

for sim_year in set(agents_df["sim_year"].values):

    #ax = axes[sim_year]

    print(sim_year)

    cond = ((agents_df["run"]==run) & (agents_df["sim_year"]==sim_year))
    
    ax.hist(agents_df[var].loc[cond].values, alpha=0.5, bins=10, label=str(sim_year+2010), color=cm(sim_year/12))
    
    ax.set_xlim(0,1)
    #ax.set_ylim(0,400)
    ax.legend(loc="upper right")

ax.set_xlabel(var)

ax.set_ylabel("Absolute frequency [number of agents]")



