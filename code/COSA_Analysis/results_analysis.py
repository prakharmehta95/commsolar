# -*- coding: utf-8 -*-
"""
Current version: June, 2020
@authors: Alejandro Nuñez-Jimenez
"""
#%% IMPORT REQUIRED PACKAGES
import sys, os, glob, json, feather

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
data_subfolder = 'code\\COSA_Outputs\\2_results\\el-change-11-new-cal-fallback-intention\\'
#data_subfolder = 'code\\COSA_Outputs\\'
input_subfolder = 'code\\COSA_Data\\'
data_dir = files_dir[:files_dir.rfind('code')] + data_subfolder
input_dir = files_dir[:files_dir.rfind('code')] + input_subfolder

# Create container dictionaries to import simulation results
model_files = {}
com_files = {}
#ag_files = {}

# Create a dictionary of file name and container dict
out_dict = {"model_vars_":model_files, "com_formed_":com_files,}#"agent_vars_":ag_files}

# Loop through container dictionaries
for key, val in out_dict.items():
    for input_file in glob.glob(join(data_dir,'*'+key+'.feather')):

        # Create a label from the name of the experiment that created the data
        label = "_".join(input_file.split("_")[5:9])
        #label = "_".join(input_file.split("_")[4:6])
        #label = label.replace("_el2yr_"+key+'.feather',"")
        print(label)

        # Read data and store it in the container dictionary
        with open(input_file, "r") as mydata:
            val[label] = feather.read_dataframe(input_file)

# Rename second column in analysed files to "variable"
for key, df in out_dict["model_vars_"].items():
    df.rename(columns={"Unnamed: 0":'sim_year'}, inplace=True)

# Put all data frames into one
model_df = pd.concat(out_dict["model_vars_"])
communities_df = pd.concat(out_dict["com_formed_"])
#agents_df = pd.concat(out_dict["agent_vars_"])

# Create a scenario label
model_df["scenario_label"] = np.char.array(model_df["com_year"].values.astype(str))+ "_" + np.char.array(model_df["direct_market"].values.astype(str)) + "_" + np.char.array(model_df["direct_market_th"].values.astype(str))

# Import agents info
agents_info = pd.read_csv(input_dir+'buildings_info.csv', sep=",")

# Make agent id the index
agents_info = agents_info.set_index("bldg_name")

# Import calibration data
with open(input_dir+"cal_data.json", "r") as cal_file:
    cal_data = json.load(cal_file)

#%% ANALYSE MODEL RESULTS PER VARIABLE

vars = ['n_ind', 'inst_cum_ind', 'n_com', 'inst_cum_com', 'pol_cost_sub_ind', 'pol_cost_sub_com']

n_ticks =  len(set(model_df["sim_year"]))
n_runs = len(set(model_df["run"]))

sc_results = {}
for sc_lab in list(set(model_df["scenario_label"])):
    sc_results[sc_lab] = {}

    for var in vars:
        sc_results[sc_lab][var] = pd.DataFrame(np.zeros((n_ticks, n_runs)),
        index=range(n_ticks), columns=range(n_runs))
        
        for run in range(n_runs):
            cond = (model_df["scenario_label"]==sc_lab) & (model_df["run"]==run)
            
            sc_results[sc_lab][var][run] = model_df[var].loc[cond].values

            sc_results[sc_lab][var][run] = pd.to_numeric(sc_results[sc_lab][var][run])
        
        # Compute percentiles
        p5 = sc_results[sc_lab][var].quantile(q=0.05, axis="columns")
        p50 = sc_results[sc_lab][var].quantile(q=0.5, axis="columns")
        p95 = sc_results[sc_lab][var].quantile(q=0.95, axis="columns")

        # Store percentiles
        sc_results[sc_lab][var]["p05"] = p5
        sc_results[sc_lab][var]["p50"] = p50
        sc_results[sc_lab][var]["p95"] = p95

        # Compute and store 90% confidence interval
        sc_results[sc_lab][var]["CI90"] = sc_results[sc_lab][var]["p95"] - sc_results[sc_lab][var]["p05"]

    sc_results[sc_lab] = pd.concat(sc_results[sc_lab].values(), 
                            keys=sc_results[sc_lab].keys())

    sc_results[sc_lab] = sc_results[sc_lab].reset_index()

    sc_results[sc_lab].rename(columns={"level_0":"variable", "level_1":"sim_year"}, inplace=True)

sc_results_analysed = pd.concat(sc_results.values(), keys=sc_results.keys(), names=["scenario"]).reset_index(level=1, drop=True)

#%% ANALYSE COMMUNITIES

def classify(x):
    """
    Categorises the community by the combination of uses of the members.
    """
    # Check if all buildings have the same use
    if checkequal(x) == True:

        if (x[0] == "MULTI_RES") or (x[0] == "SINGLE_RES"):
            category = "all_residential"
        else:
            category = "all_commercial"
    
    elif checkequal(x) == False:
        category = "mixed_use"

    return category

def checkequal(lst):
    return not lst or lst.count(lst[0]) == len(lst)


# Make the data about agents a dictionary
ag_d = agents_info.to_dict(orient="index")

# Create a column with the ids of the buildings in each community
communities_df["buildings"] = communities_df["community_id"].str.split("_")

# Create column with number of buildings
communities_df["n_members"] = np.array([len(x) for x in communities_df["buildings"].values])

# Create a column with use of each building
communities_df["building_uses"] = np.array([[ag_d[x[i]]["bldg_type"] for i in range(len(x))] for x in communities_df["buildings"].values])

# Create column classifying communities by building use
communities_df["category"] = np.array([classify(x) for x in communities_df["building_uses"].values])

# Identify the block of the community
communities_df["community_block"] = np.array([ag_d[x[0]]["plot_id"] for x in communities_df["buildings"].values])

# Count the number of buildings in the block
communities_df["community_block_n_buildings"] = np.array([np.sum(agents_info["plot_id"].values==x) for x in communities_df["community_block"].values])

# Compute the ratio of buildings in the community per block
communities_df["community_block_ratio_com"] = np.array([communities_df["n_members"].values[i] / communities_df["community_block_n_buildings"].values[i] for i in range(len(communities_df["community_block_n_buildings"].values))])

#%% PLOT NEIGHBOR INFLUENCE EFFECT

plt.scatter(communities_df["community_block_n_buildings"], communities_df["community_block_ratio_com"], alpha=0.5)

#%% PLOT SCR PER COMMUNITY CATEGORY

fig, ax = plt.subplots(1,1)

for cat in ["all_residential", "all_commercial", "mixed_use"]:

    ax.hist(communities_df["SC"].loc[(communities_df["category"].values == cat)]/communities_df["demand"].loc[(communities_df["category"].values == cat)], alpha=0.25, bins=np.arange(0,1.05, 0.025), label=cat, density=True)

    ax.set_xlabel("Self-sufficiency ratio [% of demand met by solar generation]")
    ax.set_ylabel("Number of communities [-]")

    ax.

    ax.legend()

#%% PLOT INSTALLATIONS

# Define vars to plot
plotvars = ["inst_cum_ind", "inst_cum_com"]

color_d = {"inst_cum_ind":"blue", "inst_cum_com":"red"}

labels_d = {"inst_cum_ind":"Individual installations", "inst_cum_com":"Community installations"}

for scenario in set(sc_results_analysed.index):

    # Create figure
    fig_inst, ax_inst = plt.subplots(1,1, figsize=(6.5,4))

    # Select data
    plot_df = sc_results_analysed.loc[scenario]

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

        #ax_inst.plot(plot_df["p50"].loc[plot_df["variable"]=="inst_cum_ind"].values+plot_df["p50"].loc[plot_df["variable"]=="inst_cum_com"].values, color="k")

    ax_inst.set_ylim(0,20000)

    # Add legend
    ax_inst.legend(loc="upper left")

    #ax_inst.plot(cal_data['inst_cum_ZH_wiedikon_cal'], color="k")

    # Add scenario title
    fig_inst.suptitle(scenario)

#%% COMPARE TOTAL INSTALLATIONS
# Define vars to plot
plotvars = ["inst_cum_ind", "inst_cum_com"]

color_d = {"inst_cum_ind":"blue", "inst_cum_com":"red"}

# Create figure
fig_inst, ax_inst = plt.subplots(1,1, figsize=(6.5,4))

for scenario in set(sc_results_analysed.index):

    # Select data
    plot_df = sc_results_analysed.loc[scenario]

    # Define time variable
    x = range(len(set(plot_df["sim_year"])))

    # Plot confidence interval
    ax_inst.fill_between(x,plot_df["p05"].loc[plot_df["variable"]=="inst_cum_ind"].values+plot_df["p05"].loc[plot_df["variable"]=="inst_cum_com"].values,plot_df["p95"].loc[plot_df["variable"]=="inst_cum_ind"].values+plot_df["p05"].loc[plot_df["variable"]=="inst_cum_com"].values, alpha=0.5)

    # Plot median
    ax_inst.plot(plot_df["p50"].loc[plot_df["variable"]=="inst_cum_ind"].values+plot_df["p50"].loc[plot_df["variable"]=="inst_cum_com"].values, label=scenario)

# Set Y-axis label
ax_inst.set_ylabel("Cumulative installed capacity [kWp]")

# Set X-axis labels
d_yrs = 3
ax_inst.set_xticks(np.arange(0,len(x),d_yrs))
ax_inst.set_xticklabels(np.arange(min(model_df["sim_year"]),max(model_df["sim_year"])+1,d_yrs))

ax_inst.set_ylim(0,20000)

# Add legend
ax_inst.legend(loc="upper left")

#%% COMPARE TOTAL COST

# Create figure
fig_inst, ax_inst = plt.subplots(1,1, figsize=(6.5,4))

for scenario in set(sc_results_analysed.index):

    # Select data
    plot_df = sc_results_analysed.loc[scenario]

    # Define time variable
    x = range(len(set(plot_df["sim_year"])))

    # Plot confidence interval
    ax_inst.fill_between(x,plot_df["p05"].loc[plot_df["variable"]=="pol_cost_sub_ind"].values+plot_df["p05"].loc[plot_df["variable"]=="pol_cost_sub_com"].values,plot_df["p95"].loc[plot_df["variable"]=="pol_cost_sub_ind"].values+plot_df["p05"].loc[plot_df["variable"]=="pol_cost_sub_com"].values, alpha=0.5)

    # Plot median
    ax_inst.plot(plot_df["p50"].loc[plot_df["variable"]=="pol_cost_sub_ind"].values + plot_df["p50"].loc[plot_df["variable"]=="pol_cost_sub_com"].values, label=scenario)

# Set Y-axis label
ax_inst.set_ylabel("Cumulative installed capacity [kWp]")

# Set X-axis labels
d_yrs = 3
ax_inst.set_xticks(np.arange(0,len(x),d_yrs))
ax_inst.set_xticklabels(np.arange(min(model_df["sim_year"]),max(model_df["sim_year"])+1,d_yrs))

#ax_inst.set_ylim(0,20000)

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

for scenario in set(sc_results_analysed.index):

    # Create figure
    fig_inst, ax_inst = plt.subplots(1,1, figsize=(6.5,4))

    # Select data
    plot_df = sc_results_analysed.loc[scenario]

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

        ax_inst.plot(plot_df["p50"].loc[plot_df["variable"]=="pol_cost_sub_ind"].values/1000+plot_df["p50"].loc[plot_df["variable"]=="pol_cost_sub_com"].values/1000, color="k")

    ax_inst.set_ylim(0,4500)

    # Add legend
    ax_inst.legend(loc="upper left")

#%% EXPORT FIGURE
fig_inst.savefig(files_dir +"\\fig.svg", format="svg")
fig_inst.savefig(files_dir +"\\fig.png", format="png", bbox_inches="tight", dpi=210)

#%% PLOT BUBBLE FIGURE

fig_bub, ax_bub = plt.subplots(1,1)

for scenario in set(sc_results_analysed.index):

    sc_df = sc_results_analysed.loc[scenario]

    plot_df = sc_df.loc[(sc_df["sim_year"]==max(set(sc_df["sim_year"])))]

    tot_cost = plot_df.loc[plot_df["variable"]=="pol_cost_sub_com"].iloc[:,2:51].values + plot_df.loc[plot_df["variable"]=="pol_cost_sub_ind"].iloc[:,2:51].values

    tot_inst = plot_df.loc[plot_df["variable"]=="inst_cum_ind"].iloc[:,2:51].values + plot_df.loc[plot_df["variable"]=="inst_cum_com"].iloc[:,2:51].values

    ax_bub.scatter(tot_cost/1000000,tot_inst/1000, alpha=0.5, label=scenario)

    ax_bub.set_xlabel("Total policy cost [million CHF]")
    ax_bub.set_ylabel("Cumulative installed capacity [MWp]")

    ax_bub.legend()

#%% PLOT BUBBLE GRAPH WITH HISTOGRAMS

color_d = {"2050_False_100000":"red", "2019_False_100000":"orange", "2019_True_100000":"cyan", "2019_True_10000":"purple"}

def scatter_hist(x, y, ax, ax_histx, ax_histy, label, color_d):

    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # the scatter plot:
    bubbles = ax.scatter(x, y, alpha=0.5, label=label, color=color_d[label])

    # plot the median values in scatter plot:
    ax.scatter(np.median(x[0]), np.median(y[0]), color=None, edgecolor="k", zorder=100)

    # annotate median
    if label == "2019_True_100000":
        text = "Com. since 2019 \nDM > 100 MWh/yr"
        ax.annotate(text, xy=(np.median(x[0]), np.median(y[0])), xytext=(np.median(x[0])*0.5, np.median(y[0])*3), arrowprops=dict(arrowstyle="->"))
    else:
        ax.annotate(label, xy=(np.median(x[0]), np.median(y[0])), xytext=(np.median(x[0])*1.5, np.median(y[0])*0.5), arrowprops=dict(arrowstyle="->"))

    # Set axes limits
    ax.set_xlim(0,6.4)
    ax.set_ylim(0,34)

    # Set axes labels
    ax.set_xlabel("Total policy cost [million CHF]")
    ax.set_ylabel("Cumulative installed capacity [MWp]")

    # Add legend
    ax.legend()

    # the histograms
    ax_histx.hist(x[0], bins=np.arange(0,6.5,0.5), histtype="step", color=color_d[label])
    ax_histy.hist(y[0], bins=np.arange(0,36,2), orientation='horizontal', histtype="step", color=color_d[label])

    # set histograms limits
    ax_histx.set_ylim(0,20)
    ax_histy.set_xlim(0,20)

    # set histograms labels
    ax_histx.set_ylabel("Frequency [%]")
    ax_histy.set_xlabel("Frequency [%]")

    histx_ylabels = ax_histx.get_yticks().tolist()
    ax_histx.set_yticklabels(['{:,.0%}'.format(x/50) for x in histx_ylabels])

    histy_xlabels = ax_histy.get_xticks().tolist()
    ax_histy.set_xticklabels(['{:,.0%}'.format(x/50) for x in histy_xlabels])

# start with a square Figure
fig = plt.figure(figsize=(8, 8))

# Add a gridspec with two rows and two columns and a ratio of 2 to 7 between the size of the marginal axes and the main axes in both directions.
# Also adjust the subplot parameters for a square plot.
gs = fig.add_gridspec(2, 2,  width_ratios=(7, 2), height_ratios=(2, 7),
                    left=0.1, right=0.9, bottom=0.1, top=0.9,
                    wspace=0.05, hspace=0.05)

ax = fig.add_subplot(gs[1, 0])
ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)

for scenario in set(sc_results_analysed.index):

    sc_df = sc_results_analysed.loc[scenario]

    plot_df = sc_df.loc[(sc_df["sim_year"]==max(set(sc_df["sim_year"])))]

    tot_cost = plot_df.loc[plot_df["variable"]=="pol_cost_sub_com"].iloc[:,2:51].values + plot_df.loc[plot_df["variable"]=="pol_cost_sub_ind"].iloc[:,2:51].values

    tot_inst = plot_df.loc[plot_df["variable"]=="inst_cum_ind"].iloc[:,2:51].values + plot_df.loc[plot_df["variable"]=="inst_cum_com"].iloc[:,2:51].values

    x = tot_cost/1000000
    y = tot_inst/1000

    # use the previously defined function
    scatter_hist(x, y, ax, ax_histx, ax_histy, scenario, color_d)

plt.show()

#%%

fig, ax = plt.subplots(1,1)

for lab in ["com_2019_no_dm", "com_2019_dm_100"]:

    plot_df = model_df.loc[lab]

    ax.hist(plot_df["inst_cum_com"].loc[plot_df["sim_year"]==2035], bins=25, histtype="step", label=lab)

    av = np.average(plot_df["inst_cum_com"].loc[plot_df["sim_year"]==2035])

    ax.axvline(x=av)
    print(lab, av)

ax.legend()
#%%

scs = ["com_2019_dm_100", "com_2019_no_dm"]

fig, ax = plt.subplots(1,1)

for sc in scs:

    sc_df = communities_df.loc[sc]

    coms_above = []
    power_above = []

    for run in set(sc_df.run):

        data = sc_df["demand"].loc[sc_df["run"]==run].values

        coms_above.append(len(data[np.where(data>100000)]))

        data_power = sc_df["pv_size"].loc[sc_df["run"]==run].values

        power_above.append(np.sum(data_power[np.where(data>100000)]))
    
    ax.hist(power_above,bins=np.arange(0,25000,1000), alpha=0.5, label=sc)
    ax.axvline(x=np.average(power_above))
    ax.legend()

#%%

fig, ax = plt.subplots(1,1)

for sc in ["2050_False_100000", "2019_False_100000", "2019_True_100000"]:

    data_df = sc_results_analysed.loc[sc]

    y = data_df["p50"].loc[data_df["variable"]=="n_ind"].values + data_df["p50"].loc[data_df["variable"]=="n_com"].values
    
    ax.plot(y)




#%%

# #%% PLOT ENVIRONMENTAL AWARENESS OF ONE RUN

# # Create figure and axes
# fig_att, ax = plt.subplots(1,1,figsize=(6.5,5))

# # Define condition to find the right data to plot
# cond = ((agents_df["run"]==0) & (agents_df["sim_year"]==0))

# # Plot histogram of environmental attitudes
# ax.hist(agents_df["attitude"].loc[cond].values, bins=100, edgecolor="k")

# # Set X-axis limits
# ax.set_xlim(0,1)

# # Set axes labels
# ax.set_xlabel("Environmental awareness [-]")
# ax.set_ylabel("Absolute frequency [number of agents]")
# #%% EXPORT FIGURE
# fig_att.savefig(files_dir +"\\fig_att.svg", format="svg")
# fig_att.savefig(files_dir +"\\fig_att.png", format="png", bbox_inches="tight", dpi=210)
# #%%

# var = "peer_effect"

# run = 0

# fig, ax = plt.subplots(1,1,figsize=(6.5,5), sharex=True)

# cm = plt.cm.get_cmap('viridis')

# for sim_year in set(agents_df["sim_year"].values):

#     #ax = axes[sim_year]

#     print(sim_year)

#     cond = ((agents_df["run"]==run) & (agents_df["sim_year"]==sim_year))
    
#     ax.hist(agents_df[var].loc[cond].values, alpha=0.5, bins=10, label=str(sim_year+2010), color=cm(sim_year/12))
    
#     ax.set_xlim(0,1)
#     #ax.set_ylim(0,400)
#     ax.legend(loc="upper right")

# ax.set_xlabel(var)

# ax.set_ylabel("Absolute frequency [number of agents]")