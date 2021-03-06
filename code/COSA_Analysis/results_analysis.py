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

#%% SUMMARY OF RESULTS

labels_d = {"2050_False_100000":"IND", "2019_False_100000":"COM", "2019_True_100000":"ZEV", "2019_True_1":"ZEV+"}

variables = ["inst_cum_com", "inst_cum_ind", "n_com", "n_ind", "pol_cost_sub_com", "pol_cost_sub_ind"]

output = {}

for sc in labels_d.keys():

    data_sc = sc_results_analysed.loc[sc]

    output[sc] = {}

    for var in variables:

        data_v = data_sc.loc[data_sc["variable"]==var]

        output[sc][var+"_median"] = data_v["p50"].loc[data_v["sim_year"]==25].values[0]
        output[sc][var+"_p5"] = data_v["p05"].loc[data_v["sim_year"]==25].values[0]
        output[sc][var+"_p95"] = data_v["p95"].loc[data_v["sim_year"]==25].values[0]

        if (var == "pol_cost_sub_com") or (var == "pol_cost_sub_ind"):

            data = np.sum(data_v.iloc[:,2:52].values,axis=0)
            output[sc][var+"_median"] = np.median(data)
            output[sc][var+"_p5"] = np.percentile(data, q=5)
            output[sc][var+"_p95"] = np.percentile(data, q=95)

output_df = pd.DataFrame(output)

output_df.to_csv(files_dir +"\\results_table.csv", sep=";")

#%% PLOT BUBBLE GRAPH WITH HISTOGRAMS
# Color dictionary
color_d = {'dark_blue':(0.,114/255,178/255), 
'flx_monthly':(213/255,94/255,0.),
'pid_monthly':(204/255,121/255,167/255), 
'fix_trigger':(86/255,180/255,233/255),
'flx_trigger':(230/255,159/255,0.),
'green':(0.,158/255,115/255),}

color_d = {"2050_False_100000":(213/255,94/255,0.), "2019_False_100000":(230/255,159/255,0.), "2019_True_100000":(86/255,180/255,233/255), "2019_True_1":(0.,114/255,178/255)}

labels_d = {"2050_False_100000":"IND", "2019_False_100000":"COM", "2019_True_100000":"ZEV", "2019_True_1":"ZEV+"}

def scatter_hist(x, y, ax, ax_histx, ax_histy, sc, color_d):

    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # the scatter plot:
    bubbles = ax.scatter(x,y,alpha=0.5,label=labels_d[sc],color=color_d[sc])

    # plot the median values in scatter plot:
    ax.scatter(np.median(x[0]), np.median(y[0]), color=color_d[sc],alpha=0.5, edgecolor="k", zorder=100)

    # annotate median
    if sc == "2019_True_100000":
        # ZEV
        ax.annotate(labels_d[sc]+" median", xy=(np.median(x[0]), np.median(y[0])), xytext=(np.median(x[0])*0.2, np.median(y[0])*2.25), arrowprops=dict(arrowstyle="->"))
    elif sc == "2050_False_100000":
        # IND
        ax.annotate(labels_d[sc]+" median", xy=(np.median(x[0]), np.median(y[0])), xytext=(np.median(x[0])*1.7, np.median(y[0])*0.6), arrowprops=dict(arrowstyle="->"))
    elif sc == "2019_False_100000":
        # COM
        ax.annotate(labels_d[sc]+" median", xy=(np.median(x[0]), np.median(y[0])), xytext=(np.median(x[0])*1, np.median(y[0])*0.3), arrowprops=dict(arrowstyle="->"))
    elif sc == "2019_True_1":
        # ZEV+
        ax.annotate(labels_d[sc]+" median", xy=(np.median(x[0]), np.median(y[0])), xytext=(np.median(x[0])*1, np.median(y[0])*3), arrowprops=dict(arrowstyle="->"))

    # Set axes limits
    ax.set_xlim(0,10.5)
    ax.set_ylim(0,54)

    # Set axes labels
    ax.set_xlabel("Total policy cost [million CHF]")
    ax.set_ylabel("Cumulative installed capacity [MWp]")

    # Add legend
    ax.legend(loc="lower right")

    # the histograms
    ax_histx.hist(x[0], bins=np.arange(0,12,0.5), histtype="step", color=color_d[sc])
    ax_histy.hist(y[0], bins=np.arange(0,54,2), orientation='horizontal', histtype="step", color=color_d[sc])

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
fig_bubhist = plt.figure(figsize=(8, 8))

# Add a gridspec with two rows and two columns and a ratio of 2 to 7 between the size of the marginal axes and the main axes in both directions.
# Also adjust the subplot parameters for a square plot.
gs = fig_bubhist.add_gridspec(2, 2,  width_ratios=(7, 2), height_ratios=(2, 7),
                    left=0.1, right=0.9, bottom=0.1, top=0.9,
                    wspace=0.05, hspace=0.05)

ax = fig_bubhist.add_subplot(gs[1, 0])
ax_histx = fig_bubhist.add_subplot(gs[0, 0], sharex=ax)
ax_histy = fig_bubhist.add_subplot(gs[1, 1], sharey=ax)

for scenario in ["2050_False_100000", "2019_False_100000", "2019_True_100000", "2019_True_1"]:

    sc_df = sc_results_analysed.loc[scenario]

    plot_df = sc_df.loc[(sc_df["sim_year"]==max(set(sc_df["sim_year"])))]

    tot_cost = plot_df.loc[plot_df["variable"]=="pol_cost_sub_com"].iloc[:,2:51].values + plot_df.loc[plot_df["variable"]=="pol_cost_sub_ind"].iloc[:,2:51].values

    tot_inst = plot_df.loc[plot_df["variable"]=="inst_cum_ind"].iloc[:,2:51].values + plot_df.loc[plot_df["variable"]=="inst_cum_com"].iloc[:,2:51].values

    x = tot_cost/1000000
    y = tot_inst/1000

    # use the previously defined function
    scatter_hist(x, y, ax, ax_histx, ax_histy, scenario, color_d)

plt.show()

#%% EXPORT FIGURE
fig_bubhist.savefig(files_dir +"\\fig_bubhist.svg", format="svg")
fig_bubhist.savefig(files_dir +"\\fig_bubhist.png", format="png", bbox_inches="tight", dpi=210)
#%% PLOT INSTALLATIONS

color_d = {"2050_False_100000":(213/255,94/255,0.), "2019_False_100000":(86/255,180/255,233/255), "2019_True_100000":(230/255,159/255,0.), "2019_True_10000":"purple"}

labels_d = {"2050_False_100000":"IND", "2019_False_100000":"COM", "2019_True_100000":"ZEV", "2019_True_1":"ZEV+"}

# Define vars to plot
plotvars = ["inst_cum_ind", "inst_cum_com"]

color_d = {"inst_cum_ind":"blue", "inst_cum_com":"red"}

# Create figure
fig_inst, axes_inst = plt.subplots(2,2, figsize=(6.5,6.5), sharex=True, sharey=True)

for scenario in set(sc_results_analysed.index):

    # Select axes
    if scenario == "2050_False_100000":
        ax_inst = axes_inst[0,0]
        
        # Set Y-axis label
        ax_inst.set_ylabel("Cumulative installed capacity [MWp]")

    elif scenario == "2019_False_100000":
        ax_inst = axes_inst[0,1]
    elif scenario == "2019_True_100000":
        ax_inst = axes_inst[1,0]

        # Set Y-axis label
        ax_inst.set_ylabel("Cumulative installed capacity [MWp]")

    elif scenario == "2019_True_1":
        ax_inst = axes_inst[1,1]

    # Select data
    plot_df = sc_results_analysed.loc[scenario]

    # Define time variable
    x = range(len(set(plot_df["sim_year"])))

    # Loop through variables
    for var in plotvars:

        # Define variable to plot
        cond_var = plot_df["variable"]==var

        # Plot median
        ax_inst.plot(plot_df["p50"].loc[cond_var].values/1000, color=color_d[var], label=var)

        # Plot confidence interval
        ax_inst.fill_between(x,plot_df["p05"].loc[cond_var].values/1000,plot_df["p95"].loc[cond_var].values/1000, color=color_d[var], alpha=0.5)

        # Set X-axis labels
        d_yrs = 5
        ax_inst.set_xticks(np.arange(0,len(x),d_yrs))
        ax_inst.set_xticklabels(np.arange(min(model_df["sim_year"]),max(model_df["sim_year"])+1,d_yrs))

    ax_inst.set_ylim(0,20)
    ax_inst.set_xlim(0,25)

    #ax_inst.plot(cal_data['inst_cum_ZH_wiedikon_cal'], color="k")

    # Add scenario title
    ax_inst.set_title(labels_d[scenario])

# Add legend
axes_inst[1,0].legend(["Individual adoption", "Community systems"], loc="center", bbox_to_anchor=(1.1, -0.2), ncol=2)

#%% EXPORT FIGURE
fig_inst.savefig(files_dir +"\\fig_inst.svg", format="svg")
fig_inst.savefig(files_dir +"\\fig_inst.png", format="png", bbox_inches="tight", dpi=210)

#%% PLOT NUMBER OF ADOPTERS

# Create figure
fig_adopt, axes_adopt = plt.subplots(2,2, figsize=(6.5,6.5), sharex=True, sharey=True)

for scenario in set(sc_results_analysed.index):
    
    # Select axes
    if scenario == "2050_False_100000":
        ax_adopt = axes_adopt[0,0]
        
        # Set Y-axis label
        ax_adopt.set_ylabel("Number of PV adopters [-]")

    elif scenario == "2019_False_100000":
        ax_adopt = axes_adopt[0,1]
    elif scenario == "2019_True_100000":
        ax_adopt = axes_adopt[1,0]

        # Set Y-axis label
        ax_adopt.set_ylabel("Number of PV adopters [-]")

    elif scenario == "2019_True_1":
        ax_adopt = axes_adopt[1,1]

    # Select data
    plot_df = sc_results_analysed.loc[scenario]

    # Define time variable
    x = range(len(set(plot_df["sim_year"])))

    n_pv = plot_df["p50"].loc[plot_df["variable"]=="n_ind"] + plot_df["p50"].loc[plot_df["variable"]=="n_com"]

    n_pv_05 = plot_df["p05"].loc[plot_df["variable"]=="n_ind"] + plot_df["p05"].loc[plot_df["variable"]=="n_com"]

    n_pv_95 = plot_df["p95"].loc[plot_df["variable"]=="n_ind"] + plot_df["p95"].loc[plot_df["variable"]=="n_com"]

    ax_adopt.plot(x, n_pv)
    ax_adopt.fill_between(x, n_pv_05, n_pv_95, alpha=0.5)

    ax_adopt.set_title(labels_d[scenario])

    ax_adopt.set_xlim(0,25)
    ax_adopt.set_ylim(0,200)

    # Set X-axis labels
    d_yrs = 5
    ax_adopt.set_xticks(np.arange(0,len(x),d_yrs))
    ax_adopt.set_xticklabels(np.arange(min(model_df["sim_year"]),max(model_df["sim_year"])+1,d_yrs))

#%% EXPORT FIGURE
fig_adopt.savefig(files_dir +"\\fig_adopters.svg", format="svg")
fig_adopt.savefig(files_dir +"\\fig_adopters.png", format="png", bbox_inches="tight", dpi=210)

#%% PLOT NEIGHBOR INFLUENCE EFFECT

labels_d = {"com_2019_no_dm":"COM", "com_2019_dm_100":"ZEV", "com_2019_dm_1":"ZEV+"}

fig_ni, axes_ni = plt.subplots(3,3, figsize=(10,12), sharey=True, sharex=True)

# Remove space between subplots
plt.subplots_adjust(wspace=0, hspace=0)

scs = ["com_2019_no_dm", "com_2019_dm_100", "com_2019_dm_1"]

cats = ["all_residential", "all_commercial", "mixed_use"]

colorcat_d = {"all_residential":"blue", "all_commercial":"darkorange", "mixed_use":"green"}

for sc in scs:

    data = communities_df.loc[sc]

    for cat in cats:

        ax_ni = axes_ni[scs.index(sc), cats.index(cat)]

        cat_median = []
        cat_p5 = []
        cat_p95 = []
        cat_sizes = set(data["community_block_n_buildings"].loc[(data["category"].values == cat)])

        for block_s in cat_sizes:

            cat_median.append(np.median(data["community_block_ratio_com"].loc[(data["category"].values == cat) & (data["community_block_n_buildings"].values == block_s)]))

            cat_p5.append(np.percentile(data["community_block_ratio_com"].loc[(data["category"].values == cat) & (data["community_block_n_buildings"].values == block_s)], q=5))

            cat_p95.append(np.percentile(data["community_block_ratio_com"].loc[(data["category"].values == cat) & (data["community_block_n_buildings"].values == block_s)], q=95))

        ax_ni.plot(sorted(list(cat_sizes)), cat_median, label=cat, color=colorcat_d[cat])

        ax_ni.fill_between(sorted(list(cat_sizes)), cat_p5, cat_p95, alpha=0.5, color=colorcat_d[cat])

        ax_ni.set_xlim(0,85)
        ax_ni.set_ylim(0,1.1)

        if scs.index(sc) == 2:
            ax_ni.legend(bbox_to_anchor=(0.8,-0.15), frameon=False)
        
        ax_ni.annotate(labels_d[sc], xy=(0.5, 0.92), xycoords="axes fraction")

axes_ni[0,0].set_ylabel("Fraction community members \n[% buildings in block]")
axes_ni[1,0].set_ylabel("Fraction community members \n[% buildings in block]")
axes_ni[2,0].set_ylabel("Fraction community members \n[% buildings in block]")
axes_ni[2,0].set_xlabel("Block size [number of buildings]")
axes_ni[2,1].set_xlabel("Block size [number of buildings]")
axes_ni[2,2].set_xlabel("Block size [number of buildings]")

#%% EXPORT FIGURE
fig_ni.savefig(files_dir +"\\fig_ni.svg", format="svg")
fig_ni.savefig(files_dir +"\\fig_ni.png", format="png", bbox_inches="tight", dpi=210)

#%% PLOT BLOCK SIZES

blocks = np.array([x["plot_id"] for x in ag_d.values()])
block_d = {id: np.sum(blocks == id) for id in set(blocks)}
block_sizes = np.array(list(block_d.values()))

fig_blocks, ax_blocks = plt.subplots(1,1,figsize=(6.3,4))

ax_blocks.hist(block_sizes, bins=np.arange(0,100,1), edgecolor="k")

ax_blocks.set_xlim(1,100)
ax_blocks.set_ylim(0,30)

ax_blocks.set_xlabel("Block size [number of buildings]")
ax_blocks.set_ylabel("Absolute frequency [number of blocks]")

#%% EXPORT FIGURE
fig_blocks.savefig(files_dir +"\\fig_blocks.svg", format="svg")
fig_blocks.savefig(files_dir +"\\fig_blocks.png", format="png", bbox_inches="tight", dpi=210)

#%% PLOT NUMBERS OF COMMUNITIES PER CATEGORY
import seaborn as sns

scs = ["com_2019_no_dm", "com_2019_dm_100", "com_2019_dm_1"]

cats = ["all_residential", "all_commercial", "mixed_use"]

colorcat_d = {"all_residential":"red", "all_commercial":"blue", "mixed_use":"green"}

fig_cats, ax_cats = plt.subplots(3,1, figsize=(6.5,6), sharex=True)
# Remove space between subplots
plt.subplots_adjust(wspace=0, hspace=0)

for sc in scs:

    data = communities_df.loc[sc]

    cats_df = pd.DataFrame(None, columns=cats, index=range(50))

    for run in set(communities_df["run"].values):

        data_run = data.loc[data["run"]==run]

        for cat in cats:
            cats_df.at[run, cat] = np.sum([(data_run["category"].values==cat)])
    
    sns.stripplot(data=cats_df, ax=ax_cats[scs.index(sc)], orient="h", alpha=0.5)
    sns.boxplot(data=cats_df, ax=ax_cats[scs.index(sc)], orient="h", showbox=False, showfliers=False, sym='k.',whis=[5,95],
    zorder=1)

    ax_cats[scs.index(sc)].set_xlim(-5,105)

    ax_cats[scs.index(sc)].annotate(labels_d[sc], xy=(0.92,0.85), xycoords="axes fraction")

    ax_cats[scs.index(sc)].set_yticklabels(["All commerical", "All residential", "Mixed use"])

ax_cats[2].set_xlabel("Solar communities [number]")
#%% EXPORT FIGURE
fig_cats.savefig(files_dir +"\\fig_stripplots.svg", format="svg")
fig_cats.savefig(files_dir +"\\fig_stripplots.png", format="png", bbox_inches="tight", dpi=210)

#%% PLOT EVOLUTION OF NUMBER OF COMMUNITIES
scs = ["com_2019_no_dm", "com_2019_dm_100", "com_2019_dm_1"]

cats = ["all_residential", "all_commercial", "mixed_use"]

colorcat_d = {"all_residential":"red", "all_commercial":"blue", "mixed_use":"green"}

fig_ncoms, ax_ncoms = plt.subplots(3,1, figsize=(6.5,10), sharex=True)

# Remove space between subplots
plt.subplots_adjust(wspace=0, hspace=0)

for sc in scs:

    data = communities_df.loc[sc]

    res_com = []
    com_com = []
    mix_com = []

    for run in set(communities_df["run"].values):

        data_run = data.loc[data["run"]==run]

        n_res_com = np.array([np.sum((data_run["category"].values=="all_residential") & (data_run["year"].values==yr)) for yr in range(26)])
        n_com_com = np.array([np.sum((data_run["category"].values=="all_commercial") & (data_run["year"].values==yr)) for yr in range(26)])
        n_mix_com = np.array([np.sum((data_run["category"].values=="mixed_use") & (data_run["year"].values==yr)) for yr in range(26)])

        ax_ncoms[scs.index(sc)].plot(np.cumsum(n_res_com), color='blue', alpha=0.1)
        ax_ncoms[scs.index(sc)].plot(np.cumsum(n_com_com), color='darkorange', alpha=0.1)
        ax_ncoms[scs.index(sc)].plot(np.cumsum(n_mix_com), color='green', alpha=0.1)

        res_com.append(np.cumsum(n_res_com))
        com_com.append(np.cumsum(n_com_com))
        mix_com.append(np.cumsum(n_mix_com))

    ax_ncoms[scs.index(sc)].fill_between(range(26),np.cumsum(np.percentile(mix_com, axis=0, q=5)),np.cumsum(np.percentile(res_com, axis=0, q=95)), color='lime', alpha=0.25)
    ax_ncoms[scs.index(sc)].fill_between(range(26),np.cumsum(np.percentile(res_com, axis=0, q=5)),np.cumsum(np.percentile(res_com, axis=0, q=95)), color='cyan', alpha=0.25)
    ax_ncoms[scs.index(sc)].fill_between(range(26),np.cumsum(np.percentile(com_com, axis=0, q=5)),np.cumsum(np.percentile(res_com, axis=0, q=95)), color='orange', alpha=0.25)

    ax_ncoms[scs.index(sc)].plot(np.median(mix_com, axis=0), color='green')
    ax_ncoms[scs.index(sc)].plot(np.median(res_com, axis=0), color='blue')
    ax_ncoms[scs.index(sc)].plot(np.median(com_com, axis=0), color='darkorange')
    
    ax_ncoms[scs.index(sc)].set_ylim(0,109)
    ax_ncoms[scs.index(sc)].set_xlim(9,25)

    #ax_ncoms[scs.index(sc)].set_yscale("log")

    ax_ncoms[scs.index(sc)].annotate(labels_d[sc], xy=(0.02,0.92), xycoords="axes fraction")

    ax_ncoms[scs.index(sc)].set_ylabel("Solar communities [-]")

ax_ncoms[2].set_xticks(np.arange(9,26,2))
ax_ncoms[2].set_xticklabels(np.arange(2019,2036,2))

# add custom legend
from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color="green", lw=4),
                Line2D([0], [0], color="blue", lw=4),
                Line2D([0], [0], color="darkorange", lw=4)]

ax_ncoms[2].legend(custom_lines, ['Mixed use', 'All commercial', 'All residential'], ncol=3, bbox_to_anchor=(0.92,-0.1), frameon=False)
#%% EXPORT FIGURE
fig_ncoms.savefig(files_dir +"\\fig_ncoms_evolution.svg", format="svg")
fig_ncoms.savefig(files_dir +"\\fig_ncoms_evolution.png", format="png", bbox_inches="tight", dpi=210)
#%% PLOT EVOLUTION OF member size

var = "n_members"

labels_d = {"com_2019_no_dm":"COM", "com_2019_dm_100":"ZEV", "com_2019_dm_1":"ZEV+"}

fig_comscr, axes_comscr = plt.subplots(3,3, figsize=(10,12), sharey=True, sharex=True)

# Remove space between subplots
plt.subplots_adjust(wspace=0, hspace=0)

scs = ["com_2019_no_dm", "com_2019_dm_100", "com_2019_dm_1"]

cats = ["all_residential", "all_commercial", "mixed_use"]

colorcat_d = {"all_residential":"blue", "all_commercial":"darkorange", "mixed_use":"green"}

for sc in scs:

    data = communities_df.loc[sc]

    for cat in cats:

        ax_cs = axes_comscr[scs.index(sc), cats.index(cat)]

        members_median = []
        members_p5 = []
        members_p95 = []
        years = set(data["year"].loc[(data["category"].values == cat)])

        for year in years:

            members_median.append(np.median(data[var].loc[(data["category"].values == cat) & (data["year"].values == year)]))

            members_p5.append(np.percentile(data[var].loc[(data["category"].values == cat) & (data["year"].values == year)], q=5))

            members_p95.append(np.percentile(data[var].loc[(data["category"].values == cat) & (data["year"].values == year)], q=95))

        ax_cs.plot(sorted(list(years)), members_median, label=cat, color=colorcat_d[cat])

        ax_cs.fill_between(sorted(list(years)), members_p5, members_p95, alpha=0.5, color=colorcat_d[cat])

        ax_cs.set_xlim(8,27)
        #ax_cs.set_ylim(0,1.1)

        if scs.index(sc) == 2:
            ax_cs.legend(bbox_to_anchor=(0.8,-0.1), frameon=False)
        
        ax_cs.annotate(labels_d[sc], xy=(0.5, 0.92), xycoords="axes fraction")

# Set X-axis labels
d_yrs = 5
for i in range(3):
    axes_comscr[2,i].set_xticks(np.arange(8,len(x),d_yrs))
    axes_comscr[2,i].set_xticklabels(np.arange(2019,max(model_df["sim_year"])+1,d_yrs))
    axes_comscr[i,0].set_ylabel("Self-consumption rate \n[% solar generation consumed on site]")

#%% PLOT COMMUNITY SIZE EVOLUTION

labels_d = {"com_2019_no_dm":"COM", "com_2019_dm_100":"ZEV", "com_2019_dm_1":"ZEV+"}

fig_comsize, axes_comsize = plt.subplots(3,3, figsize=(10,12), sharey=True, sharex=True)

# Remove space between subplots
plt.subplots_adjust(wspace=0, hspace=0)

scs = ["com_2019_no_dm", "com_2019_dm_100", "com_2019_dm_1"]

cats = ["all_residential", "all_commercial", "mixed_use"]

colorcat_d = {"all_residential":"blue", "all_commercial":"darkorange", "mixed_use":"green"}

for sc in scs:

    data = communities_df.loc[sc]

    for cat in cats:

        ax_cs = axes_comsize[scs.index(sc), cats.index(cat)]

        members_median = []
        members_p5 = []
        members_p95 = []
        years = set(data["year"].loc[(data["category"].values == cat)])

        for year in years:

            members_median.append(np.median(data["n_members"].loc[(data["category"].values == cat) & (data["year"].values == year)]))

            members_p5.append(np.percentile(data["n_members"].loc[(data["category"].values == cat) & (data["year"].values == year)], q=5))

            members_p95.append(np.percentile(data["n_members"].loc[(data["category"].values == cat) & (data["year"].values == year)], q=95))

        ax_cs.plot(sorted(list(years)), members_median, label=cat, color=colorcat_d[cat])

        ax_cs.fill_between(sorted(list(years)), members_p5, members_p95, alpha=0.5, color=colorcat_d[cat])

        ax_cs.set_xlim(8,27)
        ax_cs.set_ylim(0,13)

        if scs.index(sc) == 2:
            ax_cs.legend(bbox_to_anchor=(0.8,-0.1), frameon=False)
        
        ax_cs.annotate(labels_d[sc], xy=(0.5, 0.92), xycoords="axes fraction")

axes_comsize[0,0].set_ylabel("Community size [number of buildings]")
axes_comsize[1,0].set_ylabel("Community size [number of buildings]")
axes_comsize[2,0].set_ylabel("Community size [number of buildings]")

# Set X-axis labels
d_yrs = 5
for i in range(3):
    axes_comsize[2,i].set_xticks(np.arange(8,len(x),d_yrs))
    axes_comsize[2,i].set_xticklabels(np.arange(2019,max(model_df["sim_year"])+1,d_yrs))
#%% EXPORT FIGURE
fig_comsize.savefig(files_dir +"\\fig_comsize.svg", format="svg")
fig_comsize.savefig(files_dir +"\\fig_comsize.png", format="png", bbox_inches="tight", dpi=210)

#%% PLOT EVOLUTION OF SCR

var = "SCR"

labels_d = {"com_2019_no_dm":"COM", "com_2019_dm_100":"ZEV", "com_2019_dm_1":"ZEV+"}

fig_comscr, axes_comscr = plt.subplots(3,3, figsize=(10,12), sharey=True, sharex=True)

# Remove space between subplots
plt.subplots_adjust(wspace=0, hspace=0)

scs = ["com_2019_no_dm", "com_2019_dm_100", "com_2019_dm_1"]

cats = ["all_residential", "all_commercial", "mixed_use"]

colorcat_d = {"all_residential":"blue", "all_commercial":"darkorange", "mixed_use":"green"}

for sc in scs:

    data = communities_df.loc[sc]

    for cat in cats:

        ax_cs = axes_comscr[scs.index(sc), cats.index(cat)]

        members_median = []
        members_p5 = []
        members_p95 = []
        years = set(data["year"].loc[(data["category"].values == cat)])

        for year in years:

            members_median.append(np.median(data[var].loc[(data["category"].values == cat) & (data["year"].values == year)]))

            members_p5.append(np.percentile(data[var].loc[(data["category"].values == cat) & (data["year"].values == year)], q=5))

            members_p95.append(np.percentile(data[var].loc[(data["category"].values == cat) & (data["year"].values == year)], q=95))

        ax_cs.plot(sorted(list(years)), members_median, label=cat, color=colorcat_d[cat])

        ax_cs.fill_between(sorted(list(years)), members_p5, members_p95, alpha=0.5, color=colorcat_d[cat])

        ax_cs.set_xlim(8,27)
        ax_cs.set_ylim(0,1.1)

        if scs.index(sc) == 2:
            ax_cs.legend(bbox_to_anchor=(0.8,-0.1), frameon=False)
        
        ax_cs.annotate(labels_d[sc], xy=(0.5, 0.92), xycoords="axes fraction")

# Set X-axis labels
d_yrs = 5
for i in range(3):
    axes_comscr[2,i].set_xticks(np.arange(8,len(x),d_yrs))
    axes_comscr[2,i].set_xticklabels(np.arange(2019,max(model_df["sim_year"])+1,d_yrs))
    axes_comscr[i,0].set_ylabel("Self-consumption rate \n[% solar generation consumed on site]")
#%% EXPORT FIGURE
fig_comscr.savefig(files_dir +"\\fig_comscr.svg", format="svg")
fig_comscr.savefig(files_dir +"\\fig_comscr.png", format="png", bbox_inches="tight", dpi=210)

#%% PLOT SCR PER COMMUNITY CATEGORY
import matplotlib.ticker as mtick

cats = ["all_residential", "all_commercial", "mixed_use"]

scs = ["com_2019_no_dm", "com_2019_dm_100", "com_2019_dm_1"]

colors_d = {"all_residential":"darkorange", "all_commercial":"blue", "mixed_use":"green"}

fig_scr, axes_scr = plt.subplots(3,3, figsize=(6.5,8), sharex=True)
# Remove space between subplots
plt.subplots_adjust(hspace=0)
plt.tight_layout()

for sc in scs:

    data = communities_df.loc[sc]

    for cat in cats:

        ax = axes_scr[scs.index(sc), cats.index(cat)]

        ax.hist(data["SC"].loc[(data["category"].values == cat)]/data["demand"].loc[(data["category"].values == cat)], alpha=1, bins=np.arange(0,1.05, 0.025), label=cat, density=True, color=colors_d[cat])

        ax.set_xlim(0,1)

        ax.set_title(labels_d[sc])
        
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))

        axes_scr[2,cats.index(cat)].set_xlabel("Self-sufficiency rate [%]")

    axes_scr[scs.index(sc), 0].set_ylabel("Share of communities [%]")
#%% EXPORT FIGURE
fig_scr.savefig(files_dir +"\\fig_scrcatsc.svg", format="svg")
fig_scr.savefig(files_dir +"\\fig_scrcatsc.png", format="png", bbox_inches="tight", dpi=210)
#%% PLOT SSR PER COMMUNITY CATEGORY
import matplotlib.ticker as mtick

cats = ["all_residential", "all_commercial", "mixed_use"]

scs = ["com_2019_no_dm", "com_2019_dm_100", "com_2019_dm_1"]

colors_d = {"all_residential":"darkorange", "all_commercial":"blue", "mixed_use":"green"}

fig_scr_all, axes_scr = plt.subplots(1,3, figsize=(6.5,3), sharex=True)
plt.tight_layout()

data = communities_df

for cat in cats:

    ax = axes_scr[cats.index(cat)]

    ax.hist(data["SC"].loc[(data["category"].values == cat)]/data["demand"].loc[(data["category"].values == cat)], alpha=1, bins=np.arange(0,1.05, 0.05), label=cat, density=True, color=colors_d[cat], edgecolor="k")

    ax.set_xlim(0,1)

    ax.set_title(cat.replace("_", " ").title())
    
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(100,decimals=0))
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1,decimals=0))

    axes_scr[cats.index(cat)].set_xlabel("Self-sufficiency rate [%]")

axes_scr[0].set_ylabel("Share of communities [%]")
#%% EXPORT FIGURE
fig_scr_all.savefig(files_dir +"\\fig_scr_all.svg", format="svg")
fig_scr_all.savefig(files_dir +"\\fig_scr_all.png", format="png", bbox_inches="tight", dpi=210)
#%% PLOT SCR PER COMMUNITY CATEGORY
import matplotlib.ticker as mtick

cats = ["all_residential", "all_commercial", "mixed_use"]

scs = ["com_2019_no_dm", "com_2019_dm_100", "com_2019_dm_1"]

colors_d = {"all_residential":"darkorange", "all_commercial":"blue", "mixed_use":"green"}

fig_scr_all, axes_scr = plt.subplots(2,3, figsize=(6.5,5))
plt.tight_layout()
plt.subplots_adjust(hspace=0.3)

data = communities_df

for cat in cats:

    binwidth = 0.05

    ssr = data["SC"].loc[(data["category"].values == cat)].values / data["demand"].loc[(data["category"].values == cat)].values 

    axes_scr[0, cats.index(cat)].hist(data["SCR"].loc[(data["category"].values == cat)].values, alpha=0.75, bins=np.arange(0,1.05, binwidth), label=cat, density=True, color=colors_d[cat])

    axes_scr[0, cats.index(cat)].axvline(x=np.median(data["SCR"].loc[(data["category"].values == cat)].values), color='k')

    axes_scr[1, cats.index(cat)].hist(ssr, alpha=0.75, bins=np.arange(0,1.05, binwidth), label=cat, density=True, color=colors_d[cat])

    axes_scr[1, cats.index(cat)].axvline(x=np.median(ssr), color='k')

    axes_scr[0, cats.index(cat)].set_xlim(0,1)
    axes_scr[1, cats.index(cat)].set_xlim(0,1)

    axes_scr[0, cats.index(cat)].set_title(cat.replace("_", " ").title())

    yticks = axes_scr[0, cats.index(cat)].get_yticks()
    axes_scr[0, cats.index(cat)].set_yticklabels(['{:,.0%}'.format(x*binwidth) for x in yticks])
    yticks = axes_scr[1, cats.index(cat)].get_yticks()
    axes_scr[1, cats.index(cat)].set_yticklabels(['{:,.0%}'.format(x*binwidth) for x in yticks])

    if cat == "all_commercial":
        axes_scr[1, 1].annotate("Median", xytext=(0.5, 0.72), xycoords="axes fraction",xy=(0.28, 0.6), arrowprops=dict(arrowstyle="->"))
    
    
    from matplotlib.ticker import AutoMinorLocator
    for i in range(2):
        axes_scr[i, cats.index(cat)].xaxis.set_major_formatter(mtick.PercentFormatter(1,decimals=0))
        axes_scr[i, cats.index(cat)].xaxis.set_minor_locator(AutoMinorLocator())

    axes_scr[0, cats.index(cat)].set_xlabel("Self-consumption rate [%]")
    axes_scr[1, cats.index(cat)].set_xlabel("Self-sufficiency rate [%]")

axes_scr[0,0].set_ylabel("Share of communities [%]")
axes_scr[1,0].set_ylabel("Share of communities [%]")

#%% EXPORT FIGURE
fig_scr_all.savefig(files_dir +"\\fig_scr_all.svg", format="svg")
fig_scr_all.savefig(files_dir +"\\fig_scr_all.png", format="png", bbox_inches="tight", dpi=210)

#%% PLOT NUMBER OF COMMUNITIES WITH GREATER DEMAND

cats = ["all_residential", "all_commercial", "mixed_use"]

scs = ["com_2019_no_dm", "com_2019_dm_100", "com_2019_dm_1"]

labels_d = {"com_2019_no_dm":"COM", "com_2019_dm_100":"ZEV", "com_2019_dm_1":"ZEV+"}

color_d = {"com_2019_no_dm":(230/255,159/255,0.), "com_2019_dm_100":(86/255,180/255,233/255), "com_2019_dm_1":(0.,114/255,178/255)}

fig_cd, ax = plt.subplots(1,1, figsize=(6.5,4), sharex=True, sharey=True)
plt.subplots_adjust(wspace=0)

for sc in scs:
    
    data_sc = communities_df.loc[sc]

    med_sc = np.zeros(50)
    p5_sc = np.zeros(50)
    p95_sc = np.zeros(50)

    nums_d = []

    for run in range(50):

        data_run = data_sc.loc[data_sc["run"]==run]

        demands = data_run["demand"].values / 1000

        cat_d = np.arange(0,5000,100)
        num_d = np.array([np.sum((demands > d)) for d in cat_d])

        if np.max(num_d) > 0:
            nums_d.append(num_d/np.max(num_d))
        else:
            nums_d.append(num_d)

    median = np.median(np.array(nums_d), axis=0)
    p5 = np.percentile(np.array(nums_d), axis=0, q=5)
    p95 = np.percentile(np.array(nums_d), axis=0, q=95)

    ax.plot(median, label=labels_d[sc], color=color_d[sc])

    ax.fill_between(range(len(cat_d)), p5,p95,alpha=0.2, color=color_d[sc], zorder=z)

ax.axvline(x=1, color="k", linestyle="--")

ax.annotate("100 MWh/year", xy=(1,0.85), xytext=(7,0.9), arrowprops=dict(arrowstyle="->"), ha="left")

ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())

import matplotlib.ticker as mtick
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))

ax.set_ylim(0,1)
ax.set_xlim(0,40)

ax.set_xlabel("Community electricity demand [MWh/year]")
ax.set_ylabel("Communities with larger demand [%]")

ax.set_xticklabels(np.arange(0,4500,500))

ax.legend(frameon=False)

#%% EXPORT FIGURE
fig_cd.savefig(files_dir +"\\fig_com_dem_num.svg", format="svg")
fig_cd.savefig(files_dir +"\\fig_com_dem_num.png", format="png", bbox_inches="tight", dpi=210)

#%% PLOT communities net-present values

cats = ["all_residential", "all_commercial", "mixed_use"]

scs = ["com_2019_no_dm", "com_2019_dm_100", "com_2019_dm_1"]

labels_d = {"com_2019_no_dm":"COM", "com_2019_dm_100":"ZEV", "com_2019_dm_1":"ZEV+"}

color_d = {"com_2019_no_dm":(230/255,159/255,0.), "com_2019_dm_100":(86/255,180/255,233/255), "com_2019_dm_1":(0.,114/255,178/255)}

fig_cnpv, axes = plt.subplots(3,1, figsize=(6.5,5), sharex=True, sharey=True)
plt.subplots_adjust(wspace=0, hspace=0)

for sc in scs:

    ax = axes[scs.index(sc)]
    
    data_sc = communities_df.loc[sc]

    npvs = []

    for run in range(50):

        data_run = data_sc.loc[data_sc["run"]==run]

        npvs.extend(data_run["npv"].values / 1000)

    bins = np.logspace(1,5,100)

    ax.hist(np.clip(npvs, bins[0],bins[-1]), color=color_d[sc], bins=bins)

    ax.axvline(x=np.median(npvs), linestyle="--", linewidth=0.5, color="k")
    ax.annotate("Median: "+str(round(np.median(npvs)/1000,1))+" m CHF ", xy=(np.median(npvs),0.05*len(npvs)), ha="right", fontsize=8)
    ax.axvline(x=np.average(npvs), linestyle="--", linewidth=0.5,  color="red")
    ax.annotate(" Average: "+str(round(np.average(npvs)/1000,1))+" m CHF ", xy=(np.average(npvs),0.05*len(npvs)),color="red", fontsize=8)

    ax.annotate(labels_d[sc], xy=(0.02,0.85), xycoords="axes fraction")

    print("n ", len(npvs))

    ax.set_ylim(0,0.06*len(npvs))
    ax.set_xlim(10,100000)
    ax.set_xscale("log")

    ax.set_yticks(np.arange(0,0.06,0.02)*len(npvs))
    ax.set_yticklabels(['{:,.1%}'.format(x) for x in np.arange(0,0.06,0.02)])

axes[2].set_xlabel("Community net-present value [k CHF]")

fig_cnpv.text(0.02, 0.5, "Share of communities [%]", rotation=90, va="center")
#%% EXPORT FIGURE
fig_cnpv.savefig(files_dir +"\\fig_com_npv.svg", format="svg")
fig_cnpv.savefig(files_dir +"\\fig_com_npv.png", format="png", bbox_inches="tight", dpi=210)
#%% PLOT SIZE PER COMMUNITY CATEGORY
import matplotlib.ticker as mtick

cats = ["all_residential", "all_commercial", "mixed_use"]

scs = ["com_2019_no_dm", "com_2019_dm_100", "com_2019_dm_1"]

colors_d = {"all_residential":"darkorange", "all_commercial":"blue", "mixed_use":"green"}

fig_com_size_hist, axes_scr = plt.subplots(1,3, figsize=(6.5,3), sharex=True)
plt.tight_layout()

data = communities_df

for cat in cats:

    ax = axes_scr[cats.index(cat)]

    ax.hist(data["n_members"].loc[(data["category"].values == cat)], alpha=1, bins=np.arange(0,21, 1), label=cat, density=True, color=colors_d[cat], edgecolor="k")

    ax.set_xlim(0,20)

    ax.set_title(cat.replace("_", " ").title())
    
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))

    axes_scr[cats.index(cat)].set_xlabel("Community members [-]")

axes_scr[0].set_ylabel("Share of communities [%]")
#%% EXPORT FIGURE
fig_com_size_hist.savefig(files_dir +"\\fig_com_size_hist.svg", format="svg")
fig_com_size_hist.savefig(files_dir +"\\fig_com_size_hist.png", format="png", bbox_inches="tight", dpi=210)
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
    ax_inst.fill_between(x,plot_df["p05"].loc[plot_df["variable"]=="inst_cum_ind"].values+plot_df["p05"].loc[plot_df["variable"]=="inst_cum_com"].values,plot_df["p95"].loc[plot_df["variable"]=="inst_cum_ind"].values+plot_df["p05"].loc[plot_df["variable"]=="inst_cum_com"].values, alpha=0.10)

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
#%% PLOT EXAMPLE OF MIXED USE COMMUNITY -> most frequent mixed use one
import sys, os, re, json, glob, time, pickle, datetime, feather

# Define path to data files
data_path = 'c:\\Users\\anunezji\\Documents\\P4_comm_solar\\code\\COSA_Data\\'

# Define file name for data inputs
solar_data_file = "CEA_Disaggregated_SolarPV_3Dec.pickle"
demand_data_file = "CEA_Disaggregated_TOTAL_FINAL_06MAR.pickle"

# Import data of solar irradiation resource
solar = pd.read_pickle(os.path.join(data_path, solar_data_file))

# Import data of electricity demand profiles
demand = pd.read_pickle(os.path.join(data_path, demand_data_file))

# Find most common combination for mixed use communities
list_combinations = ["_".join(x) for x in communities_df["building_uses"].loc[communities_df["category"]=="mixed_use"].values if len(x) == 3]

names_list = list(communities_df["community_id"].loc[communities_df["category"]=="mixed_use"].values)

mode_combinations = max(set(names_list), key=names_list.count)
mode_combinations = "B2366130_B2366109_B148467"
#mode_combinations = "B2371401_B302030889_B145527"
# mode_combinations = "B2366130_B2366109_B148467"
#mode_combinations = "B302019273_B143535_B2366932"
# PLOT STARTS HERE

week_summer = 36
week_winter = 2

buildings = mode_combinations.split("_")

# Create a figure with two columns (summer, winter)
fig_com_example, axes_ce = plt.subplots(4,2, figsize=(10,12.313), sharex=True, sharey=True)

# Remove space between subplots
plt.subplots_adjust(wspace=0, hspace=0)

data_d = {}

for col in range(2):

    if col == 0:
        week = week_winter
    elif col == 1:
        week = week_summer

    data_d[col] = {}

    for b in buildings:

        data_d[col][b] = {}

        start = week*7*24+48
        end = (week+1)*7*24+48

        sun = solar[b].values[start:end]
        dem = demand[b].values[start:end]

        grid = np.array([dem[i] - sun[i] if (dem[i] > sun[i]) else 0 for i in range(len(dem))])

        sc = np.array([sun[i] if (dem[i] > sun[i]) else dem[i] for i in range(len(dem))])

        fed = np.array([sun[i] - dem[i] if (sun[i] > dem[i]) else 0 for i in range(len(dem))])


        data_d[col][b]["sun"] = sun
        data_d[col][b]["dem"] = dem
        data_d[col][b]["grid"] = grid
        data_d[col][b]["sc"] = sc
        data_d[col][b]["fed"] = fed

        axes_ce[buildings.index(b),col].fill_between(range(7*24), np.zeros(7*24), data_d[col][b]["grid"], color="cornflowerblue")
        #axes_ce[buildings.index(b),col].plot(data_d[col][b]["grid"], color="k", linestyle="--", linewidth=0.5)
        #axes_ce[buildings.index(b),col].plot((-1)*data_d[col][b]["fed"], color="k", linestyle="--", linewidth=0.5)
        axes_ce[buildings.index(b),col].plot(data_d[col][b]["dem"], color="midnightblue", linewidth=0.5)

        axes_ce[buildings.index(b),col].fill_between(range(7*24), np.zeros(7*24), (-1)*data_d[col][b]["sun"], color="gold")
        axes_ce[buildings.index(b),col].fill_between(range(7*24), np.zeros(7*24), (-1)*data_d[col][b]["sc"], color="orangered")
        axes_ce[buildings.index(b),col].plot((-1)*data_d[col][b]["sun"], color="darkred", linewidth=0.5)

        axes_ce[buildings.index(b),col].set_xlim(0,7*24)
        axes_ce[buildings.index(b),col].axhline(y=0, color="k")

        axes_ce[buildings.index(b),0].set_ylabel("Power [kW]")

        if buildings.index(b) != 1:
            axes_ce[buildings.index(b),col].annotate(ag_d[b]["bldg_type"], xy=(0.98,0.93), xycoords="axes fraction", fontsize=8, ha="right")

            axes_ce[buildings.index(b),col].annotate("SCR="+'{:,.1%}'.format(np.sum(sc)/np.sum(sun)), xy=(0.98,0.85), xycoords="axes fraction", fontsize=8, ha="right")

            axes_ce[buildings.index(b),col].annotate("SSR="+'{:,.1%}'.format(np.sum(sc)/np.sum(dem)), xy=(0.98,0.78), xycoords="axes fraction", fontsize=8, ha="right")
        else:
            axes_ce[buildings.index(b),col].annotate(ag_d[b]["bldg_type"], xy=(0.98,0.98), xycoords="axes fraction", fontsize=8, ha="right")

            axes_ce[buildings.index(b),col].annotate("SCR="+'{:,.1%}'.format(np.sum(sc)/np.sum(sun)), xy=(0.98,0.9), xycoords="axes fraction", fontsize=8, ha="right")

            axes_ce[buildings.index(b),col].annotate("SSR="+'{:,.1%}'.format(np.sum(sc)/np.sum(dem)), xy=(0.98,0.83), xycoords="axes fraction", fontsize=8, ha="right")
   
    # Compute data for community

    com = "_".join(buildings)

    data_d[col][com] = {}

    sun = np.sum([data_d[col][b]["sun"] for b in buildings],axis=0)
    data_d[col][com]["sun"] = sun

    dem = np.sum([data_d[col][b]["dem"] for b in buildings],axis=0)
    data_d[col][com]["dem"] = dem

    data_d[col][com]["grid"] = np.array([dem[i] - sun[i] if (dem[i] > sun[i]) else 0 for i in range(len(dem))])
    data_d[col][com]["sc"] = np.array([sun[i] if (dem[i] > sun[i]) else dem[i] for i in range(len(dem))])
    data_d[col][com]["fed"] = np.array([sun[i] - dem[i] if (sun[i] > dem[i]) else 0 for i in range(len(dem))])
       
    axes_ce[3,col].fill_between(range(7*24), np.zeros(7*24), data_d[col][com]["grid"], color="cornflowerblue")
    #axes_ce[3,col].plot(data_d[col][com]["grid"], color="k", linestyle="--", linewidth=0.5)
    axes_ce[3,col].plot(data_d[col][com]["dem"], color="midnightblue", linewidth=0.5)

    axes_ce[3,col].fill_between(range(7*24), np.zeros(7*24), (-1)*data_d[col][com]["sun"], color="gold")
    axes_ce[3,col].fill_between(range(7*24), np.zeros(7*24), (-1)*data_d[col][com]["sc"], color="orangered")
    axes_ce[3,col].plot((-1)*data_d[col][com]["sun"], color="darkred", linewidth=0.5)

    axes_ce[3,col].axhline(y=0, color="k", linewidth=0.5)

    axes_ce[3,0].set_ylabel("Power [kW]")

    axes_ce[3,col].set_xticks(np.arange(0,7*24, 24))

    axes_ce[3,col].annotate("Community", xy=(0.98,0.21), xycoords="axes fraction", fontsize=8, ha="right")

    axes_ce[3,col].annotate("SCR="+'{:,.1%}'.format(np.sum(data_d[col][com]["sc"] )/np.sum(sun)), xy=(0.98,0.14), xycoords="axes fraction", fontsize=8, ha="right")

    axes_ce[3,col].annotate("SSR="+'{:,.1%}'.format(np.sum(data_d[col][com]["sc"] )/np.sum(dem)), xy=(0.98,0.07), xycoords="axes fraction", fontsize=8, ha="right")

axes_ce[3,0].set_xlabel("Hour of the week")
axes_ce[3,1].set_xlabel("Hour of the week")

axes_ce[0,0].set_title("Example winter week (w=2)")
axes_ce[0,1].set_title("Example summer week (w=36)")

from matplotlib.ticker import AutoMinorLocator
axes_ce[0,0].yaxis.set_minor_locator(AutoMinorLocator())
axes_ce[0,0].xaxis.set_minor_locator(AutoMinorLocator())
axes_ce[0,0].set_ylim(-135,135)

for col in range(2):
    for row in range(3):
        axes_ce[row,col].spines['top'].set_visible(False)
        axes_ce[row,col].spines['bottom'].set_visible(False)
    axes_ce[3,col].spines['top'].set_visible(False)

# add custom legend
from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color="midnightblue", lw=1, linestyle="--"),
                Line2D([0], [0], color="cornflowerblue", lw=4),
                Line2D([0], [0], color="darkred", lw=1),
                Line2D([0], [0], color="orangered", lw=4),
                Line2D([0], [0], color="gold", lw=4)
                ]

axes_ce[3,0].legend(custom_lines, ['Load profile', 'Net demand', 'PV generation', 'Self-consumption', 'PV exports'], ncol=5, bbox_to_anchor=(2,-0.2), frameon=False)

#%% EXPORT FIGURE
fig_com_example.savefig(files_dir +"\\fig_com_example.svg", format="svg")
fig_com_example.savefig(files_dir +"\\fig_com_example.png", format="png", bbox_inches="tight", dpi=210)