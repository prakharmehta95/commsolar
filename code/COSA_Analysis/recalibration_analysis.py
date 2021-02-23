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
data_subfolder = 'code\\COSA_Outputs\\1_calibration\\cal_dec\\10_rep\\'
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
        label = "_".join(input_file.split("_")[7:8])
        print(label)
        
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

# Add new label
model_df["file_label"] = model_df["cal_label"] + ["_"+x[0] for x in list(model_df.index)]

#%% ANALYSE MODEL RESULTS PER VARIABLE

vars = ['n_ind', 'inst_cum_ind', 'n_com', 'inst_cum_com']

n_ticks =  len(set(model_df["sim_year"]))
n_ticks = 15
n_runs = len(set(model_df["run"]))

sc_results = {}
#for cal_lab in list(set(model_df["cal_label"])):
for cal_lab in list(set(model_df["file_label"])):
    sc_results[cal_lab] = {}

    for var in vars:
        sc_results[cal_lab][var] = pd.DataFrame(np.zeros((n_ticks, n_runs)),
        index=range(n_ticks), columns=range(n_runs))
        
        for run in range(n_runs):

            #cond = (model_df["cal_label"]==cal_lab) & (model_df["run"]==run)
            cond = (model_df["file_label"]==cal_lab) & (model_df["run"]==run)
            
            sc_results[cal_lab][var][run] = model_df[var].loc[cond].values[:15]
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

#%% CALIBRATION ANALYSIS

# Create new variables to store detail analysis resuls and summary
cal_analysis = {}
cal_summary = 0

# Loop through all calibration combinations
for sc, sc_df in sc_results.items():

    data_df = sc_df.loc[sc_df["variable"]=="inst_cum_ind"]

    # Add calibration series
    data_df["inst_cum_ind_cal"] = cal_data["inst_cum_ZH_wiedikon_cal"]

    # Compute number of repetitions for each scenario
    n_rep = len(sc_df.columns) - 6
    # Note: 6 is the number of columns that "variable", "sim_year", "p05", "p50", "p95", and "CI90" take in the dataframes.

    # Loop through each simulation run and compute square-root error to cal
    for run in range(n_rep):
        data_df["diff_"+str(run)] = ((np.array(data_df[run]) - np.array(cal_data["inst_cum_ZH_wiedikon_cal"])) ** 2) ** 0.5

    # Analyse errors
    col_labs = ["diff_"+str(nrun) for nrun in range(n_rep)]
    data_df["diff_sum"] = np.sum(data_df[col_labs].values, axis=1)
    data_df["diff_median"] = np.median(data_df[col_labs].values, axis=1)
    data_df["diff_av"] = np.average(data_df[col_labs].values, axis=1)
    data_df["diff_p05"] = np.quantile(data_df[col_labs].values, axis=1, q=0.05)
    data_df["diff_p95"] = np.quantile(data_df[col_labs].values, axis=1, q=0.95)

    # Store results per scenario
    cal_analysis[sc] = data_df

    # Summarize results and store them
    if type(cal_summary) == int:
        vars = ["diff_"+x for x in ["sum", "median", "av", "p05", "p95"]]
        cal_summary = pd.DataFrame(data_df[vars].sum(axis=0), columns=[sc])
    
    else:
        cal_summary[sc] = data_df[vars].sum(axis=0).values

# Transpose and order the summary of results by "diff_sum"
cal_summary = cal_summary.transpose().sort_values(by="diff_median")

# Add calibration parameter columns
ix = cal_summary.index.values
cal_summary["w_e"] = pd.to_numeric(pd.Series(ix).str.split("_").str[0]).values
cal_summary["w_p"] = pd.to_numeric(pd.Series(ix).str.split("_").str[1]).values
cal_summary["w_a"] = pd.to_numeric(pd.Series(ix).str.split("_").str[2]).values

#%% PLOT ERROR HEATMAP

# Create error figure
fig_err_heatmap, ax_err_heatmap = plt.subplots(1,2, figsize=(6.5,4))

# Plot error bubbles
errors = ax_err_heatmap[0].scatter(x=cal_summary["w_e"], y=cal_summary["w_a"], s=20, marker="s", c=cal_summary["diff_median"]/1000, cmap="RdYlGn_r", vmin=0, vmax=100, alpha=0.5)

# Plot zoom into calibration region
ax_err_heatmap[1].scatter(x=cal_summary["w_e"], y=cal_summary["w_a"], s=100, marker="s", c=cal_summary["diff_median"]/1000, cmap="RdYlGn_r", vmin=0, vmax=100, alpha=0.5, edgecolor="k")

# Set limits of zoom-in
ax_err_heatmap[1].set_ylim(0.34,0.44)
ax_err_heatmap[1].set_xlim(0.08,0.21)

# Add axes labels
ax_err_heatmap[0].set_xlabel("$w_{e}$")
ax_err_heatmap[0].set_ylabel("$w_{a}$")
ax_err_heatmap[1].set_xlabel("$w_{e}$")

# Create axis for colorbar
cbar_ax = fig_err_heatmap.add_axes([0.92, 0.1, 0.02, 0.8])

# Add color bar
fig_err_heatmap.colorbar(errors, cax=cbar_ax,label="Median error [GW]")
#%% EXPORT FIGURE
fig_err_heatmap.savefig(files_dir +"\\cal_err_heatmap.svg", format="svg")
fig_err_heatmap.savefig(files_dir +"\\cal_err_heatmap.png", format="png", bbox_inches="tight", dpi=210)

#%% PLOT HEATMAP IN 3D
from mpl_toolkits.mplot3d import Axes3D

fig_err_3d = plt.figure(figsize=(8,6))
ax_3d = fig_err_3d.add_subplot(111, projection='3d')

xs = cal_summary["w_e"]
ys = cal_summary["w_p"]
zs = cal_summary["w_a"]
errors = ax_3d.scatter(xs, ys, zs, s=50, c=cal_summary["diff_median"]/1000,cmap="RdYlGn_r", edgecolor="k", vmin=0, vmax=100)

ax_3d.set_xlabel("$w_{e}$")
ax_3d.set_ylabel("$w_{p}$")
ax_3d.set_zlabel("$w_{a}$")

ax_3d.tick_params(axis='both', which='major', labelsize=8)

# Add color bar
cb = plt.colorbar(errors, label="Median error [GW]")

plt.show()
#%% EXPORT FIGURE
fig_err_3d.savefig(files_dir +"\\cal_err_3d.svg", format="svg")
fig_err_3d.savefig(files_dir +"\\cal_err_3d.png", format="png", bbox_inches="tight", dpi=210)
#%% PLOT INDIVIDUAL CAL PARAMETER ERROR

# Define calibration parameters and labels
pars = ["w_e", "w_p", "w_a"]
pars_labs = {"w_e":"$w_{e}$", "w_p":"$w_{p}$", "w_a":"$w_{a}$"}

# Create error figure
fig_ind_err, axes_ind_err = plt.subplots(1,3, figsize=(6.5,4), sharey=True)

# Remove space between subplots
plt.subplots_adjust(wspace=0, hspace=0)

# Loop through the variables
for par in pars:

    # Select subplot
    ax = axes_ind_err[pars.index(par)]

    # Plot error bubbles
    errors = ax.scatter(x=cal_summary[par], y=cal_summary["diff_median"]/1000, c=cal_summary["diff_median"]/1000, cmap="RdYlGn_r", #edgecolor="k",
    linewidth=0.5, alpha=0.5, vmin=0, vmax=100)

    # Compute median and 50% confidence interval of errors
    median = []
    p25 = []
    p75 = []
    for xvar in sorted(list(set(cal_summary[par]))):
        median.append(np.median(cal_summary["diff_median"].loc[cal_summary[par]==xvar])/1000)
        p25.append(np.percentile(cal_summary["diff_median"].loc[cal_summary[par]==xvar],q=25)/1000)
        p75.append(np.percentile(cal_summary["diff_median"].loc[cal_summary[par]==xvar],q=75)/1000)
    
    # Plot median error
    ax.plot(sorted(list(set(cal_summary[par]))),median, color="gray")
    
    # Plot 50% confidence interval
    ax.fill_between(sorted(list(set(cal_summary[par]))),p25,p75, color="gray", alpha=0.5)

    # Add axes labels
    ax.set_xlabel(pars_labs[par])

    # Set x-axis limit
    ax.set_xlim(-0.1, .55)

# Add vertical axis label
axes_ind_err[0].set_ylabel("Median error [GW]")

# Create axis for colorbar
cbar_ax = fig_ind_err.add_axes([0.92, 0.1, 0.02, 0.8])

# Add color bar
fig_ind_err.colorbar(errors, cax=cbar_ax,label="Median error [GW]")
#%% EXPORT FIGURE
fig_ind_err.savefig(files_dir +"\\ind_cal_par_errors.svg", format="svg")
fig_ind_err.savefig(files_dir +"\\ind_cal_par_errors.png", format="png", bbox_inches="tight", dpi=210)
#%% PLOT INSTALLED CAPACITIES PER SCENARIO

n_sc = 10

# Create figure
fig_cal, ax_cal = plt.subplots(1,1, figsize=(6.5,4))

# Colormap
col = plt.cm.coolwarm(np.linspace(0,1,n_sc))

# for sc in set(sc_results_analysed.index):
for sc in list(cal_summary.index)[:n_sc]:

    # Select data
    plot_df = sc_results_analysed.loc[sc,:]

    # Define time variable
    x = range(len(set(plot_df["sim_year"])))

    # Define scenario label
    v = sc.split("_")[:4]
    sc_lab = "$w_{e}=$"+str(v[0])+", $w_{p}=$"+str(v[1])+", $w_{a}=$"+str(v[2])
    sc_lab = plot_df.index[0].split("_")[-1]

    # Define variable to plot
    cond_var = plot_df["variable"]=="inst_cum_ind"

    # Plot median
    ax_cal.step(range(15),plot_df["p50"].loc[cond_var].values, label=sc_lab,
    #color=col[list(cal_summary.index).index(sc)])
    #color=scs.index(sc)
    )

# Plot calibration data
zh_col = (36/255,139/255,204/255)
cal = cal_data['inst_cum_ZH_wiedikon_cal']
ax_cal.step(range(10),cal[:10], color="k", linestyle="--", label="Wiedikon (estimated)")
proj = [np.nan]*9
proj.extend(cal[-6:])
ax_cal.step(range(15),proj, color="red", linestyle="--", label="Wiedikon (15% growth)")

# Set Y-axis limits
#ax_cal.set_ylim(0,6000)
ax_cal.set_ylim(0,)

# Add labels to axes
ax_cal.set_ylabel("Cumulative installed capacity [kWp]")
#ax_cal.set_xlabel("Simulation year")

# Arrange X-axis ticks and labels
ax_cal.set_xticks(np.arange(0,len(x),2))
ax_cal.set_xticklabels(np.arange(min(model_df["sim_year"]),max(model_df["sim_year"])+1,2))

# Add legend
ax_cal.legend(loc='upper left', ncol=1, frameon=False, fontsize=8)
#%% EXPORT FIGURE
fig_cal.savefig(files_dir +"\\calibration_cal_el11.svg", format="svg")
fig_cal.savefig(files_dir +"\\calibration_cal_el11.png", format="png", bbox_inches="tight", dpi=210)
#%% PLOT INSTALLED CAPACITIES OF BEST CALIBRATION

zh_col = (36/255,139/255,204/255)

fig_best, ax_best = plt.subplots(1,1, figsize=(6.5,4))

# Select best calibration
sc = list(cal_summary.index)[0]
#sc = '0.13_0.1_0.45_0.32_0.5_0_0.698_0.18_0.95_no'
sc = '0.13_0.1_0.45_0.32_0.5_0_0.698_0.18_0.95_recal-3'
# '0.13_0.3_0.45_0.12_0.5_0_0.698_0.18_0.95_recal-3'
# '0.13_0.2_0.45_0.22_0.5_0_0.698_0.18_0.95_recal-3'
#sc = '0.14_0.1_0.44_0.32_0.5_0_0.698_0.18_0.95_recal-3'
# '0.14_0.2_0.44_0.22_0.5_0_0.698_0.18_0.95_recal-3'
# '0.14_0.3_0.44_0.12_0.5_0_0.698_0.18_0.95_recal-3'
# '0.14_0.3_0.45_0.11_0.5_0_0.698_0.18_0.95_recal-3'
# '0.13_0.3_0.44_0.13_0.5_0_0.698_0.18_0.95_recal-3'
# '0.13_0.2_0.44_0.23_0.5_0_0.698_0.18_0.95_recal-3'
# '0.14_0.1_0.45_0.31_0.5_0_0.698_0.18_0.95_recal-3'
# '0.14_0.2_0.45_0.21_0.5_0_0.698_0.18_0.95_recal-3'

# Define scenario label
v = sc.split("_")[:4]
sc_lab = "$w_{e}=$"+str(v[0])+", $w_{p}=$"+str(v[1])+", $w_{a}=$"+str(v[2])

# Select data
plot_df = sc_results_analysed.loc[sc,:]

# Define time variable
x = range(len(set(plot_df["sim_year"])))

cond_var = plot_df["variable"]=="inst_cum_ind"

ax_best.fill_between(x, plot_df["p05"].loc[cond_var].values, plot_df["p95"].loc[cond_var].values, alpha=0.25, color="cornflowerblue")

for run in set(model_df["run"]):
    ax_best.plot(plot_df[run].loc[cond_var].values, color=zh_col, alpha=0.1)

ax_best.plot(plot_df["p50"].loc[cond_var].values, color=zh_col)

# Plot calibration data
zh_col = (36/255,139/255,204/255)
cal = cal_data['inst_cum_ZH_wiedikon_cal']
ax_best.plot(cal[:10], color="k", linestyle="--")
proj = [np.nan]*9
proj.extend(cal[-6:])
ax_best.plot(proj, color="red", linestyle="--")

ax_best.set_ylabel("Cumulative installed capacity [kWp]")
ax_best.set_xticks(np.arange(0,len(x),2))
ax_best.set_xticklabels(np.arange(min(model_df["sim_year"]),max(model_df["sim_year"])+1,2))

#ax_best.set_ylim(0,10000)
ax_best.set_title(sc_lab+", diff_median: "+str(int(cal_summary["diff_median"].loc[sc]))+", 90% C.I. ("+str(int(cal_summary["diff_p05"].loc[sc]))+", "+str(int(cal_summary["diff_p95"].loc[sc]))+")", fontsize=10)

# Add legend
leg_elements = [
    Line2D([0], [0], color=zh_col, lw=1),
    Line2D([0], [0], color='cornflowerblue', lw=4, alpha=0.25),
    Line2D([0], [0], color=zh_col, lw=1, alpha=0.1),
    Line2D([0], [0], color='k', lw=1, linestyle="--"),
    Line2D([0], [0], color='red', lw=1, linestyle="--"),]
leg_labels = [
    "Median 50 simulations",
    "90% Confidence interval",
    "Individual run",
    "Wiedikon (estimated)",
    "Wiedikon (15% growth)"
    ]
ax_best.legend(handles = leg_elements , labels=leg_labels,loc='upper left', 
             ncol=1, frameon=False)

#%% EXPORT FIGURE
fig_best.savefig(files_dir +"\\calibration_best_el11.svg", format="svg")
fig_best.savefig(files_dir +"\\calibration_best_el11.png", format="png", bbox_inches="tight", dpi=210)

#%%

var = "ideation_total"
#var = "pp"
#var = "ind_inv"
#var = "neighbor_influence"
#var = "peer_effect"
#var = "ind_scr"
#var = "ind_npv"

fig_h, ax_h = plt.subplots(1,1)

# Only for payback period
# plot_var_0 = [15*(1-x) for x in list(set(agents_df[var].loc[(agents_df["sim_year"]==0) & (agents_df["run"]==0)]))]
# plot_var_8 = [15*(1-x) for x in list(set(agents_df[var].loc[(agents_df["sim_year"]==8) & (agents_df["run"]==0)]))]

plot_var_0 = list(set(agents_df[var].loc[(agents_df["sim_year"]==0) & (agents_df["run"]==0)]))
plot_var_8 = list(set(agents_df[var].loc[(agents_df["sim_year"]==8) & (agents_df["run"]==0)]))

ax_h.hist(plot_var_0, bins=100, color="blue", alpha=0.5)
ax_h.hist(plot_var_8, bins=100, color="red", alpha=0.5)
ax_h.set_xlabel(var)
ax_h.set_ylabel("Number of agents")

#%%
new_df = pd.DataFrame(None)
new_df["attitude"] = agents_df["attitude"].loc[agents_df["run"]==0]
new_df["pp"] = agents_df["pp"].loc[agents_df["run"]==0]

activated_ags = {}
for k_a in np.arange(0.4, 0.51, 0.01):
    
    for k_p in np.arange(0.1, 0.21, 0.01):
        
        label = str(k_a)+"_"+str(k_p)

        new_df[label] = np.array(new_df["pp"]) * k_p + np.array(new_df["attitude"]) * k_a

        activated_ags[label] = (np.array(new_df[label]) > 0.5).sum()

chosen = {k:activated_ags[k] for k in list(activated_ags.keys()) if ((activated_ags[k] > 0) and (activated_ags[k] < 1000))}

#%% PLOT IDEATION VARIABLES

ax_d = {"pp":(0,0), "neighbor_influence":(1,0), "peer_effect":(1,1), "attitude":(0,1)}

fig_idea, axes_idea = plt.subplots(nrows=2,ncols=2,figsize=(6.5,6.5))

for var in ["pp", "neighbor_influence", "peer_effect", "attitude"]:

    # Select subplot
    ax = axes_idea[ax_d[var][0], ax_d[var][1]]


    ax.hist(list(agents_df[var].loc[(agents_df["sim_year"]==0) & (agents_df["run"]==0)]), bins=100, color="blue", alpha=0.5)

    ax.hist(list(agents_df[var].loc[(agents_df["sim_year"]==8) & (agents_df["run"]==0)]), bins=100, color="red", alpha=0.5)

    ax.set_xlabel(var)

    ax.set_xlim(0,1)

#%% PLOT AGENTS WITH POSITIVE INTENTION OVER TIME

cd = {"cal14testFITtiming":"red", "cal14testpeers":"blue"}

fig_int, ax_int = plt.subplots(1,1)

for t in ["cal14testFITtiming", "cal14testpeers"]:
    df = agents_df.loc[t,:]

    for r in range(1):
        int_r = []
        for y in range(15):
            int_v = np.sum(df["intention"].loc[(df["run"]==r) & (df["sim_year"]==y)].values)

            int_r.append(int_v)

        ax_int.plot(np.cumsum(int_r), c=cd[t], label=t, alpha=0.5)

ax_int.set_ylim(0,)
ax_int.set_xticklabels(range(2008,2026,2))
ax_int.set_ylabel("Agents with positive intention")
ax_int.legend(loc='upper left')

#%% PLOT SIZES of AGENTS WITH POSITIVE INTENTION OVER TIME

cd = {"cal14testFITtiming":"red", "cal14testpeers":"blue"}

fig_s, ax_s = plt.subplots(1,1, figsize=(8,6))

for t in ["cal14testFITtiming"]:
    df = agents_df.loc[t,:]

    for r in range(1):
        s_r = []
        for y in range(15):
            ids = df["building_id"].loc[(df["run"]==r) & (df["sim_year"]==y) & (df["intention"]==1)].values

            s_v = [agents_info["pv_size_kw"].loc[bid] for bid in ids]

            s_r.append(s_v)

            ax_s.scatter([y]*len(s_v),s_v)
            ax_s.scatter(y, np.mean(s_v),c="k",marker="_")

ax_s.axhline(y=np.mean(agents_info["pv_size_kw"].values), c="red", ls="--",lw=1)
ax_s.set_ylim(0,)
ax_s.set_xticklabels(range(2008,2025,2))
ax_s.set_ylabel("Sizes of agents with positive intention")
#ax_s.legend(loc='upper left')

#%% PLOT SIZES of AGENTS WITH POSITIVE INTENTION OVER TIME

cd = {"cal14testFITtiming":"red", "cal14testpeers":"blue"}

sizes = [0, 10, 30, 100, 500, 1000]

fig_s, ax_s = plt.subplots(1,1, figsize=(8,6))

for t in ["cal14testFITtiming"]:
    df = agents_df.loc[t,:]

    for r in range(1):

        for s in sizes:

            s_r = []

            for y in range(15):  

                if sizes.index(s) == 0:

                    ids = []
                else:
                    ids = list(agents_info.loc[(agents_info["pv_size_kw"]<s) & (agents_info["pv_size_kw"]>=sizes[sizes.index(s)-1])].index.values)

                s_v = np.mean(df["pp"].loc[(df["run"]==r) & (df["sim_year"]==y)& (df["building_id"].isin(ids))].values)

                s_r.append(s_v)

            ax_s.plot(s_r, label="Systems up to "+str(s)+" kW")

ax_s.set_ylim(0,)
ax_s.set_xticklabels(range(2008,2025,2))
ax_s.set_ylabel("Payback period variable (before weight)")
ax_s.legend(loc='upper left')

#%% PLOTS for exploring data

color_d = {
    "inst_cum_ind":"blue",
    #"inst_cum_com":"green",
    #"n_ind":"blue", 
#"n_com":"red"
}

fig_inst, ax_inst = plt.subplots(1,1, figsize=(6.5,4))

for sc in set(sc_results_analysed.index):

    # Select data
    plot_df = sc_results_analysed.loc[sc,:]

    x = range(len(set(plot_df["sim_year"])))

    for var in color_d.keys():

        cond_var = plot_df["variable"]==var

        #if (plot_df["p50"].loc[cond_var].values[0] < 500) and (plot_df["p50"].loc[cond_var].values[-1] < 10000):
        if sc[:3]=="0.1":

            #ax_inst.fill_between(x, plot_df["p05"].loc[cond_var].values, plot_df["p95"].loc[cond_var].values, alpha=0.25, color=color_d[var])

            #for run in set(model_df["run"]):
            #    ax_inst.plot(plot_df[run].loc[cond_var].values, color=color_d[var], alpha=0.5)

            ax_inst.plot(plot_df["p50"].loc[cond_var].values, color=color_d[var])

        
            print(sc)
            print(plot_df["p50"].loc[cond_var].values[-1])
            ax_inst.annotate(sc, xy=(14, plot_df["p50"].loc[cond_var].values[-1]))


ax_inst.set_ylabel("Number of individual adopters")
ax_inst.set_xticks(np.arange(0,len(x),2))
ax_inst.set_xticklabels(np.arange(min(model_df["sim_year"]),max(model_df["sim_year"])+1,2))

#ax_inst.set_ylim(0,10000)

# Plot calibration data
cal = cal_data['inst_cum_ZH_wiedikon_cal']
ax_inst.plot(cal[:10], color="red", linestyle="--")
proj = [np.nan]*9
proj.extend(cal[-6:])
ax_inst.plot(proj, color="red", linestyle="--")

#%% PLOT ONE SCENARIO

sc = '0.1_0.2_0.5_0.2_0.5_0_0.698_0.18_0.95'
sc = '0.1_0.4_0.5_0_0.5_0_0.698_0.18_0.95'
#sc = '0.05_0.25_0.4_0.3_0.5_0_0.698_0.18_0.95'
# zero
#sc = '0.05_0.25_0.45_0.25_0.5_0_0.698_0.18_0.95'
# zero
#sc = '0.05_0.35_0.35_0.25_0.5_0_0.698_0.18_0.95'
# zero
#sc = '0.05_0.35_0.45_0.15_0.5_0_0.698_0.18_0.95'
# zero
#sc = '0.05_0.35_0.4_0.2_0.5_0_0.698_0.18_0.95'
# zero
#sc = '0.15_0.25_0.35_0.25_0.5_0_0.698_0.18_0.95'
# zero
#sc = '0.15_0.25_0.4_0.2_0.5_0_0.698_0.18_0.95'
# adoption only after 2018
#sc = '0.15_0.25_0.45_0.15_0.5_0_0.698_0.18_0.95'
# good shape, too high, compare with last
#sc = '0.15_0.35_0.35_0.15_0.5_0_0.698_0.18_0.95'
# zero
# sc = '0.15_0.35_0.4_0.1_0.5_0_0.698_0.18_0.95'
# adoption only after 2018
#sc = '0.15_0.35_0.45_0.05_0.5_0_0.698_0.18_0.95'
# Good shape, but begins too high
#sc = '0.08_0.34_0.42_0.16_0.5_0_0.698_0.18_0.95'
# zero
# sc = '0.08_0.34_0.4_0.18_0.5_0_0.698_0.18_0.95'
# zero
# sc = '0.12_0.34_0.42_0.12_0.5_0_0.698_0.18_0.95'
# installations only after 2018
# sc = '0.12_0.34_0.4_0.14_0.5_0_0.698_0.18_0.95'
# very few installations after 2019
sc = '0.14_0.34_0.42_0.1_0.5_0_0.698_0.18_0.95'
# GOOD - good fit >2018, low before but not zero!
#sc = '0.14_0.34_0.4_0.12_0.5_0_0.698_0.18_0.95'
# only >2018
# sc = '0.1_0.34_0.42_0.14_0.5_0_0.698_0.18_0.95'
# too low >2018
# sc = '0.1_0.34_0.4_0.16_0.5_0_0.698_0.18_0.95'
# zero
#sc = '0.14_0.34_0.46_0.06_0.5_0_0.698_0.18_0.95'
# good, but too high early on
sc = '0.16_0.34_0.44_0.06_0.5_0_0.698_0.18_0.95'
# GOOD GOOD - but too high late
#sc = '0.16_0.34_0.46_0.04_0.5_0_0.698_0.18_0.95'
# too high
#sc = '0.13_0.34_0.42_0.11_0.5_0_0.698_0.18_0.95'
# too low until 2018, later a bit low
sc = '0.13_0.34_0.43_0.1_0.5_0_0.698_0.18_0.95'
# PROMISING
sc = '0.13_0.34_0.44_0.09_0.5_0_0.698_0.18_0.95'
# **** VERY GOOD, although a bit too high
sc = '0.14_0.34_0.42_0.1_0.5_0_0.698_0.18_0.95'
# *** excellent fit after 2019, too low before
sc = '0.14_0.34_0.43_0.09_0.5_0_0.698_0.18_0.95'
# ** very good fit until 2018, later gets a bit high
#sc = '0.14_0.34_0.44_0.08_0.5_0_0.698_0.18_0.95'
# good but high
#sc = '0.14_0.34_0.4_0.12_0.5_0_0.698_0.18_0.95'
# too low
#sc = '0.15_0.34_0.44_0.07_0.5_0_0.698_0.18_0.95'
# good but high
#sc = '0.16_0.34_0.43_0.07_0.5_0_0.698_0.18_0.95'
# good but high
#sc = '0.16_0.34_0.44_0.06_0.5_0_0.698_0.18_0.95'
# good, high

scs = [
    '0.13_0.34_0.43_0.1_0.5_0_0.698_0.18_0.95',
    '0.14_0.34_0.42_0.1_0.5_0_0.698_0.18_0.95',
    '0.14_0.34_0.43_0.09_0.5_0_0.698_0.18_0.95',
]
#scs = [
#    "0.14_0.3_0.42_0.14_0.5_0_0.698_0.18_0.95",
#    "0.35_0.2_0.25_0.2_0.5_0_0.698_0.18_0.95",
    #"0.3_0.2_0.3_0.2_0.5_0_0.698_0.18_0.95",
    #"0.4_0.2_0.2_0.2_0.5_0_0.698_0.18_0.95",
    #'0.45_0.2_0.3_0.05_0.5_0_0.698_0.18_0.95'
#]

#scs = [
 #'0.13_0.1_0.43_0.34_0.5_0_0.698_0.18_0.95',
 #'0.13_0.2_0.43_0.24_0.5_0_0.698_0.18_0.95',
 #'0.13_0.3_0.43_0.14_0.5_0_0.698_0.18_0.95',
 #'0.13_0.4_0.43_0.04_0.5_0_0.698_0.18_0.95',
# '0.14_0.1_0.42_0.34_0.5_0_0.698_0.18_0.95',
 #'0.14_0.2_0.42_0.24_0.5_0_0.698_0.18_0.95',
 #'0.14_0.3_0.42_0.14_0.5_0_0.698_0.18_0.95',
 #'0.14_0.4_0.42_0.04_0.5_0_0.698_0.18_0.95'
#]
# Cal8
# '0.13_0.1_0.42_0.35_0.5_0_0.698_0.18_0.95'
# '0.13_0.2_0.42_0.25_0.5_0_0.698_0.18_0.95'
# '0.13_0.3_0.42_0.15_0.5_0_0.698_0.18_0.95'
# '0.13_0.4_0.42_0.05_0.5_0_0.698_0.18_0.95'
# Too low

#'0.14_0.1_0.43_0.33_0.5_0_0.698_0.18_0.95'
#'0.14_0.2_0.43_0.23_0.5_0_0.698_0.18_0.95'
#'0.14_0.3_0.43_0.13_0.5_0_0.698_0.18_0.95'
#'0.14_0.4_0.43_0.03_0.5_0_0.698_0.18_0.95'
# Good early, too high later


fig_one, ax_one = plt.subplots(1,1)

for sc in scs:

    sc_data = sc_results_analysed.loc[sc]
    plot_data = sc_data.loc[sc_data["variable"]=="inst_cum_ind"]

    ax_one.plot(plot_data["p50"].values, color="blue")
    ax_one.fill_between(range(15), plot_data["p05"].values, plot_data["p95"].values, alpha=0.25, color="blue")

    # Plot calibration data
    cal = cal_data['inst_cum_ZH_wiedikon_cal']
    ax_one.plot(cal[:10], color="red", linestyle="--")
    proj = [np.nan]*9
    proj.extend(cal[-6:])
    ax_one.plot(proj, color="red", linestyle="--")

    ax_one.set_title(sc)
    ax_one.set_xticklabels(range(2008,2026,2))

    ax_one.annotate(sc, xy=(14,plot_data["p50"].values[-1]))

    ax_one.set_ylim(0,6000)

# %%

ag0 = agents_df.loc[agents_df["run"]==0].copy()

#%%
fig, axes = plt.subplots(2,2, figsize=(8,8), sharex=True, sharey="row")
plt.tight_layout()

fig_at, ax_at = plt.subplots(1,2, figsize=(10,6), sharey=True)

row = {"attitude":0, "pp":0, "peer_effect":1, "neighbor_influence":1}
col = {"attitude":0, "pp":1, "peer_effect":0, "neighbor_influence":1}
c = {"cal14testFITtiming":"red", "cal14testpeers":"blue"}
wd = {"attitude":"w_att", "pp":"w_econ", "peer_effect":"w_swn", "neighbor_influence":"w_subplot"}

for v in ["pp", "peer_effect", "neighbor_influence"]:

    ax = axes[row[v], col[v]]

    for t in ["cal14testFITtiming", "cal14testpeers"]:

        v_mean = []
        df = ag0.loc[t,:]
        w = df[wd[v]].values[0]

        for r in range(15):
            v_mean.append(w*np.mean(df[v].loc[(df["sim_year"]==r)].values))
        
        ax.plot(range(15), v_mean, c=c[t], label=t)
        #ax.plot(range(14), [(v_mean[ix] - v_mean[ix-1]) for ix in range(1,len(v_mean))], c=c[t])
        #ax.axhline(y=0, c="black", linestyle="--")

        ax.set_title(v+", weight = "+str(w))
        ax.set_ylabel("Influence on ideation after weights [-]")

        ax.set_xticklabels(range(2008, 2025,2))

        #ax.set_ylim(0,0.03)
    
    if v == "neighbor_influence":
        ax.legend(loc="upper center")
#%%
td = {"cal14testFITtiming":0, "cal14testpeers":1}
for t in ["cal14testFITtiming", "cal14testpeers"]:

    v_mean = []
    df = ag0.loc[t,:]

    for r in range(15):
        v_mean.append(df["ideation_total"].loc[(df["sim_year"]==r)].values)

    ax_at[td[t]].plot(v_mean, alpha=0.5)
    ax_at[td[t]].set_title(t)

#%% COMPARISON WITH RESULT OUTPUTS FROM JULY
af = "2020-07-16-20-02-14-360775_com_2019_dm_100_intention_agent_vars_.feather"
af = "2020-07-11-09-16-15-730938_cal_19_agent_vars_.feather"
af_dir = "C:\\Users\\anunezji\\Documents\\P4_comm_solar\\code\\COSA_Outputs\\2_results\\el-change-11-new-cal-fallback-intention\\"
af_dir = "C:\\Users\\anunezji\\Documents\\P4_comm_solar\\code\\COSA_Outputs\\1_calibration\\cal_july\\"

ag_old_df = feather.read_dataframe(af_dir+af)

#%% PLOT IDEATION VARIABLES

fig, axes = plt.subplots(2,2, figsize=(8,8), sharex=True, sharey="row")
plt.tight_layout()

row = {"attitude":0, "pp":0, "peer_effect":1, "neighbor_influence":1}
col = {"attitude":0, "pp":1, "peer_effect":0, "neighbor_influence":1}
c = {"cal14testFITtiming":"red", "cal14testpeers":"blue"}
wd = {"attitude":0, "pp":1, "peer_effect":2, "neighbor_influence":3}

for v in ["attitude", "pp", "peer_effect", "neighbor_influence"]:

    ax = axes[row[v], col[v]]

    v_mean = []
    w = float(ag_old_df["cal_label"].values[0].split("_")[wd[v]])

    for r in range(15):
        v_mean.append(w*np.mean(ag_old_df[v].loc[(ag_old_df["sim_year"]==r)].values))
    
    ax.plot(range(15), v_mean)

    ax.set_title(v+", weight = "+str(w))
    ax.set_ylabel("Influence on ideation after weights [-]")

    ax.set_xticklabels(range(2008, 2025,2))

    #ax.set_ylim(0,)
    
    if v == "neighbor_influence":
        ax.legend(loc="upper center")