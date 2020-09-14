# -*- coding: utf-8 -*-
"""
Current version: July, 2020
@authors: Alejandro Nuñez-Jimenez
"""
#%% IMPORT REQUIRED PACKAGES
import sys, os, re, json, glob, time, pickle, datetime, feather

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

from os.path import join
from matplotlib import colors
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator

#%% IMPORT DATA
files_dir = os.path.dirname(__file__)

# Set directory with data files
data_subfolder = 'code\\COSA_Outputs\\2_results\\el-change-11-new-cal-fallback-intention\\'
#data_subfolder = 'code\\COSA_Outputs\\'
input_subfolder = 'code\\COSA_Data\\'
data_dir = files_dir[:files_dir.rfind('code')] + data_subfolder
input_dir = files_dir[:files_dir.rfind('code')] + input_subfolder

# Import agents info
agents_info = pd.read_csv(input_dir+'buildings_info.csv', sep=",")

# Make agent id the index
agents_info = agents_info.set_index("bldg_name")

# Import calibration data
with open(input_dir+"cal_data.json", "r") as cal_file:
    cal_data = json.load(cal_file)

# Define path to data files
data_path = 'c:\\Users\\anunezji\\Documents\\P4_comm_solar\\code\\COSA_Data\\'

# Define file name for data inputs
solar_data_file = "CEA_Disaggregated_SolarPV_3Dec.pickle"
demand_data_file = "CEA_Disaggregated_TOTAL_FINAL_06MAR.pickle"

# Import data of solar irradiation resource
solar = pd.read_pickle(os.path.join(data_path, solar_data_file))

# Import data of electricity demand profiles
demand = pd.read_pickle(os.path.join(data_path, demand_data_file))

#%% PLOT NUMBER AND TYPE OF BUILDINGS

# Count the number of buildings per type of use
typeb_d = {typeb: np.sum(agents_info["bldg_type"].values == typeb) for typeb in set(agents_info["bldg_type"].values)}
sizeb_d = {typeb: np.sum(agents_info["pv_size_kw"].loc[agents_info["bldg_type"].values == typeb].values) for typeb in set(agents_info["bldg_type"].values)}

# Sort dictionary
typeb_d = {k: v for k, v in sorted(typeb_d.items(), key=lambda item: item[1])}
sizeb_d = {k: v for k, v in sorted(sizeb_d.items(), key=lambda item: item[1])}

# Plot bars
fig_use, ax_use = plt.subplots(1,2, figsize=(6.5,4),)

# Remove space between subplots
plt.subplots_adjust(wspace=0, hspace=0)

# Go through each building use
for typeb in list(typeb_d.keys()):

    # Plot each bar
    ax_use[0].barh(typeb, typeb_d[typeb])
    ax_use[1].barh(typeb, sizeb_d[typeb]/1000)

    # Annotate each bar
    ax_use[0].annotate(str(typeb_d[typeb]), xy=(50+typeb_d[typeb], list(typeb_d.keys()).index(typeb)), va="center", ha="right", fontsize=8)
    ax_use[1].annotate(str(round(sizeb_d[typeb]/1000,1)), xy=(1+sizeb_d[typeb]/1000, list(typeb_d.keys()).index(typeb)), va="center", ha="left", fontsize=8)

# Put labels of axes
ax_use[0].set_xlabel("Number of buildings")
ax_use[1].set_xlabel("Potential PV capacity [MWp]")
#ax_use[0].set_ylabel("Building use")

# Put X-axes on top
for i in range(2):
    #ax_use[i].set_xscale("log")
    ax_use[i].xaxis.tick_top()
    ax_use[i].xaxis.set_label_position('top') 

# Set axes limits
ax_use[0].set_xlim(0, 4000)
ax_use[1].set_xlim(0, 150)

# Invert X-axis of first subplot
ax_use[0].invert_xaxis()

# Remove ticks and labels from Y-axis in second subplot
ax_use[1].set_yticks([])
ax_use[1].set_yticklabels([])
#%% EXPORT FIGURE
fig_use.savefig(files_dir +"\\fig_building_use_inst.svg", format="svg")
fig_use.savefig(files_dir +"\\fig_building_use_inst.png", format="png", bbox_inches="tight", dpi=210)

#%% PLOT ANNUAL GENERATION PROFILES
freq = "day"

f = 1
if freq == "day":
    f = 24
elif freq == "week":
    f = 24*7
          
fig_sun, ax = plt.subplots(2,1, figsize=(6.5,8))

y = solar[list(agents_info.index)[0]].values

day = [np.sum(y[x*24:x*24+24]) for x in range(1+int(len(y)/24))]

ax[1].plot([np.sum(y[x*f:x*f+f]) for x in range(1+int(len(y)/f))], color="red", linewidth=0.5)

ax[1].set_xlim(0,365)
ax[1].set_ylim(0,55)

ax[1].set_ylabel("PV generation [kWh/day]")
ax[1].set_xlabel("Day of the year")

ax[1].annotate(list(agents_info.index)[0], xy=(0.98, 0.92), xycoords="axes fraction", ha="right")
ax[1].annotate(str(agents_info.at[list(agents_info.index)[0],"pv_size_kw"])+" kWp", xy=(0.98, 0.86), xycoords="axes fraction", ha="right")

ag = 0

y = solar[list(agents_info.index)[ag]].values

day = np.array([y[x*24:x*24+24] for x in range(int(len(y)/24))])

yd = 0
for d in day:
    color = plt.cm.coolwarm(200*np.sum(d)/np.sum(y))
    ax[0].plot(d, linewidth=0.25, alpha=0.5, color=color)
    yd += 1

ax[0].plot(np.median(day, axis=0), color="k", linestyle="--", label="Median")
ax[0].fill_between(range(24), np.percentile(day, axis=0, q=25), np.percentile(day, axis=0, q=75), alpha=0.5, color="k", label="Q1-Q3 interval")

#ax.yaxis.set_minor_locator(AutoMinorLocator())
ax[0].xaxis.set_minor_locator(AutoMinorLocator())
ax[1].xaxis.set_minor_locator(AutoMinorLocator())

ax[0].set_xlim(0,23)
ax[0].set_ylim(0,10)

ax[0].set_ylabel("PV generation [kW]")
ax[0].set_xlabel("Hour of the day")

ax[0].annotate(list(agents_info.index)[ag], xy=(0.98, 0.92), xycoords="axes fraction", ha="right")
ax[0].annotate(str(agents_info.at[list(agents_info.index)[ag],"pv_size_kw"])+" kWp", xy=(0.98, 0.86), xycoords="axes fraction", ha="right")
#%% EXPORT FIGURE
fig_sun.savefig(files_dir +"\\fig_sun_yr_day.svg", format="svg")
fig_sun.savefig(files_dir +"\\fig_sun_yr_day.png", format="png", bbox_inches="tight", dpi=210)
#%% PLOT MATRIX OF DAILY DEMAND

d = 32*7+2

ax_d = {'SINGLE_RES': [0,0], 'LIBRARY':[0,1], 'PARKING':[0,2], 
'RESTAURANT': [1,0], 'INDUSTRIAL':[1,1], 'HOSPITAL':[1,2], 
'RETAIL': [3,0], 'HOTEL':[3,1], 'GYM':[3,2], 
'SCHOOL': [2,0], 'MULTI_RES':[2,1], 'OFFICE':[2,2]}

fig_dem, axes_dem = plt.subplots(4,3,figsize=(6.5,9),sharex=True, sharey="row")
# Remove space between subplots
plt.subplots_adjust(wspace=0, hspace=0)

for use in set(agents_info['bldg_type'].values):

    row =  ax_d[use][0]
    col =  ax_d[use][1]

    if use == "PARKING":
        ag = 3
    else:
        ag = 0

    b = list(agents_info.loc[agents_info["bldg_type"].values==use].index)[ag]

    axes_dem[row,col].fill_between(range(24),np.zeros(24),demand[b][24*d:24*(d+1)].values, color="cornflowerblue")

    axes_dem[row,col].plot(demand[b][24*d:24*(d+1)].values, color="midnightblue", linewidth=0.5)

    #for d in range(365):
    #    color = plt.cm.coolwarm(200*np.sum(demand[b][24*d:24*(d+1)].values)/np.sum(demand[b].values))
    #    axes_dem[row,col].plot(demand[b][24*d:24*(d+1)].values, #color="midnightblue", 
    #    linewidth=0.5, alpha=0.5, color=color)

    axes_dem[row,col].annotate(use, xy=(0.5,0.9), xycoords="axes fraction", ha="center")

axes_dem[0,1].set_ylim(0,2)
axes_dem[1,1].set_ylim(0,11)
axes_dem[2,1].set_ylim(0,22)
axes_dem[3,1].set_ylim(0,108)

for row in range(4):
    axes_dem[row,0].set_ylabel("Power demand [kW]")
    #axes_dem[row,0].yaxis.set_minor_locator(AutoMinorLocator())

for col in range(3):
    
    axes_dem[3,col].set_xlim(0,23)
    axes_dem[3,col].set_xlabel("Hour of the day")
    axes_dem[3,col].xaxis.set_minor_locator(AutoMinorLocator())
    axes_dem[3,col].set_xticks(np.arange(0,24,4))

#%% EXPORT FIGURE
fig_dem.savefig(files_dir +"\\fig_dem_matrix.svg", format="svg")
fig_dem.savefig(files_dir +"\\fig_dem_matrix.png", format="png", bbox_inches="tight", dpi=210)

#%% PRINT MAIN CHARACTERISTICS PER BUILDING TYPE

b_data = {}

for use in set(agents_info['bldg_type'].values):

    b_data[use] = {}

    n_b = np.sum(agents_info['bldg_type'].values==use)

    min_pv = np.min(agents_info["pv_size_kw"].loc[agents_info['bldg_type'].values==use].values)
    max_pv = np.max(agents_info["pv_size_kw"].loc[agents_info['bldg_type'].values==use].values)
    av_pv = np.average(agents_info["pv_size_kw"].loc[agents_info['bldg_type'].values==use].values)

    b_ids = list(agents_info.loc[agents_info['bldg_type'].values==use].index)

    min_d = np.nanmin(np.sum(demand[b_ids].values,axis=0))
    max_d = np.nanmax(np.sum(demand[b_ids].values,axis=0))
    av_d = np.nanmean(np.sum(demand[b_ids].values,axis=0))

    min_s = np.nanmin(np.sum(solar[b_ids].values,axis=0))
    max_s = np.nanmax(np.sum(solar[b_ids].values,axis=0))
    av_s = np.nanmean(np.sum(solar[b_ids].values,axis=0))

    b_data[use]["Number of buildings"] = n_b
    b_data[use]["Min. demand"] = min_d
    b_data[use]["Max. demand"] = max_d
    b_data[use]["Av. demand"] = av_d
    b_data[use]["Min. PV output"] = min_s
    b_data[use]["Max. PV output"] = max_s
    b_data[use]["Av. PV output"] = av_s

b_df = pd.DataFrame(b_data).T

b_df.to_csv(files_dir +"\\building_type_data.csv", sep=";")

#%% PLOT HISTOGRAMS

fig_hist, axes_hist = plt.subplots(12,2,figsize=(6.5,12), sharex=True)

for use in set(agents_info['bldg_type'].values):

    row = list(set(agents_info['bldg_type'].values)).index(use)

    b_ids = list(agents_info.loc[agents_info['bldg_type'].values==use].index)

    axes_hist[row,0].hist(np.sum(demand[b_ids].values,axis=0))
    axes_hist[row,1].hist(np.sum(solar[b_ids].values,axis=0))

#%% PLOT WEEKLY LOAD PROFILES

typeb_d = {k: list(agents_info.loc[agents_info["bldg_type"].values==k].index) for k in set(agents_info["bldg_type"].values)}

freq = "day"
f = 1
if freq == "day":
    f = 24
elif freq == "week":
    f = 24*7
          
#for typeb in typeb_d.keys():
for typeb in ["INDUSTRIAL"]:

    fig, ax = plt.subplots(1,1, figsize=(6.5,4))

    #y = np.sum(demand[typeb_d[typeb]].values, axis=1)

    #ax.plot([np.sum(y[x*f:x*f+f]) for x in range(1+int(len(y)/f))], color="red", linewidth=0.5)

    for b in typeb_d[typeb]:

        y = [np.sum(demand[b][7*24*x:7*24*(x+1)])/np.sum(demand[b]) for x in range(53)]

        ax.plot(y, alpha=0.5)

    ax.set_title(typeb)
    ax.set_ylabel("Demand [kWh of week / kWh year]")
    ax.set_xlabel("Week of the year")

    #ax.set_xlim(0,365)
    #ax.set_ylim(0,np.max(y)*1.1)

    #fig.savefig(files_dir +"\\fig_load_"+typeb+".png", format="png", bbox_inches="tight", dpi=210)

#%% PLOT ANNUAL LOAD AND GENERATION PROFILES

typeb_d = {k: list(agents_info.loc[agents_info["bldg_type"].values==k].index) for k in set(agents_info["bldg_type"].values)}

# PLOT STARTS HERE

week_summer = 36
week_winter = 2

buildings = [typeb_d[t][2] for t in ["MULTI_RES", "RETAIL","INDUSTRIAL","OFFICE"]]

# Create a figure with two columns (summer, winter)
fig_load_example, axes_le = plt.subplots(4,2, figsize=(10,12.313), sharex=True)

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

        dem = demand[b].values[start:end]

        axes_le[buildings.index(b),col].fill_between(range(end-start), np.zeros(end-start), dem, color="cornflowerblue")
        axes_le[buildings.index(b),col].plot(dem, color="midnightblue", linewidth=0.5)

        axes_le[buildings.index(b),col].set_xlim(0,end-start)
        axes_le[buildings.index(b),col].set_ylim(0,max(dem)*1.1)

        axes_le[buildings.index(b),1].set_yticks([])
        axes_le[buildings.index(b),1].set_yticklabels([])
        
        axes_le[buildings.index(b),col].annotate(agents_info.at[b,"bldg_type"], xy=(0.98,0.93), xycoords="axes fraction", fontsize=8, ha="right")

        axes_le[buildings.index(b),0].set_ylabel("Power demand [kW]")

axes_le[3,0].set_xlabel("Hour of the week")
axes_le[3,1].set_xlabel("Hour of the week")

axes_le[0,0].set_title("Example winter week (w=2)")
axes_le[0,1].set_title("Example summer week (w=36)")

axes_le[0,0].yaxis.set_minor_locator(AutoMinorLocator())
axes_le[0,0].xaxis.set_minor_locator(AutoMinorLocator())

for col in range(2):
    for row in range(3):
        axes_le[row,col].spines['top'].set_visible(False)
        axes_le[row,col].spines['bottom'].set_visible(False)
    axes_le[3,col].spines['top'].set_visible(False)

# add custom legend
from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color="midnightblue", lw=1, linestyle="--"),
                Line2D([0], [0], color="cornflowerblue", lw=4)
                ]

axes_le[3,0].legend(custom_lines, ['Load profile', 'Grid demand'], ncol=2, bbox_to_anchor=(1.4,-0.2), frameon=False)

#%% EXPORT FIGURE
fig_load_example.savefig(files_dir +"\\fig_building_load.svg", format="svg")
fig_load_example.savefig(files_dir +"\\fig_building_load.png", format="png", bbox_inches="tight", dpi=210)
