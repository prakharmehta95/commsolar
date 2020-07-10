#%% IMPORT REQUIRED PACKAGES

import sys, os, glob, json

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
input_subfolder = 'code\\COSA_Data\\'
input_dir = files_dir[:files_dir.rfind('code')] + input_subfolder

# Import error from electricity price simulation
errors_df = pd.read_csv(input_dir+"error_wmep.csv", dtype=float, na_values="#VALUE!")

#%% PLOT HISTOGRAMS
colors = [(91/255,155/255,213/255),(237/255, 125/255, 49/255),(165/255,165/255,165/255), (255/255, 192/255, 0), (68/255,114/255,196/255)]

fig, ax = plt.subplots(1,1,figsize=(6.5,5))

for y in range(5):
    data = errors_df.T.values[y]
    ax.hist(data.flatten(),bins=100, histtype="step", label=str(2015+y),color=colors[y])

ax.set_xlabel("Deviation simulated to historical price [€/MWh]")
ax.set_ylabel("Absolute frequency [number of hours in a year]")
ax.legend(loc="upper right")

#%% EXPORT FIGURE
fig.savefig(files_dir +"\\error_sim_el_price.svg", format="svg")
fig.savefig(files_dir +"\\error_sim_el_price.png", format="png", bbox_inches="tight", dpi=210)