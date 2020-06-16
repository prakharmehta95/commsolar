# -*- coding: utf-8 -*-
"""
Current version: June, 2020
@author: Prakhar Mehta, Alejandro Nuñez-Jimenez
"""
#%% IMPORT PACKAGES AND SCRIPTS

# Import python packages
import sys, os, re, json, glob, time, pickle, datetime
import pandas as pd
import numpy as np

from multiprocessing import Pool
from time import gmtime, strftime

# Import functions from own scripts
from COSA_Tools.npv_ind import calculate_ind_npv
from COSA_Tools.swn import make_swn
from COSA_Tools.SimulateExperiment import (import_parameters, import_data,
    save_results, run_experiment)

# Import object classes for model and agents
from COSA_Model.SolarAdoptionModel import SolarAdoptionModel
from COSA_Agent.BuildingAgent import BuildingAgent

#%% IMPORT SIMULATION PARAMETERS

# Record time of start of the program
start = time.time()

# Read current directory
files_dir = os.path.dirname(os.path.abspath(__file__))

# Add current file's directory to path
sys.path.append(files_dir)

# Import simulation parameters
experiment_inputs = import_parameters(files_dir)

# Import input data
agents_info, distances, solar, demand = import_data(files_dir)

#%% SIMULATE EXPERIMENTS

if __name__ == '__main__':
        
    # Loop through experiments
    exp = 0
    for inputs in experiment_inputs:

        # Print what experiment is currently running and how many are in the list
        print("= Run exp "+str(exp+1)+" of "+str(len(experiment_inputs))+" =")

        # Read simulation parameters
        sim_pars = inputs["simulation_parameters"]
        
        # Calculate individual NPVs
        print(strftime("%H:%M:%S", gmtime()))
        ind_npv_outputs = calculate_ind_npv(inputs, agents_info, solar, demand)

        # Simulate experiment
        exp_results = run_experiment(inputs, BuildingAgent, SolarAdoptionModel, 
            ind_npv_outputs, agents_info, distances, solar, demand)
        
        print(strftime("%H:%M:%S", gmtime()))
        print("save_results")
        # Export results
        save_results(exp_results, files_dir, start)
        print(strftime("%H:%M:%S", gmtime()))

    # Read end time
    end = time.time()

    # Print elapsed computation time to screen
    print("Code Execution Time = ",end - start)
    print("==FIN==")
    print(strftime("%H:%M:%S", gmtime()))