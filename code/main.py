# -*- coding: utf-8 -*-
"""
Current version: June, 2020
@authors: Prakhar Mehta, Alejandro Nuñez-Jimenez
"""
#%% IMPORT PACKAGES AND SCRIPTS

# Import python packages
import sys, os, re, json, glob, time, pickle, datetime, feather
import pandas as pd
import numpy as np
import itertools as it

from multiprocessing import Pool
from time import gmtime, strftime

# Import functions from own scripts
from COSA_Tools.npv_ind import calculate_ind_npv
from COSA_Tools.swn import make_swn
from COSA_Tools.SimulateExperiment import (import_parameters, import_data,
    save_results, run_experiment, initialize_scenario_inputs)

# Import object classes for model and agents
from COSA_Model.SolarAdoptionModel import SolarAdoptionModel
from COSA_Agent.BuildingAgent import BuildingAgent

# Record time of start of the program
start = time.time()

#%% SIMULATE EXPERIMENTS

if __name__ == '__main__':

    # Read current directory
    files_dir = os.path.dirname(os.path.abspath(__file__))

    # Add current file's directory to path
    sys.path.append(files_dir)

    # Identifiy the time when the simulation was carried out -> timestamp
    timestamp_format = "%Y-%m-%d-%H-%M-%S-%f"
    timestamp = datetime.datetime.now().strftime(timestamp_format)

    # Import simulation parameters
    experiment_inputs = import_parameters(files_dir)

    # Import input data
    agents_info, distances, solar, demand = import_data(files_dir)
    
    # Loop through experiments
    exp = 0
    for inputs in experiment_inputs:

        # Print what experiment is running and how many are in the list
        print("= Run exp "+str(exp+1)+" of "+str(len(experiment_inputs))+" =")
        print(strftime("%H:%M:%S", gmtime()))

        # Initialize the scenario inputs
        sc_inputs, n_econ_scenarios = initialize_scenario_inputs(inputs)

        # Compute the individual economic evaluations per scenario
        sc_ind_npvs = []
        if n_econ_scenarios == 1:  

            # Calculate individual NPVs
            ind_npv_outputs = calculate_ind_npv(sc_inputs[0], agents_info, solar, demand)

            # Store it in list
            sc_ind_npvs.append(ind_npv_outputs)

        else:
            
            # Loop through the scenarios
            for sc_input in sc_inputs:

                # Calculate individual NPVs
                ind_npv_outputs = calculate_ind_npv(sc_input, agents_info, solar, demand)

                # Store it in list
                sc_ind_npvs.append(ind_npv_outputs)
        
        print("Finished individual NPVs")
        print(strftime("%H:%M:%S", gmtime()))

        # Run experiment
        exp_results = run_experiment(sc_inputs, BuildingAgent,         SolarAdoptionModel, sc_ind_npvs, agents_info, distances, solar, demand)
        
        print(strftime("%H:%M:%S", gmtime()))
        print("save_results")
        # Export results
        save_results(sc_inputs[0]["exp_name"], exp_results, files_dir, timestamp)

    # Read end time
    end = time.time()

    # Print elapsed computation time to screen
    print("Code Execution Time = ",end - start)
    print("==FIN==")