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
from time import gmtime, strftime                                                                             

from random import seed

# Import functions from own scripts
from COSA_Tools.npv_ind import calculate_ind_npv
from COSA_Tools.swn import make_swn

# Import object classes for model and agents
from COSA_Model.SolarAdoptionModel import SolarAdoptionModel
from COSA_Agent.BuildingAgent import BuildingAgent

# Record time of start of the program
start = time.time()

#%% IMPORT SIMULATION PARAMETERS

# Read current directory
files_dir = os.path.dirname(os.path.abspath(__file__))

# Add current file's directory to path
sys.path.append(files_dir)

# Create a list with all the experiments to be simulated
# each .JSON file will become one item of this list
experiment_inputs = []

# Read all the JSON files in current directory
# Note that changing the ending of the JSON file we can import experiment
# input files for different purposes (e.g., "_cal.json" for calibration)
for inputs_file in glob.glob('*_cal.json'):

    # Save their content as input values for different experiments
    with open(inputs_file, "r") as myinputs:
        experiment_inputs.append(json.loads(myinputs.read()))

#%% IMPORT SIMULATION DATA
print("Importing data")

# Define path to data files
data_path = files_dir + "\\COSA_Data\\"

# Define file name for data inputs
agents_info_file = "buildings_info_test.csv"
distances_data_file = "distances_data.csv"
solar_data_file = "CEA_Disaggregated_SolarPV_3Dec.pickle"
demand_data_file = "CEA_Disaggregated_TOTAL_FINAL_06MAR.pickle"

# Import data about buildings (1 building = 1 agent)
agents_info = pd.read_csv(data_path + agents_info_file)

# Set bldg_name as the index
agents_info = agents_info.set_index('bldg_name', drop = False)

# Import data of distances between all buildings
distances = pd.read_csv(data_path + distances_data_file)

# Import data of solar irradiation resource
solar = pd.read_pickle(data_path + solar_data_file)
# IMPORTANT THIS NEEDS TO BE CONVERTED TO AC

# Import data of electricity demand profiles
demand = pd.read_pickle(data_path + demand_data_file)

#%% SIMULATE EXPERIMENTS

# Loop through experiments
exp = 0
for inputs in experiment_inputs:

    # Print what experiment is currently running and how many are in the list
    print("= Run exp "+str(exp+1)+" of "+str(len(experiment_inputs))+" =")

    # Read simulation parameters
    sim_pars = inputs["simulation_parameters"]
    print(strftime("%H:%M:%S", gmtime()))
    ## Calculate individual NPVs
    ind_npv_outputs = calculate_ind_npv(inputs, agents_info, solar, demand)
    print(strftime("%H:%M:%S", gmtime()))

    ## Create Small World Network
    AgentsNetwork = make_swn(distances, agents_info.bldg_name, 
                                sim_pars["n_peers"], sim_pars["peer_seed"])

    #empty dictionaries to store results
    results_agent = {}
    results_model = {}
    communities = {}

    # Define random seed
    randomseed = sim_pars["randomseed"]

    #main loop for the ABM simulation
    for j in range(sim_pars["runs"]):
        print("Simulation run = ",j)
        print(strftime("%H:%M:%S", gmtime()))

        #642 is just any number to change the seed for every run 
        randomseed = randomseed + j * 642

        # Create one instantiation of the model
        model = SolarAdoptionModel(BuildingAgent, inputs, randomseed,
                                    ind_npv_outputs, AgentsNetwork, agents_info,
                                    distances, solar, demand)

        # seed for attitude changes in a new run
        att_seed = sim_pars["att_seed"] + j * 10          

        # Loop through the number of years to simulate
        for i in range(sim_pars["years"]):

            # Print current year of simulation
            print("YEAR:",i+1)
            print(strftime("%H:%M:%S", gmtime()))

            #f or the environmental attitude which remains constant for
            #  an agent in a particular run
            seed(sim_pars["att_seed"])
            
            # CHECK HOW THIS WORKS BECAUSE NOW IN MODEL CLASS
            Combos_formed_Info = model.step()
            
            # store communities formed
            #lab_2 = "combos_info_{0}_{1}".format(str(j),str(i))
            #communities[lab_2] = pd.DataFrame.copy(Combos_formed_Info)
        
        # Collect agent and model variables
        agent_vars = model.datacollector.get_agent_vars_dataframe()
        model_vars = model.datacollector.get_model_vars_dataframe()
        com_formed = model.datacollector.get_table_dataframe("communities")
    
        #stores results across multiple runs
        results_agent["run_" + str(j)] = agent_vars
        results_model["run_" + str(j)] = model_vars
        communities["run_" + str(j)] = com_formed
    
    print("==FIN==")
    print(strftime("%H:%M:%S", gmtime()))

#%% SAVE SIMULATION RESULTS

# Define output directory
out_dir = files_dir + "\\COSA_Outputs\\"

# Define a dictionary of names and data to export
out_dict = {
    "results_agent": results_agent,
    "communities": communities,
    "results_model": results_model
    }

# Loop through all the data to export
for out_name, out_data in out_dict.items():

        # Name the output files
        out_file_label = '{0}_{1}_.csv'.format(start, out_name)
        
        # Transform dict data into dataframe
        #out_data_df = pd.DataFrame(data=None)
        out_data_df = pd.concat(out_data)
        
        # Save the output files into csv documents
        out_data_df.to_csv(out_dir+out_file_label, mode='w', sep=';')

# Read end time
end = time.time()

# Print elapsed computation time to screen
print("Code Execution Time = ",end - start)