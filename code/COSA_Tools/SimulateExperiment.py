import pandas as pd
import numpy as np
import sys, os, datetime, re, json, glob

from time import gmtime, strftime
from multiprocessing import Pool

from COSA_Tools.swn import make_swn

def import_parameters(files_dir):
    """
    This function imports all the experiments to be simulated in the form
    of JSON files in the current directory.

    Inputs
        files_dir = directory of current file (str)
    Return
        experiment_inputs = list of dictionaries with simulation and scenario
            parameters (list)
    """

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
    
    return experiment_inputs

def import_data(files_dir):
    """ 
    This function reads the data from the COSA_Data directory and stores it
    in the corresponding variables

    Inputs
        files_dir = directory of current file (str)
    Returns
        agents_info
        distances
        solar
        demand
        TO-DO ADD DESCRIPTIONS
    """

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

    return agents_info, distances, solar, demand

def create_scenario_inputs(inputs):
    """
    This function takes the inputs to the experiment and creates a list of
    dictionaries of inputs, each containing a unique scenario (i.e. a unique
    combination of input parameters).

    Inputs
        inputs = set of parameters for the experiment (dict)
    Outputs
        scenario_inputs = list of sets of parameters per scenario (list)
    """


def simulate_experiment_one_core(agent, model,
    inputs, ind_npv_outputs, agents_info, distances, solar, demand):
    
    #empty dictionaries to store results
    results_agent = {}
    results_model = {}
    communities = {}

    #main loop for the ABM simulation
    for run in range(inputs["simulation_parameters"]["runs"]):
        print("Simulation run = ",run)
        print(strftime("%H:%M:%S", gmtime()))

        # Define random seed
        randomseed = inputs["randomseed"][run]

        # Create Small World Network
        print(strftime("%H:%M:%S", gmtime()))
        AgentsNetwork = make_swn(distances, agents_info.bldg_name, 
            inputs["simulation_parameters"]["n_peers"], inputs["randomseed"][run])

        # Create one instantiation of the model
        sim_model = model(agent, inputs, randomseed, ind_npv_outputs, 
                        AgentsNetwork, agents_info, distances, solar, demand)      

        # Loop through the number of years to simulate
        for sim_year in range(inputs["simulation_parameters"]["years"]):

            # Print current year of simulation
            print("YEAR:",sim_year)
            print(strftime("%H:%M:%S", gmtime()))
           
            # Advance the model one step
            sim_model.step()
        
        # Collect agent and model variables of the run
        agent_vars = sim_model.datacollector.get_agent_vars_dataframe()
        model_vars = sim_model.datacollector.get_model_vars_dataframe()
        com_formed = sim_model.datacollector.get_table_dataframe("communities")
    
        #stores results across multiple runs
        results_agent["run_" + str(run)] = agent_vars
        results_model["run_" + str(run)] = model_vars
        communities["run_" + str(run)] = com_formed
    
    # Define a dictionary of names and data to export
    out_dict = {
        "results_agent": results_agent,
        "communities": communities,
        "results_model": results_model
        }
    return out_dict

def simulate_experiment_multicore():
    """
    TO-DO
    """

def save_results(out_dicts, files_dir, start):
    """
    This function exports the dictionaries in out_dict to csv files in 
    COSA_Outputs directory.
    
    Inputs
        out_dicts = list of out_dict dictionarys (list)
        files_dir = directory of current file (str)
        start = start time of code execution (str)
    
    Returns
        None (it directly saves the exported csv files in the directory)
    """

    # Define output directory
    out_dir = files_dir + "\\COSA_Outputs\\"

    # If we have a list of out_dict dictionaries, loop through them
    if isinstance(out_dicts, list):

        compiler_out_dict = {"result_agents":0,"communities":0,"results_model":0}

        """
        TO-DO
        """
        for out_dict in out_dicts:
            for out_d_key, out_d_val in out_dict:
                compiler_out_dict[out_d_key] = compiler_out_dict[out_d_key].append(out_d_val)

        # concatenate by dict type
    else:
        # If we have only one out_dict
        compiler_out_dict = out_dicts

    # Loop through all the data to export
    for out_name, out_data in compiler_out_dict.items():

            # Name the output files
            out_file_label = '{0}_{1}_.csv'.format(start, out_name)
            
            # Transform dict data into dataframe
            out_data_df = pd.concat(out_data)
            
            # Save the output files into csv documents
            out_data_df.to_csv(out_dir+out_file_label, mode='w', sep=';')