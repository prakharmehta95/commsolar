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

def run_experiment(inputs, BuildingAgent, SolarAdoptionModel, 
        ind_npv_outputs, agents_info, distances, solar, demand):

    runs = inputs["simulation_parameters"]["runs"]
    n_cores = inputs["simulation_parameters"]["n_cores"]

    in_dict = {
        "BuildingAgent": BuildingAgent, 
        "SolarAdoptionModel":SolarAdoptionModel, 
        "inputs":inputs,
        "ind_npv_outputs":ind_npv_outputs, 
        "agents_info":agents_info, 
        "distances":distances, 
        "solar":solar, 
        "demand":demand
        }
    
    # Create run inputs
    run_inputs = [[run, in_dict] for run in range(runs)]

    # Create an empty list to store the results from simulations
    exp_results = []

    if n_cores == 1:

        #main loop for the ABM simulation
        for run in range(runs):

            print("Simulation run = ",run)
            print(strftime("%H:%M:%S", gmtime()))
            
            # simulate the run
            runs_dict = simulate_run(run, run_inputs[run][1])

            # store the results in the experiment list
            exp_results.append(runs_dict)

    elif n_cores > 1:

        # Run experiment with multiple cores
        with Pool(n_cores) as p:
            exp_results = p.starmap(simulate_run, run_inputs)

            # Wait all processes to finish
            p.join

    return exp_results  

def simulate_run(run, in_dict):

    # Define random seed
    randomseed = in_dict["inputs"]["randomseed"][run]
    SolarAdoptionModel = in_dict["SolarAdoptionModel"]

    # Create Small World Network
    AgentsNetwork = make_swn(
                        in_dict["distances"], 
                        in_dict["agents_info"].bldg_name, 
                        in_dict["inputs"]["simulation_parameters"]["n_peers"], 
                        randomseed)

    # Create one instantiation of the model
    sim_model = SolarAdoptionModel(
                            in_dict["BuildingAgent"], 
                            in_dict["inputs"], 
                            randomseed, 
                            in_dict["ind_npv_outputs"], 
                            AgentsNetwork, 
                            in_dict["agents_info"], 
                            in_dict["distances"], 
                            in_dict["solar"], 
                            in_dict["demand"])      

    # Loop through the number of years to simulate
    for yr in range(in_dict["inputs"]["simulation_parameters"]["years"]):
        
        # Advance the model one step
        sim_model.step()
    
    # Collect agent and model variables of the run
    run_out_dict = {
        "agent_vars": sim_model.datacollector.get_agent_vars_dataframe(),
        "model_vars": sim_model.datacollector.get_model_vars_dataframe(),
        "com_formed": sim_model.datacollector.get_table_dataframe("communities"),
    }

    return run_out_dict

def save_results(exp_results, files_dir, start):
    """
    This function exports the simulation outputs of one experiment.
    
    Inputs
        exp_results = list of out_dict dictionarys (list)
        files_dir = directory of current file (str)
        start = start time of code execution (str)
    
    Returns
        None (it directly saves the exported csv files in the directory)
    """

    # Define output directory
    out_dir = files_dir + "\\COSA_Outputs\\"

    # Create an empty dictionary to compile results from individual runs
    compiler_out_dict = {}

    # Loop through all the runs in the experiment
    for run_dict in exp_results:

        # Extract the results of each run
        for key, val in run_dict.items():
            
            # If this is the first run, create a dataframe 
            if len(compiler_out_dict) < len(run_dict.keys()):
                compiler_out_dict[key] = val

            # Else, just concatenate the dataframes
            else:
                compiler_out_dict[key] = compiler_out_dict[key].append(val)                

    # Loop through all the data to export
    for out_name, out_data in compiler_out_dict.items():

            # Name the output files
            out_file_label = '{0}_{1}_.csv'.format(start, out_name)
            
            # Save the output files into csv documents
            out_data.to_csv(out_dir+out_file_label, mode='w', sep=';')