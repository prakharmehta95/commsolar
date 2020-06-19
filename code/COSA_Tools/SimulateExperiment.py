
import pandas as pd
import numpy as np
import itertools as it
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
    for inputs_file in glob.glob('*_COSA.json'):

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
        agents_info = attributes of each building (df, column = variable, 
            index = building_ids)
        distances = distances between buildings (df, columns have building_id
            and dist_building_id where building_id column is a list of buildings
            and dist_building_id the distance between the building_id heading 
            the column and the one in the corresponding row)
        solar = dataframe of hourly solar generation per building (df, 
            columns = building_id, rows = 8760 hourly power demand)
        demand = dataframe of hourly electricity demand per building (df,
            columns = building id, rows = 8760 hourly power demand)
    """
    print("Importing data")

    # Define path to data files
    data_path = os.path.join(files_dir,"COSA_Data")

    # Define file name for data inputs
    agents_info_file = "buildings_info_test.csv"
    distances_data_file = "distances_data.csv"
    solar_data_file = "CEA_Disaggregated_SolarPV_3Dec.pickle"
    demand_data_file = "CEA_Disaggregated_TOTAL_FINAL_06MAR.pickle"

    # Import data about buildings (1 building = 1 agent)
    agents_info_df = pd.read_csv(os.path.join(data_path,agents_info_file))

    # Make agents ID the index
    agents_info_df = agents_info_df.set_index('bldg_name', drop = False)

    # Make it a dictionary for accessing it faster
    agents_info = agents_info_df.to_dict('index')

    # Import data of distances between all buildings
    distances = pd.read_csv(os.path.join(data_path, distances_data_file))

    # Import data of solar irradiation resource
    solar = pd.read_pickle(os.path.join(data_path, solar_data_file))

    # Import data of electricity demand profiles
    demand = pd.read_pickle(os.path.join(data_path, demand_data_file))

    return agents_info, distances, solar, demand

def run_experiment(sc_inputs, BuildingAgent, SolarAdoptionModel, 
        sc_ind_npvs, agents_info, distances, solar, demand):
    """
    This function runs one experiment of the model.

    Inputs:
        sc_inputs = simulation and scenario inputs (list of dict)
        BuildingAgent = object class for agents in model (class)
        SolarAdoptionModel = object class for model (class)
        sc_ind_npvs = individual economic evaluation variables (list of dict of dfs)
        agents_info = attributes of each building (df, column = variable, 
            index = building_ids)
        distances = distances between buildings (df, columns have building_id
            and dist_building_id where building_id column is a list of buildings
            and dist_building_id the distance between the building_id heading 
            the column and the one in the corresponding row)
        solar = dataframe of hourly solar generation per building (df, 
            columns = building_id, rows = 8760 hourly power demand)
        demand = dataframe of hourly electricity demand per building (df,
            columns = building id, rows = 8760 hourly power demand)
    Returns:
        exp_results = list of dictionaries with results per run (list)
    """

    # Define the number of runs for this experiment
    runs = sc_inputs[0]["simulation_parameters"]["runs"]

    # Define the number of cores
    n_cores = sc_inputs[0]["simulation_parameters"]["n_cores"]

    # Pack inputs into a dictionary
    in_dicts = []
    for ind_npv_outputs in sc_ind_npvs:
        in_dict = {
            "BuildingAgent": BuildingAgent, 
            "SolarAdoptionModel":SolarAdoptionModel,
            "ind_npv_outputs":ind_npv_outputs, 
            "agents_info":agents_info, 
            "distances":distances, 
            "solar":solar, 
            "demand":demand
            }
        in_dicts.append(in_dict)
   
    # Create run inputs
    run_inputs = []
    for in_dict in in_dicts:
        for sc_dict in sc_inputs:
            for run in range(runs):
                run_inputs.append([run, in_dict, sc_dict])

    # Create an empty list to store the results from simulations
    exp_results = []

    # Use single core computation
    if n_cores == 1:

        # Loop through runs
        for run_input in run_inputs:
            
            # simulate the run
            runs_dict = simulate_run(*run_input)

            # store the results in the experiment list
            exp_results.append(runs_dict)

    # Use multicore pool computing
    elif n_cores > 1:

        # Run experiment with multiple cores
        with Pool(n_cores) as p:
            exp_results = p.starmap(simulate_run, run_inputs)

            # Wait all processes to finish
            p.join

    return exp_results  

def simulate_run(run, in_dict, sc_dict):
    """
    This function runs a single run of an experiment.

    Input:
        run = identifier of simulation run (int)
        in_dict = dictionary containing inputs required by model (dict)
        sc_dict = dictionary with inputs for this scenario (dict)
    Returns:
        run_out_dict = results for one simulation run (dict)
    """

    # Define random seed
    randomseed = sc_dict["randomseed"][run]
    SolarAdoptionModel = in_dict["SolarAdoptionModel"]

    # Create Small World Network
    AgentsNetwork = make_swn(
                        in_dict["distances"], 
                        list(in_dict["agents_info"].keys()), 
                        sc_dict["simulation_parameters"]["n_peers"], 
                        randomseed)

    # Create one instantiation of the model
    sim_model = SolarAdoptionModel(
                            in_dict["BuildingAgent"], 
                            sc_dict, 
                            in_dict["ind_npv_outputs"], 
                            AgentsNetwork, 
                            in_dict["agents_info"], 
                            in_dict["distances"], 
                            in_dict["solar"], 
                            in_dict["demand"],
                            seed=randomseed)

    # Loop through the number of years to simulate
    for yr in range(sc_dict["simulation_parameters"]["years"]):
        
        # Advance the model one step
        sim_model.step()
    
    # Collect agent and model variables of the run
    run_out_dict = {
        "agent_vars": sim_model.datacollector.get_agent_vars_dataframe(),
        "model_vars": sim_model.datacollector.get_model_vars_dataframe(),
        "com_formed": sim_model.datacollector.get_table_dataframe("communities"),
    }

    # Create labels for the scenario sim, cal, and eco parameters
    sim_label = "_".join([str(x) for x in sc_dict["simulation_parameters"].values()])
    cal_label = "_".join([str(x) for x in sc_dict["calibration_parameters"].values()])
    # Avoid the three last values that correspond to "smart_meter_prices", "PV_price baseline" and "hour_price"
    eco_pars = list(sc_dict["economic_parameters"].values())[:-3]
    eco_label = "_".join([str(x) for x in eco_pars])

    # Add run and scenario info
    for df in run_out_dict.values():
        df["run"] = run
        df["sim_label"] = sim_label
        df["cal_label"] = cal_label
        df["eco_label"] = eco_label

    return run_out_dict

def initialize_scenario_inputs(inputs):
    """
    This function creates a list of unique input dictionaries for each scenario.
    
    Inputs:
        inputs = inputs for experiment as taken from JSON file (dict)
    Returns:
        sc_dict_list = unique input dictionaries per scenario (list of dicts)
    
    Important - there is a list of forbidden inputs that require different 
    experiments: "smart_meter_prices", "PV_price_baseline", "hour_price" all
    in "economic_parameters".
    """
    # List of parameters that cannot be changed within one experiment
    forbidden_list = ["smart_meter_prices", "PV_price_baseline", "hour_price"]

    # Unpack parameters by type
    sim_pars = inputs["simulation_parameters"]
    cal_pars = inputs["calibration_parameters"]
    eco_pars = {k:v for k,v in inputs["economic_parameters"].items()}

    # Remove keys not allowed to be in combinations
    for f_key in forbidden_list:
        del eco_pars[f_key]

    # Create list of parameter dictionaries
    pars_d_list = [sim_pars, cal_pars, eco_pars]

    # Create a list with three items (one per parameter dictionary type) that
    # contains a list of tuples, each one with the values for a unique combination
    combo_list = [list(it.product(*list(pars.values()))) for pars in pars_d_list]

    # Create a list containing three lists (one per type of parameters), in which
    # there is a list of unique dictionaries for each type of parameter
    par_type_unique_dict_list = []

    # Loop through the three types of original dictionaries of parameters
    for par in range(len(pars_d_list)):

        # Create a list of unique dictionaries per parameter type
        list_dicts = [
            # Create a unique dictionary
            {key: combo_list[par][n_d][list(pars_d_list[par].keys()).index(key)] 
            # by looping through the keys for this type of parameters
            for key in pars_d_list[par].keys()} 
            # for as many combinations as required
            for n_d in range(len(combo_list[par]))]
        
        par_type_unique_dict_list.append(list_dicts)

    # Define the number of economic scenarios
    n_econ_sc = len(par_type_unique_dict_list[2])

    # Create the output list containing a complete input dictionary per unique
    # combination with all three different parameter types
    sc_dict_list = []
    for s_d in par_type_unique_dict_list[0]:

        for c_d in par_type_unique_dict_list[1]:
            for e_d in par_type_unique_dict_list[2]:

                sc_d = {}
                sc_d["simulation_parameters"] = s_d
                sc_d["calibration_parameters"] = c_d          
                sc_d["economic_parameters"] = e_d
                
                # Add the variables in the forbidden list
                for key in forbidden_list:
                    sc_d["economic_parameters"][key] = inputs["economic_parameters"][key]

                sc_dict_list.append(sc_d)

    # Include inputs keys out of three parameter types
    out_inputs = ["exp_name", "randomseed"]
    for oi in out_inputs:
        for d in sc_dict_list:
            d[oi] = inputs[oi]
            
    return sc_dict_list, n_econ_sc

def save_results(exp_name, exp_results, files_dir, timestamp):
    """
    This function exports the simulation outputs of one experiment.
    
    Inputs
        exp_name = name of experiment (str)
        exp_results = list of out_dict dictionarys (list)
        files_dir = directory of current file (str)
        timestamp = start time of code execution (str)
    
    Returns
        None (it directly saves the exported csv files in the directory)
    """

    # Define output directory
    out_dir = os.path.join(files_dir,"COSA_Outputs")

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
            out_label = '{0}_{1}_{2}_.csv'.format(timestamp, exp_name, out_name)
            
            # Save the output files into csv documents
            out_data.to_csv(os.path.join(out_dir,out_label), mode='w', sep=';')