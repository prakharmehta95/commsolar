# Community solar adoption agent-based model (COSA-ABM)

The code in this repertory uses the mesa python library for the formulation of
an agent-based model of community solar adoption.

TO-DO:  (see file "updates_before_submission")

Terms:  - experiment = simulation and scenarios inputs in JSON file.
        - scenario = unique combination of simulation, calibration, and 
        economic parameters. One experiment can contain one or several scenarios.
        - batch = set of simulation runs computed for one scenario..
        - run = deterministic simulation of the model for one scenario.
        - timestep = one year in the simulation model.

Important: results are reported at experiment level, after computing one batch
for each scenario in the experiment.

main.py -> principal program: initializes the data required for running the
            COSA-ABM, reads the inputs for the simulations to be carried, and
            creates an instantiation of the model that then simulates and
            collects data from.

COSA_Model -> contains the code for the SolarAdoptionModel object class

COSA_Agent -> contains the code for the BuildingAgent object class

COSA_Tools -> contains auxiliary code for economic evaluation of installation
            of solar PV, data collection, a modified scheduler that allows a
            coherent randomisation control, and a creator of social networks

COSA_Data -> contains the input data required for the simulations
            **Note: large input files are ignored and not uploaded to the repo

COSA_Outputs -> stores the simulation outputs (all content ignored because of
            the large size of the files)
