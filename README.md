# == Community solar adoption - Agent-based model (COSA - ABM) ==

The code in this repertory uses the mesa python library for the formulation of
an agent-based model of community solar adoption.

main.py -> principal program: initializes the data required for running the
            COSA-ABM, reads the inputs for the simulations to be carried, and
            creates an instantiation of the model that then simulates and
            collects data from.

            TO-DO: allow for parallel computing to enable cluster-based runs

COSA_Model -> contains the code for the SolarAdoptionModel object class

COSA_Agent -> contains the code for the BuildingAgent object class

COSA_Tools -> contains auxiliary code for economic evaluation of installation
            of solar PV, data collection, a modified scheduler that allows a
            coherent randomisation control, and a creator of social networks

COSA_Data -> contains the input data required for the simulations
            **Note: large input files are ignored and not uploaded to the repo

COSA_Outputs -> stores the simulation outputs (all content ignored because of
            the large size of the files)
