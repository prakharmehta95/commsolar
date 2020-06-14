#%% IMPORT REQUIRED PACKAGES
import pandas as pd
import numpy as np

def calculate_ind_npv(inputs, agents_info, solar, demand):
    '''
    This function computes the economic parameters for the evaluation of
    adoption of each building for each year simulated if they invest in 
    PV individually.

    Inputs
        inputs = simulation parameters (dict)
        agents_info = data about the buildings (df)
        solar = hourly solar generation potential for each building in a year (df)
        demand = hourly electricity demand for each building in a year (df)

    Returns
        ind_npv_outputs = contains a dictioanry of dataframes with results:
            "Agents_NPVs" = NPV per agent per year of simulation (df)
            "Agents_Investment_Costs" = total inv cost per sim year per agent (df) 
            "Agents_PPs_Norm" = normalized pp per sim year per agent (df)
            "Agents_SCRs" = self-consumption rate per agent per operation year
                of PV system in the building (df)       
    '''
    # DEFINE ECONOMIC PARAMETERS
    
    # Read the dictionary of economic parameters from the inputs dictionary
    econ_pars = inputs["economic_parameters"]

    # Set PV lifetime
    PV_lifetime = econ_pars["PV_lifetime"]

    # Defince the degradation rate of PV systems
    deg_rate = econ_pars["PV_degradation"]

    # Define discount rate
    disc_rate = econ_pars["disc_rate"]

    # Define maximum payback period
    max_pp = econ_pars["max_payback_period"]

    # Set PV prices baseline
    PV_price_baseline = econ_pars["PV_price_baseline"]

    # Create PV price projections
    PV_price_projection = {}

    for key, value in PV_price_baseline.items():
        PV_price_projection[key] = np.interp(list(range(1,24)), [1,23],[value,value/2])

    # Define a dictionary of smart meter prices
    # Key = limit of smart meters for price category (string) (e.g., "12")
    # Value = price per smart meter for than number of smart meters (int)
    smp_dict = econ_pars["smart_meter_prices"]

    # Multiply the solar PV data with an efficiency factor to convert to AC
    solar = solar.copy()*econ_pars["AC_conv_eff"] 

    # All hours of the year are "low" except from Mon-Sat from 6-21
    hour_price = econ_pars["hour_price"]

    # Create list of agents
    agent_list_final = agents_info.bldg_name
    agents_info = agents_info.set_index('bldg_name')
    
    # Loop through all agents
    for ag in range(len(agent_list_final)):

        print("B = " + str(ag) + " of " + str(len(agent_list_final)))

        # Set building
        i = agent_list_final[ag]

        ## INVESTMENT COST CALCULATIONS

        ## COST OF SMART METERS

        # Read the number of smart meters for the agent's building
        n_sm = agents_info.at[i,'num_smart_meters']

        # Compute the investment cost of smart meters
        sm_inv = compute_smart_meters_inv(n_sm, smp_dict)

        ## COST OF PV SYSTEM

        # Read the agent's PV size
        pv_size = agents_info.at[i,'pv_size_kw']

        # Read the subsidy the agent can access for the installation
        pv_sub = agents_info.at[i,'pv_subsidy']

        # Compute the investment cost of the PV system for each year simulated
        pv_inv_years = compute_pv_inv_years(pv_size, pv_sub, PV_price_projection)

        ## TOTAL INVESTMENT COST

        # Compute total investment by adding smart meter cost to PV cost
        inv_years = [x + sm_inv for x in pv_inv_years]

        ## CASHFLOW CALCULATIONS

        # Define the buildings hourly electricity demand over one year
        demand_ag = demand[i]

        # Define solar output AC for first year of PV lifetime in building
        solar_building = solar[i] * econ_pars["AC_conv_eff"]

        # Compute annual energy balances during operational life of PV
        lifetime_load_profile = compute_lifetime_load_profile(solar_building,
                                    demand_ag, PV_lifetime, deg_rate, hour_price)

        # Compute annual cashflows during operational life of PV
        lifetime_cashflows = compute_lifetime_cashflows(econ_pars,
                                            lifetime_load_profile, PV_lifetime)

        ## NET-PRESENT VALUE CALCULATIONS

        # Compute net-present value per year simulated
        npv_years = []
        for yr in range(inputs["simulation_parameters"]["years"]):
            npv_years.append(compute_npv(inv_years[yr], lifetime_cashflows, 
                                            disc_rate))

        ## PAYBACK-PERIOD CALCULATIONS

        # Depending on the simulation specifications, use simple or discounted
        # payback period calculation:
        if econ_pars["discount_pp"] == True:

            # Compute discounted payback period per year simulated
            pp_years = compute_discounted_pp(inv_years, lifetime_cashflows,
                                            max_pp, disc_rate)

        else:
            # Compute simple payback period per year simulated
            pp_years = compute_simple_pp(inv_years, lifetime_cashflows, max_pp)

        # Compute normalized payback periods
        pp_years_norm = [1 - (pp / max_pp) for pp in pp_years]
            
        # Store the results for this building/agent
        if ag == 0:
            Agents_NPVs = pd.DataFrame(data=npv_years, columns=[i])
            Agents_SCRs = pd.DataFrame(lifetime_load_profile["SCR"],
                                        columns=[i])
            Agents_Investment_Costs = pd.DataFrame(data=inv_years,
                                        columns=[i])
            Agents_PPs_Norm = pd.DataFrame(data=pp_years_norm, 
                                        columns=[i])
        else:
            Agents_NPVs[i] = npv_years
            Agents_SCRs[i] = lifetime_load_profile["SCR"]
            Agents_Investment_Costs[i] = inv_years
            Agents_PPs_Norm[i] = pp_years_norm

    # Put all outputs in one dictionary
    ind_npv_outputs = {"Agents_NPVs": Agents_NPVs, "Agents_SCRs": Agents_SCRs, 
        "Agents_Investment_Costs": Agents_Investment_Costs, 
        "Agents_PPs_Norm": Agents_PPs_Norm}

    return ind_npv_outputs

def compute_lifetime_load_profile(solar_building, demand_ag, PV_lifetime,
        deg_rate, hour_price):
    """
    Inputs
        solar_outputs = hourly electricity output of PV system in the building
            for first year of its operational lifetime (list of 8760 items)
        demand_ag = hourly electricity demand of the building (list)
        PV_lifetime = years of operational life of installation (integer)
        deg_rate  = degression rate of PV output (float)
        hour_price = price level for each hour of the year (list)
    Returns
        lifetime_load_profile = description of annual energy balances over the 
            operational lifetime of the PV installation of the buildign
            (dataframe with index = year of lifetime, columns = energy balances)
    """
    if deg_rate != 0:

        # Compute the hourly solar output AC for each operational year of PV
        solar_outputs = [solar_building * ((1 - deg_rate) ** y) for y in range(PV_lifetime)]

        for yr in range(PV_lifetime):

            # Create a dataframe with one row per hour of the year and one
            # column per building
            load_profile = pd.DataFrame(data = None, index = range(8760))

            # Create a dictionary to contain the annual energy balances
            load_profile_year = {} 
            
            # Define hourly solar system output for this building and hourly demand
            load_profile["solar"] = solar_outputs[yr]
            load_profile["demand"] = demand_ag

            # Define price of electricity per hour of the day
            load_profile["hour_price"] = hour_price

            # Compute hourly net demand from grid and hourly excess solar
            load_profile["net_demand"] = load_profile.demand - load_profile.solar
            load_profile["excess_solar"] = load_profile.solar - load_profile.demand

            # Remove negative values by making them zero
            load_profile["net_demand"].loc[load_profile["net_demand"] < 0] = 0
            load_profile["excess_solar"].loc[load_profile["excess_solar"] < 0] = 0

            # Compute hourly self-consumed electricity
            # For the hours of the year with solar generation: self-consume all
            # solar generation if less than demand (s) or up to demand (d)
            s = np.array(solar_outputs[yr])
            d = np.array(demand_ag)
            load_profile["sc"] = [min(s[i], d[i]) if s[i] > 0 else 0 for i in range(8760)]
            
            # Compute annual energy balances regardless of hour prices
            for bal in ["solar", "demand", "net_demand", "excess_solar", "sc"]:
                load_profile_year[bal] = sum(load_profile[bal])
            
            # Compute annual energy balances for high and low price hours
            for bal in ["solar", "demand", "excess_solar", "net_demand", "sc"]:
                for pl in ["high", "low"]:
                    cond = (load_profile["hour_price"] == pl)
                    load_profile_year[bal+'_'+pl] = sum(load_profile[bal].loc[cond])

            # Compute year self-consumption rate
            load_profile_year["SCR"] = 0
            if load_profile_year["sc"] > 0:
                load_profile_year["SCR"] = load_profile_year["sc"] / load_profile_year["solar"]

            # Store results in return dataframe
            if yr == 0:
                # If it is the first year, then create the dataframe
                lifetime_load_profile = pd.DataFrame(load_profile_year, index=[0])
            else:
                # Append the dictionary containing the results for this year
                lifetime_load_profile = lifetime_load_profile.append(
                                            load_profile_year, ignore_index=True)
        
    # No degradation
    else:

        # Set solar output as output first year of lifetime
        solar_outputs = solar_building

        # Create a dataframe with one row per hour of the year and one
        # column per building
        load_profile = pd.DataFrame(data = None, index = range(8760))

        # Create a dictionary to contain the annual energy balances
        load_profile_year = {} 
        
        # Define hourly solar system output for this building and hourly demand
        load_profile["solar"] = solar_outputs
        load_profile["demand"] = demand_ag

        # Define price of electricity per hour of the day
        load_profile["hour_price"] = hour_price

        # Compute hourly net demand from grid and hourly excess solar
        load_profile["net_demand"] = load_profile.demand - load_profile.solar
        load_profile["excess_solar"] = load_profile.solar - load_profile.demand

        # Remove negative values by making them zero
        load_profile["net_demand"].loc[load_profile["net_demand"] < 0] = 0
        load_profile["excess_solar"].loc[load_profile["excess_solar"] < 0] = 0

        # Compute hourly self-consumed electricity
        # For the hours of the year with solar generation: self-consume all
        # solar generation if less than demand (s) or up to demand (d)
        s = np.array(solar_outputs)
        d = np.array(demand_ag)
        load_profile["sc"] = [min(s[i], d[i]) if s[i] > 0 else 0 for i in range(8760)]
        
        # Compute annual energy balances regardless of hour prices
        for bal in ["solar", "demand", "net_demand", "excess_solar", "sc"]:
            load_profile_year[bal] = sum(load_profile[bal])
        
        # Compute annual energy balances for high and low price hours
        for bal in ["solar", "demand", "excess_solar", "net_demand", "sc"]:
            for pl in ["high", "low"]:
                cond = (load_profile["hour_price"] == pl)
                load_profile_year[bal+'_'+pl] = sum(load_profile[bal].loc[cond])

        # Compute year self-consumption rate
        load_profile_year["SCR"] = 0
        if load_profile_year["sc"] > 0:
            load_profile_year["SCR"] = load_profile_year["sc"] / load_profile_year["solar"]

        # Store results in return dataframe
        lifetime_load_profile = pd.DataFrame(load_profile_year, index=[0])

        # Make results the same for all lifetime
        lifetime_load_profile = pd.concat([lifetime_load_profile] * PV_lifetime,
            ignore_index=True)

    return lifetime_load_profile

def compute_lifetime_cashflows(econ_pars, lifetime_load_profile, PV_lifetime):
    """
    This function computes the annual cashflows over the operational lifetime
    of the PV system in the building.

    Inputs
        econ_pars = dictionary containint economic parameters (e.g., FIT level)
        lifetime_load_profile = description of annual energy balances over the 
        operational lifetime of the PV installation of the buildign
        (dataframe with index = year of lifetime, columns = energy balances)

    Returns
        lifetime_cashflows = monetary flows into and out of the project for
            each year of its operational lifetime (dataframe, index = yr,
            columns = cashflow category)
    """

    # Define annual demand for building (all years the same so take first)
    demand = lifetime_load_profile["demand"][0]

    # Define electricity prices
    # If different prices apply depending on annual demand, differentiate
    if econ_pars["diff_prices"] == 1:

        # For large consumers
        if demand >= econ_pars["demand_price_threshold"]:
            ewz_high = econ_pars["ewz_high_large"]
            ewz_low = econ_pars["ewz_low_large"]

        # For small consumers
        elif demand < econ_pars["demand_price_threshold"]:
            ewz_high = econ_pars["ewz_high_small"]
            ewz_low = econ_pars["ewz_low_small"]

    # If not, then use the price for small consumers
    elif econ_pars["diff_prices"] == 0:
        ewz_high = econ_pars["ewz_high_small"]
        ewz_low = econ_pars["ewz_low_small"]

    # Create empty dictionary to store annual results
    cashflows_year = {}

    # Loop through the years of operational life of the system
    for yr in range(PV_lifetime):

        # Read year excess solar
        ex_solar_h = lifetime_load_profile["excess_solar_high"][yr]
        ex_solar_l = lifetime_load_profile["excess_solar_high"][yr]
        
        # Compute the revenues from feeding solar electricity to the grid
        fit_h = econ_pars["fit_high"]
        fit_l = econ_pars["fit_low"]
        cashflows_year["FIT"] = ex_solar_h * fit_h + ex_solar_l * fit_l

        # Read avoided consumption from the grid (i.e. self-consumption)
        sc_h = lifetime_load_profile["sc_high"][yr]
        sc_l = lifetime_load_profile["sc_low"][yr]

        # Compute the savings from self-consuming solar electricity
        cashflows_year["savings"] = sc_h * ewz_high + sc_l * ewz_low
        
        # Compute the EWZ Solarsplit costs
        cashflows_year["ewz_split"] = (sc_h + sc_l) * econ_pars["ewz_solarsplit_fee"]

        # Compute O&M costs
        om_cost = econ_pars["OM_Cost_rate"]
        cashflows_year["O&M"] = lifetime_load_profile["solar"][yr] * om_cost

        # Compute net cashflows to the agent
        cashflows_year["net_cf"] = (cashflows_year["FIT"] 
            + cashflows_year["savings"] - cashflows_year["ewz_split"]
            - cashflows_year["O&M"])

        # Store results in return dataframe
        if yr == 0:
            # If it is the first year, then create the dataframe
            lifetime_cashflows = pd.DataFrame(cashflows_year, index=[0])
        else:
            # Append the dictionary containing the results for this year
            lifetime_cashflows = lifetime_cashflows.append(
                                        cashflows_year, ignore_index=True)
   
    return lifetime_cashflows

def compute_smart_meters_inv(n_sm, smp_dict):
    """
    This function takes the number of smart meters and their prices per 
    number in the installation of the building and provides the investment
    cost for the system.

    Inputs
        n_sm = number of smart meters in the building (integer)
        smp_dict = price per smart meter depending on number of sm (dict)
    
    Returns
        sm_inv = investment cost of smart meters (float)
    """
    
    # Convert the keys in the smp_dict into a list of integers that
    # indicate the maximum number of meters to receive that price
    smp_cats = [int(x) for x in list(smp_dict.keys())]

    # Try to find a number of smart meters in the price categories that
    # is larger then the n_sm. If you don't find any, then
    # use the lowest price category (i.e. for more than 50 meters)
    try:
        smp_ix = next(ix for ix, v in enumerate(smp_cats) if v > n_sm)
    except StopIteration:
        smp_ix = len(smp_cats)

    # Estimate investment cost of smart meters
    sm_inv = n_sm * smp_dict[str(smp_cats[smp_ix - 1])]

    return sm_inv

def compute_pv_inv_years(pv_size, pv_sub, PV_price_projection):
    """
    This function calculates the investment cost of the PV system based on 
    its size, and the price for that size category for each year with 
    available price projections.

    Inputs
        pv_size = size of the installation (float)
        pv_sub = subsidy for the installation applicable to the building (float)
        PV_price_projection = projection of PV prices per size category (dict)
            (key = size category, value = list of prices per year)

    Returns
        pv_inv_years = investment cost of PV system per year (list)
    """
    # TO DO 
    # Create version for a single year

    # Convert the keys in PV_price_baseline into a list of integers that
    # indicate the maximum pv_size to receive that price
    pvp_cats = [int(x) for x in list(PV_price_projection.keys())]

    # Try to find a price category that requires a size larger than
    # pv_size. If you don't find any, use lowest price (i.e. >150 kW)
    try:
        pvp_ix = next(ix for ix, v in enumerate(pvp_cats) if v > pv_size)
    except StopIteration:
        pvp_ix = len(pvp_cats)

    # Estimate investment cost of PV system
    prices_size = PV_price_projection[str(pvp_cats[pvp_ix - 1])]
    pv_inv_years = [pv_size * p for p in prices_size]

    # Apply subsidy for installations between 2018 and 2030
    # ASSUMPTION YEAR = 0 = 2018; YEAR = 12 = 2030
    pv_inv_years = [x - pv_sub if pv_inv_years.index(x) < 12 else x for x in pv_inv_years]

    return pv_inv_years

def compute_npv(inv, lifetime_cashflows, disc_rate):
    """
    This function provides the net-present value of installation for one
    building or community for simulation year "yr".

    Inputs
        inv = investment cost of the installation (float)
        lifetime_cashflows = annual cashflows over lifetime of the system (df)
            (index = year of lifetime, column = cashflow)
        disc_rate = time value of money for the agent (float)

    Returns
        npv = NPV for this installation and year of the simulation (float)
    """

    # Start a list with cashflows with negative inv cost
    cf = [- inv]

    # Add the net cashflows for the operational life of the syst to list
    cf.extend(list(lifetime_cashflows["net_cf"].values))

    # Compute NPV if installation occurs this year
    npv = np.npv(disc_rate, cf)

    return npv

def compute_simple_pp(inv_years, lifetime_cashflows, max_pp):
    """
    This function computes the simple payback period (without time-discounted
    values) for each year simulated for this building.
    
    Inputs
        inv_years = investment cost of the installation per year (list)
        lifetime_cashflows = annual cashflows over the lifetime of the system (df)
            (index = year of lifetime, column = cashflow)
        max_pp = maximum length of payback period considered by
            agents (integer)

    Returnsp
        pp_years = simple payback period for each year (list)
    """
    # Initialize list of results
    pp_years = []

    # Put list of investment costs into np.array
    inv = np.array(inv_years)

    # Sum up the cashflows over the lifetime of the installation
    cf = np.nansum(lifetime_cashflows["net_cf"])

    # Divide inv over cf and make max_pp in case cf == 0
    pp_years = [i / cf if cf > 0 else max_pp for i in inv]

    return pp_years

def compute_discounted_pp(inv_years, lifetime_cashflows, max_pp, disc_rate):
    """
    This function computes the simple payback period (without time-discounted
    values) for each year simulated for this building.
    
    Inputs
        inv_years = investment cost of the installation per year (list)
        lifetime_cashflows = annual cashflows over the lifetime of the system (df)
            (index = year of lifetime, column = cashflow)
        max_pp = maximum length of payback period considered by
            agents (integer)
        disc_rate = time value of money for the agent (float)

    Returns
        dpp_years = simple payback period for each year (list)
    """
    # Initialize list of results
    dpp_years = []

    for yr in range(len(inv_years)):

        # Start a list with cashflows for this year with negative inv cost
        dcf = [- inv_years[yr]]

        # Add the net cashflows for the operational life of the syst to list
        dcf.extend(list(lifetime_cashflows["net_cf"].values))

        # Dicount cashflows
        dcf = [dcf[y] / (1 + disc_rate) ** y for y in range(len(dcf))]

        # Cumulate cashflows
        cum_dcf = [dcf[i] + sum(dcf[:i]) for i in range(len(dcf))]

        # Compute discounted cash-flow by looking when cum_dcf changes sign
        try:
            dpp = next(ix for ix, v in enumerate(cum_dcf) if v > 0)

        # If it nevers become positive, assign max_pp
        except:
            dpp = max_pp
    
        # Sotre results
        dpp_years.append(dpp)

    return dpp_years

def calculate_com_npv(inputs, c_dict, year):
    '''
    This function computes the economic parameters for the evaluation of
    the formation of a solar community.

    Inputs
        inputs = simulation parameters (dict)
        c_dict = community parameters (dict)
        year = simulation year (int) (e.g., 0 == 2018)

    Returns
        com_npv_outputs = contains a dictioanry of dataframes with results:
            "Agents_NPVs" = NPV per agent per year of simulation (df)
            "Agents_Investment_Costs" = total inv cost per sim year per agent (df) 
            "Agents_PPs_Norm" = normalized pp per sim year per agent (df)
            "Agents_SCRs" = self-consumption rate per agent per operation year
                of PV system in the building (df)       
    '''

    # DEFINE ECONOMIC PARAMETERS

    # COMMUNITY SPECIFIC

    # Define the newly installed PV capacity
    pv_size = c_dict["pv_size_added"]

    # Define the number of newly installed smart meters
    n_sm = c_dict["n_sm_added"]

    # Define community's generation potential (already AC)
    solar = c_dict["solar"]

    # Define community's aggregated consumption
    demand = c_dict["demand"]

    # NON-COMMUNITY SPECIFIC
    
    # Read the dictionary of economic parameters from the inputs dictionary
    econ_pars = inputs["economic_parameters"]

    # Set PV lifetime
    PV_lifetime = econ_pars["PV_lifetime"]

    # Defince the degradation rate of PV systems
    deg_rate = econ_pars["PV_degradation"]

    # Define discount rate
    disc_rate = econ_pars["disc_rate"]

    # Set PV prices baseline
    PV_price_baseline = econ_pars["PV_price_baseline"]

    # Create PV price projections
    PV_price_projection = {}

    for key, val in PV_price_baseline.items():
        PV_price_projection[key] = np.interp(list(range(1,24)), [1,23],[val,val/2])
    
    # Define a dictionary of smart meter prices
    # Key = limit of smart meters for price category (string) (e.g., "12")
    # Value = price per smart meter for than number of smart meters (int)
    smp_dict = econ_pars["smart_meter_prices"]

    # All hours of the year are "low" except from Mon-Sat from 6-21
    hour_price = econ_pars["hour_price"]

    # Compute the load profile of the installation throughout its lifetime
    lifetime_load_profile = compute_lifetime_load_profile(solar, demand,
                                    PV_lifetime, deg_rate, hour_price)

    # Compute hte cashflows of the installation throughout its lifetime
    lifetime_cashflows = compute_lifetime_cashflows(econ_pars, 
                                    lifetime_load_profile, PV_lifetime)
    
    if pv_size < 30:
        pv_subsidy =  1600 + 460*pv_size
    elif 30 <= pv_size < 100:
        pv_subsidy =  1600 + 340*pv_size
    elif pv_size >= 100:
        pv_subsidy =  1400 + 300*pv_size
    
    # Compute the investment cost of the PV system for each year simulated
    pv_inv_years = compute_pv_inv_years(pv_size, pv_subsidy, PV_price_projection)

    # Select the PV investment cost for the current simulation year
    pv_inv = pv_inv_years[year]
    
    # Compute the investment cost of smart meters
    sm_inv = compute_smart_meters_inv(n_sm, smp_dict)
    
    # Compute the cooperation costs
    coop_cost = 0
    
    # Estimate the total investment cost
    inv = pv_inv + sm_inv + coop_cost

    # Compute the community's NPV
    npv_com = compute_npv(inv, lifetime_cashflows, disc_rate)

    return npv_com