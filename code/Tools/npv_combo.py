# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 14:13:11 2019

@author: iA
"""
#%%

def npv_calc_combos(df_solar_AC, df_demand, year_model, agent_enchamp_type, df_pv_size, 
                    df_pv_size_cost, df_num_smartmeters, df_num_smartmeters_cost,
                    df_num_members, disc_rate,
                    fit_high, fit_low, ewz_high_large,ewz_low_large,
                    ewz_high_small, ewz_low_small,ewz_solarsplit_fee,
                    PV_lifetime, PV_degradation, OM_Cost_rate,PV_price_projection,
                    list_hours, daylist,diff_prices):
    
    '''
    df_solar_AC             = PV potential of all the possible combinations
    df_demand               = Demands of all the possible combinations
    year_model              = Year the model is in. Used for taking the right cost of PV system (in future years cost reduces)
    agent_enchamp_type      = owner type of the activated agent
    df_pv_size              = PV system size of all the possible combinations
    df_pv_size_cost         = PV system size of NEW installation in all the possible combinations for which there is incurred cost
    df_num_smartmeters      = Number of smart meters of all the possible combinations
    df_num_smartmeters_cost = Number of NEW smart meters of all the possible combinations for which there is incurred cost
    df_num_members          = Number of members (agents) in all the possible combinations
    disc_rate               = Discount rate used for NPV. set to 0.05 in the main script
    fit_high                = Read in from the main script 8.5/100 #CHF per kWH
    fit_low                 = Read in from the main script 4.45/100 #CHF per kWH
    ewz_high_large          = Read in from the main script 6/100 
    ewz_low_large           = Read in from the main script 5/100
    ewz_high_small          = Read in from the main script 24.3/100
    ewz_low_small           = Read in from the main script 14.4/100
    ewz_solarsplit_fee      = Read in from the main script 4/100 #CHF per kWH      
    PV_lifetime             = Read in from the main script 25 #years
    PV_degradation          = Read in from the main script 0.994 #(0.6% every year) - not allowed to degrade as NPV is calculated only for 1 year
    OM_Cost_rate            = Read in from the main script 0.06 # CHF per kWh of solar PV production
    list_hours              = List of hours in the day from 0-23 and then repeats
    daylist                 = List of days in the week
    diff_prices             = if set to 1, different prices for retail (<100 MWh) and wholesale (>=100 MWh). Else same prices
    
    returns Agents_NPVs     = NPVs of all combos  
    '''
    
    import pandas as pd
    import numpy as np
    agent_list_final = df_solar_AC.columns
    
    install_year = year_model 
    disc_rate_npv = disc_rate

    #adding hours to the solar and demand dataframes for hourly pricing later
    df_solar_AC['Hour'] = list_hours
    df_demand['Hour'] = list_hours
    #adding day of the week 
    df_demand['Day']    = daylist
    df_solar_AC['Day']  = daylist
    
    #adding price information depending on hour of day
    df_demand['price_level']    = ""
    df_solar_AC['price_level']  = ""
    #adding 'high' and 'low' price hours for the electricity
    df_solar_AC['price_level']  = np.where(np.logical_and(np.logical_and(df_solar_AC['Hour'] > 5,df_solar_AC['Hour'] < 22), df_solar_AC['Day'] != 'Sun'),'high','low')
    df_demand['price_level']    = np.where(np.logical_and(np.logical_and(df_solar_AC['Hour'] > 5,df_solar_AC['Hour'] < 22), df_solar_AC['Day'] != 'Sun'),'high','low')
    
#%%
    """        
    NPV Calculation Preparation of dataframes, ToU pricing, etc...
    """
    #dataframes to filter high and low times
    df_HIGH = pd.DataFrame(data = None)        
    df_LOW  = pd.DataFrame(data = None)        

    Agents_Savings      = pd.DataFrame(data = None, index = agent_list_final)
    Agents_OM_Costs     = pd.DataFrame(data = None, index = agent_list_final)
    Agents_EWZ_Costs    = pd.DataFrame(data = None, index = agent_list_final)
    Agents_NetSavings   = pd.DataFrame(data = None, index = agent_list_final)
    Agents_SCRs         = pd.DataFrame(data = None, index = agent_list_final)
    
    #-------- O&M costs ------------------
    for year in range(PV_lifetime): #only calculating for one year 
        col_name = 'Year' + str(year)
        list_om_costs = []
        
        #for fixed O&M costs ( = the O&M costs in the first year, applied for the lifetime of the PV system)
        for i in agent_list_final:
            #print(df_solar_AC[i])
            OM_costs = sum(df_solar_AC[i])*OM_Cost_rate
            Agents_OM_Costs[col_name] = ""
            list_om_costs.append(OM_costs)
        Agents_OM_Costs[col_name] = list_om_costs
    #---------------------
    
    for year in range(PV_lifetime): #only calculating for one year 
        col_name = 'Year' + str(year)
        list_savings = []
        list_om_costs = []
        list_ewz_costs = []
        list_scrs = []
        for i in agent_list_final:
            total_PV_production = sum(df_solar_AC[i])#for SCR calculation
            total_demand        = sum(df_demand[i])#for SCR calculation
            
            #dataframe COLUMNS initialization to hold solar and demand during HIGH and LOW hours
            df_HIGH[i + '_solar']   = ""
            df_HIGH[i + '_demand']  = ""
            df_LOW[i + '_solar']    = ""
            df_LOW[i + '_demand']   = ""
            
            #solar PV when solar is generating and prices are high or low 
            df_HIGH[i + '_solar']   = df_solar_AC.loc[np.logical_and(df_solar_AC['price_level'] == 'high',df_solar_AC[i] > 0) , i] 
            df_LOW[i + '_solar']    =  df_solar_AC.loc[np.logical_and(df_solar_AC['price_level'] == 'low', df_solar_AC[i] > 0) , i] 
            
            #demand when solar is generating and prices are high or low
            df_HIGH[i + '_demand']  = df_demand.loc[np.logical_and(df_demand['price_level'] == 'high', df_solar_AC[i] > 0) , i] 
            df_LOW[i + '_demand']   = df_demand.loc[np.logical_and(df_demand['price_level'] == 'low',  df_solar_AC[i] > 0) , i] 
            
            #dataframe COLUMNS to hold difference between solar and demand during HIGH and LOW hours
            df_HIGH[i + '_PV-dem']  = ""
            df_LOW[i + '_PV-dem']   = ""
            
            df_HIGH[i + '_PV-dem']  = df_HIGH[i + '_solar'] - df_HIGH[i + '_demand']
            df_LOW[i + '_PV-dem']   = df_LOW[i + '_solar'] - df_LOW[i + '_demand']
            
            #for cases when feed-in occurs (PV > demand i.e. PV-dem is +ve)
            #high times
            list_extraPV_HIGH       = []
            list_dem_selfcons_HIGH  = []
            list_extraPV_HIGH       = df_HIGH.loc[df_HIGH[i + '_PV-dem'] >= 0, i + '_PV-dem']
            list_dem_selfcons_HIGH  = df_HIGH.loc[df_HIGH[i + '_PV-dem'] >= 0, i + '_demand']
            sum_extraPV_HIGH        = sum(list_extraPV_HIGH) 
            sum_dem_selfcons_HIGH   = sum(list_dem_selfcons_HIGH) 
            #low times
            list_extraPV_LOW        = []
            list_dem_selfcons_LOW   = []
            list_extraPV_LOW        = df_LOW.loc[df_LOW[i + '_PV-dem'] >= 0, i + '_PV-dem']
            list_dem_selfcons_LOW   = df_LOW.loc[df_LOW[i + '_PV-dem'] >= 0, i + '_demand']
            sum_extraPV_LOW         = sum(list_extraPV_LOW) 
            sum_dem_selfcons_LOW    = sum(list_dem_selfcons_LOW)
            
            #for cases when only SELF-CONSUMPTION i.e. NO feed-in occurs (PV < demand i.e. PV-dem is -ve)
            #high times
            list_selfcons_HIGH  = []
            list_selfcons_HIGH  = df_HIGH.loc[df_HIGH[i + '_PV-dem'] < 0, i + '_solar']
            sum_selfcons_HIGH   = sum(list_selfcons_HIGH)
            #low times
            list_selfcons_LOW   = []
            list_selfcons_LOW   =  df_LOW.loc [df_LOW[i + '_PV-dem'] < 0 , i + '_solar']
            sum_selfcons_LOW    = sum(list_selfcons_LOW)
            
            if diff_prices == 1:                    #wholesale or retail electricity pricing
                if total_demand >=100000:
                    ewz_high    = ewz_high_large       #6/100 #CHF per kWh
                    ewz_low     = ewz_low_large         #5/100 #CHF per kWh
                elif total_demand < 100000:
                    ewz_high    = ewz_high_small       #24.3/100 #CHF per kWh
                    ewz_low     = ewz_low_small         #14.4/100 #CHF per kWh
            elif diff_prices == 0:                  #retail electricity pricing for all
                ewz_high    = ewz_high_small           #24.3/100 #CHF per kWh
                ewz_low     = ewz_low_small             #14.4/100 #CHF per kWh
                
            savings = (sum_extraPV_HIGH*fit_high + sum_dem_selfcons_HIGH*ewz_high + sum_selfcons_HIGH * ewz_high +
                       sum_extraPV_LOW*fit_low   + sum_dem_selfcons_LOW*ewz_low   + sum_selfcons_LOW * ewz_low)
            #print(savings)
            
            total_self_consumption  = sum_dem_selfcons_HIGH + sum_dem_selfcons_LOW + sum_selfcons_HIGH + sum_selfcons_LOW
            ewz_solarsplit_costs    = total_self_consumption*ewz_solarsplit_fee
            
            if total_PV_production == 0:
                scrs = 0
            else:
                scrs = total_self_consumption/total_PV_production
            
            Agents_Savings[col_name]    = ""
            Agents_EWZ_Costs[col_name]  = ""
            Agents_SCRs[col_name]       = ""
    
            list_savings.append(savings)
            list_ewz_costs.append(ewz_solarsplit_costs)
            list_scrs.append(scrs)
            
            #degrading PV output every year
            df_solar_AC[i] = df_solar_AC[i]*(PV_degradation)
            
        Agents_Savings[col_name]    = list_savings
        Agents_EWZ_Costs[col_name]  = list_ewz_costs
        Agents_SCRs[col_name]       = list_scrs
    
    Agents_NetSavings = Agents_Savings - Agents_OM_Costs - Agents_EWZ_Costs
    
    '''
    Actual NPV calculation happens now
    small PV    = < 30kW 
    medium PV   =  >= 30 and  < 100kW
    large PV    = >= 100kW
    '''
    
    #print("NPV Calculation for Combos")
    Agents_NPVs = pd.DataFrame(data = None, index = agent_list_final, columns = ['npv'])
    
    temp_net_yearlysavings = []
    for row in Agents_NetSavings.iterrows():
        index, data = row
        temp_net_yearlysavings.append(data.tolist())
     
    temp_savings_df             = pd.DataFrame({'col':temp_net_yearlysavings})
    temp_savings_df['Bldg_IDs'] = ""
    temp_savings_df['Bldg_IDs'] = agent_list_final
    temp_savings_df             = temp_savings_df.set_index('Bldg_IDs')              
    inv_cost_list               = []
    temp_npv_list               = []
    temp_payback_list           = []
    smart_meter_inv_cost_list   = []
    pv_inv_cost_list            = []
    invest_rate                 = 0
    invest_meter_rate           = 0
    temp_pv_subsidy             = 0
    coop_cost                   = 0
    for i in agent_list_final:
        
        temp_pv_size    = df_pv_size_cost.loc['Size'][i]
        temp_num_meters = df_num_smartmeters_cost.loc['Num'][i]
        
        if temp_pv_size < 30:
            temp_pv_subsidy =  1600 + 460*temp_pv_size
        elif 30 <= temp_pv_size < 100:
            temp_pv_subsidy =  1600 + 340*temp_pv_size
        elif temp_pv_size >= 100:
            temp_pv_subsidy =  1400 + 300*temp_pv_size
        
        #subsidy stops
        if install_year >= 12: 
            temp_pv_subsidy =  0
        
        #depending on the PV system size, the investment cost per kW changes
        if temp_pv_size <= 2:
            invest_rate = PV_price_projection.at[install_year,'Two']
        elif temp_pv_size == 3:
            invest_rate = PV_price_projection.at[install_year,'Three']
        elif temp_pv_size == 4:
            invest_rate = PV_price_projection.at[install_year,'Four']
        elif temp_pv_size == 5:
            invest_rate = PV_price_projection.at[install_year,'Five']
        elif 5 < temp_pv_size < 10 :
            invest_rate = PV_price_projection.at[install_year,'Five']
        elif 10 <= temp_pv_size < 15:
            invest_rate = PV_price_projection.at[install_year,'Ten']
        elif 15 <= temp_pv_size < 20:
            invest_rate = PV_price_projection.at[install_year,'Fifteen']
        elif 20 <= temp_pv_size < 30:
            invest_rate = PV_price_projection.at[install_year,'Twenty']
        elif 30 <= temp_pv_size < 50:
            invest_rate = PV_price_projection.at[install_year,'Thirty']
        elif 50 <= temp_pv_size < 75:
            invest_rate = PV_price_projection.at[install_year,'Fifty']
        elif 75 <= temp_pv_size < 100:
            invest_rate = PV_price_projection.at[install_year,'Seventy-five']
        elif 100 <= temp_pv_size < 125:
            invest_rate = PV_price_projection.at[install_year,'Hundred']
        elif 125 <= temp_pv_size < 150:
            invest_rate = PV_price_projection.at[install_year,'Hundred-twenty-five']
        elif temp_pv_size == 150:
            invest_rate = PV_price_projection.at[install_year,'One-Fifty']
        elif temp_pv_size > 150:
            invest_rate = PV_price_projection.at[install_year,'Greater']
        
        #depending on the number of smart meters to be installed, the meter_investment cost per meter changes
        '''
        check these prices again before running!
        '''
        if temp_num_meters <= 8:
            invest_meter_rate = 375 #CHF per smart meter
        elif temp_num_meters == 9:
            invest_meter_rate = 360
        elif 10 <= temp_num_meters < 12:
            invest_meter_rate = 337
        elif 12 <= temp_num_meters < 15:
            invest_meter_rate = 302
        elif 15 <= temp_num_meters < 20:
            invest_meter_rate = 268
        elif 20 <= temp_num_meters < 25:
            invest_meter_rate = 233
        elif 25 <= temp_num_meters < 30:
            invest_meter_rate = 246
        elif 30 <= temp_num_meters < 35:
            invest_meter_rate = 227
        elif 35 <= temp_num_meters < 40:
            invest_meter_rate = 223
        elif 40 <= temp_num_meters < 45:
            invest_meter_rate = 212
        elif 45 <= temp_num_meters < 50:
            invest_meter_rate = 203
        elif temp_num_meters >= 50:
            invest_meter_rate = 195
        
        pv_inv_cost             = invest_rate*temp_pv_size
        smart_meter_inv_cost    = temp_num_meters*invest_meter_rate
        
        coop_cost               = 0 # COOPERATION COST IS SET TO ZERO
        investment_cost         = pv_inv_cost + smart_meter_inv_cost + coop_cost # + some calculated cooperation cost dependent on the number of potential community members
        #print(investment_cost)
        
        pv_inv_cost_list.append(pv_inv_cost)
        smart_meter_inv_cost_list.append(smart_meter_inv_cost)
        inv_cost_list.append(investment_cost) 

        net_investment  = -1*investment_cost + temp_pv_subsidy
        cash_flows      = [net_investment]
        savings_temp    = temp_savings_df.loc[i]['col']
        
        cash_flows.extend(savings_temp)
        temp_npv = np.npv(disc_rate_npv,cash_flows)
        temp_npv_list.append(temp_npv) #only npv is stored 
    
    Agents_NPVs['npv'] = temp_npv_list
    return Agents_NPVs

