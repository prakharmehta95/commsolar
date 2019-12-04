
"""
Created on Tue Apr  9 18:45:09 2019

@author: iA

__main__ = run_adoption_loop.py
"""
#%% importing arguments from run_adoption_loop
temp_try = 100
#Price Levels
from __main__ import PV_price_baseline
from __main__ import fit_high           #= 8.5/100 #CHF per kWH
from __main__ import fit_low            #= 4.45/100 #CHF per kWH
from __main__ import ewz_high_large     #= 6/100 #CHF per kWh
from __main__ import ewz_low_large      #= 5/100 #CHF per kWh
from __main__ import ewz_high_small     #= 24.3/100 #CHF per kWh
from __main__ import ewz_low_small      #= 14.4/100 #CHF per kWh
from __main__ import ewz_solarsplit_fee #= 4/100 #CHF per kWH      
from __main__ import diff_prices        #if = 1, then both wholesale and retail prices are applied where appropriate. Else, all retail prices

#PV Panel Properties
from __main__ import PV_lifetime          #= 25 #years
from __main__ import PV_degradation       #= 0.994 #(0.6% every year)
from __main__ import OM_Cost_rate         #= 0.06 # CHF per kWh of solar PV production
from __main__ import disc_rate_homeown    #discount rate for NPV Calculation
from __main__ import disc_rate_firm       #discount rate for NPV Calculation
from __main__ import disc_rate_instn      #discount rate for NPV Calculation
from __main__ import disc_rate_landlord   #discount rate for NPV Calculation
from __main__ import pp_rate              #= 0 - discount rate for the discounted payback period formula, considered zero

#Agent information
from __main__ import agents_info
from __main__ import agent_list_final

#%%
import pandas as pd
#read this now from the MAIN FILE
#agents_info = pd.read_excel(r"C:\\Users\\prakh\\OneDrive - ETHZ\\Thesis\\PM\\Data_Prep_ABM\\Skeleton_Updated_No_100MWh_Restriction.xlsx")
agents_info = agents_info.set_index('bldg_name')    

# =============================================================================
#%% IMPORT SOLAR PV GENERATION FOR EACH BUILDING
import os
#os.chdir(r'C:\Users\prakh\OneDrive - ETHZ\Thesis\PM\Codes\ABM\MasterThesis_PM\masterthesis\TPB')
import glob
import pandas as pd

#read this now from the MAIN FILE
#agent_list_final = pd.read_excel(r'C:\Users\prakh\OneDrive - ETHZ\Thesis\PM\Data_Prep_ABM\LIST_AGENTS_FINAL.xlsx')

#agent_list_final = list(agent_list_final.Bldg_IDs)
#agent_list_final = ['Z0003','Z0004']

df_solar = pd.read_pickle(r'C:\Users\iA\Dropbox\Com_Paper\05_Data\01_CEA_Disaggregated\01_PV_Disagg\CEA_Disaggregated_SolarPV_12Nov.pickle')
#df_solar = pd.DataFrame(data = None)
#for FileList in agent_list_final:
#    print(FileList)
##    path_pv = "C:\\Users\\iA\\OneDrive - ETHZ\\Thesis\\PM\\CEA Data\\Sample 1650\\outputs\\data\\potentials\\solar\\"
    #z = pd.read_csv(path_pv + FileList + '_PV.csv')
    #df_solar[FileList] = ""
    #df_solar[FileList] = z.E_PV_gen_kWh
 
##%% import demands
df_demand = pd.read_pickle(r'C:\Users\iA\Dropbox\Com_Paper\05_Data\01_CEA_Disaggregated\00_Demand_Disagg\CEA_Disaggregated_TOTAL_FINAL_14Nov.pickle')     

#%% CORRECTION FOR SOLAR because of the CEA geometry ussue
#correct_solar = pd.read_excel(r"C:\\Users\\iA\\OneDrive - ETHZ\\Thesis\\PM\\CEA Data\\Sample 1650\\outputs\\data\\potentials\\PV_Corrected_RednFactors.xlsx")
#temp_list = list(correct_solar.columns)

#for i in temp_list:
#    df_solar[i] = correct_solar[i]


#%% multiply the solar PV data with an efficiency factor to convert to AC
#commented out to avoid execution by mistake

df_solar_AC = df_solar*0.97    #multiply by 0.5 to do the 50% reduction
#df_solar_AC_OG = df_solar*0.97
#df_solar_AC = df_solar
#%% adding hours of the day to the demand and supply dataframes
list_hours = []  
ctr = 0  
for i in range(8760):
    if i % 24 == 0:
        ctr = 0
    list_hours.append(ctr)
    ctr = ctr + 1

df_solar_AC['Hour'] = ""
df_solar_AC['Hour'] = list_hours

df_demand['Hour'] = ""
df_demand['Hour'] = list_hours


##%% adding day of the week 
ctr = 0
days = ['Sat','Sun','Mon','Tue','Wed','Thu',' Fri'] #this order because 2005 started with a Saturday - very crude way to code
list_days = []
df_demand['Day'] = ""
df_solar_AC['Day'] = ""
for i in range(365):
    if ctr % 7 == 0:
            ctr = 0
    if ctr == 0:
        for x in range(24):
            list_days.append('Sat')
    if ctr == 1:
        for x in range(24):
            list_days.append('Sun')
    if ctr == 2:
        for x in range(24):
            list_days.append('Mon')
    if ctr == 3:
        for x in range(24):
            list_days.append('Tue')
    if ctr == 4:
        for x in range(24):
            list_days.append('Wed')
    if ctr == 5:
        for x in range(24):
            list_days.append('Thu')
    if ctr == 6:
        for x in range(24):
            list_days.append('Fri')
    ctr = ctr + 1
    
    
df_demand['Day'] = list_days
df_solar_AC['Day'] = list_days

#%% adding info about HIGH/LOW hours of the day
import numpy as np
df_demand['price_level'] = ""
df_solar_AC['price_level'] = ""

#df_solar_AC.loc[df_solar_AC['Hour']>5,]
df_solar_AC['price_level'] = np.where(np.logical_and(np.logical_and(df_solar_AC['Hour'] > 5,df_solar_AC['Hour'] < 22), df_solar_AC['Day'] != 'Sun'),'high','low')
df_demand['price_level'] = np.where(np.logical_and(np.logical_and(df_solar_AC['Hour'] > 5,df_solar_AC['Hour'] < 22), df_solar_AC['Day'] != 'Sun'),'high','low')





#%% PV System Price Projections
'''
PV PRICES in the next years. Base PV price data from EnergieSchweiz.
Projections Source = IEA Technology Roadmap 2014
'''

#import next line from main...    
#PV_price_baseline = pd.read_excel(r'C:\Users\iA\OneDrive - ETHZ\Thesis\PM\Data\Solar PV Cost Projections\PV_Prices.xlsx')

#this stores projected PV prices for all sizes of PV systems
PV_price_projection = pd.DataFrame(data = None)

PV_price_projection['Year'] = ""
years = list(range(2018,2041))
PV_price_projection['Year'] = years

x_array = [i for i in range(1,24)]
xp_array = [1,23]
for i in list(PV_price_baseline.columns):
    fp_array = [PV_price_baseline.loc[0][i],PV_price_baseline.loc[0][i]/2]
    y = np.interp(x_array, xp_array,fp_array)
    PV_price_projection[i] = ""
    PV_price_projection[i] = y
        
        
        
#%% Preparation for NPV Calculation - savings and costs estimations
"""        
PV output reduces every year
demand remains constant
discount rate = 5%
Lifetime = 25 years
O&M costs = 0.06 CHF per kWh of solar PV production
EWZ Fee = 4Rp./kWh of Self consumption 

Separate the high and low hours of the year as prices are different, and then calculate the savings for the year

"""
print("Prep for NPV Calculation")
#dataframes to filter high and low times
df_HIGH = pd.DataFrame(data = None)        
df_LOW = pd.DataFrame(data = None)        
# =============================================================================
# READ IN FROM THE MAIN FILE
# 
# fit_high = 8.5/100 #CHF per kWH
# fit_low =  4.45/100 #CHF per kWH
# 
# #ewz_high = 24.3/100 #CHF per kWH
# #ewz_low = 14.4/100 #CHF per kWH
# ewz_solarsplit_fee = 4/100 #CHF per kWH      
# 
# PV_lifetime = 25 #years
# PV_degradation = 0.994 #(0.6% every year)
# OM_Cost_rate = 0.06 # CHF per kWh of solar PV production
# 
# =============================================================================
Agents_Savings = pd.DataFrame(data = None, index = agent_list_final)
Agents_OM_Costs = pd.DataFrame(data = None, index = agent_list_final)
Agents_EWZ_Costs= pd.DataFrame(data = None, index = agent_list_final)
Agents_NetSavings = pd.DataFrame(data = None, index = agent_list_final)
Agents_SCRs =  pd.DataFrame(data = None, index =agent_list_final)

#-------- O&M costs ------------------
for year in range(PV_lifetime-24):
    print(year)
    col_name = 'Year' + str(year)
    list_om_costs = []
    
    #for fixed O&M costs ( = the O&M costs in the first year, applied for the lifetime of the PV system)
    for i in agent_list_final:
        OM_costs = sum(df_solar_AC[i])*OM_Cost_rate
        Agents_OM_Costs[col_name] = ""
        list_om_costs.append(OM_costs)
    Agents_OM_Costs[col_name] = list_om_costs
#---------------------

for year in range(PV_lifetime-24):
    col_name = 'Year' + str(year)
    list_savings = []
    list_om_costs = []
    list_ewz_costs = []
    list_scrs = []
    print(year)
    for i in agent_list_final:#['Z0003','Z0004']:
        #print(i)
        total_PV_production = sum(df_solar_AC[i])#for SCR calculation
        
        #dataframe COLUMNS initialization to hold solar and demand during HIGH and LOW hours
        df_HIGH[i + '_solar'] = ""
        df_HIGH[i + '_demand'] = ""
        df_LOW[i + '_solar'] = ""
        df_LOW[i + '_demand'] = ""
        
        #solar PV when solar is generating and prices are high or low 
        df_HIGH[i + '_solar'] = df_solar_AC.loc[np.logical_and(df_solar_AC['price_level'] == 'high',df_solar_AC[i] > 0) , i] 
        df_LOW[i + '_solar'] =  df_solar_AC.loc[np.logical_and(df_solar_AC['price_level'] == 'low', df_solar_AC[i] > 0) , i] 
        
        #demand when solar is generating and prices are high or low
        df_HIGH[i + '_demand'] =  df_demand.loc[np.logical_and(df_demand['price_level'] == 'high', df_solar_AC[i] > 0) , i] 
        df_LOW[i + '_demand'] =   df_demand.loc[np.logical_and(df_demand['price_level'] == 'low',  df_solar_AC[i] > 0) , i] 
        
        #dataframe COLUMNS to hold difference between solar and demand during HIGH and LOW hours
        df_HIGH[i + '_PV-dem'] = ""
        df_LOW[i + '_PV-dem'] = ""
        
        df_HIGH[i + '_PV-dem'] = df_HIGH[i + '_solar'] - df_HIGH[i + '_demand']
        df_LOW[i + '_PV-dem'] = df_LOW[i + '_solar'] - df_LOW[i + '_demand']
        
        #for cases when feed-in occurs (PV > demand i.e. PV-dem is +ve)
        #high times
        list_extraPV_HIGH = []
        list_dem_selfcons_HIGH = []
        list_extraPV_HIGH = df_HIGH.loc[df_HIGH[i + '_PV-dem'] >= 0, i + '_PV-dem']
        list_dem_selfcons_HIGH = df_HIGH.loc[df_HIGH[i + '_PV-dem'] >= 0, i + '_demand']
        sum_extraPV_HIGH = sum(list_extraPV_HIGH) 
        sum_dem_selfcons_HIGH = sum(list_dem_selfcons_HIGH) 
        #low times
        list_extraPV_LOW = []
        list_dem_selfcons_LOW = []
        list_extraPV_LOW = df_LOW.loc[df_LOW[i + '_PV-dem'] >= 0, i + '_PV-dem']
        list_dem_selfcons_LOW = df_LOW.loc[df_LOW[i + '_PV-dem'] >= 0, i + '_demand']
        sum_extraPV_LOW = sum(list_extraPV_LOW) 
        sum_dem_selfcons_LOW = sum(list_dem_selfcons_LOW)
        
        #for cases when only SELF-CONSUMPTION i.e. NO feed-in occurs (PV < demand i.e. PV-dem is -ve)
        #high times
        list_selfcons_HIGH = []
        list_selfcons_HIGH = df_HIGH.loc[df_HIGH[i + '_PV-dem'] < 0, i + '_solar']
        sum_selfcons_HIGH = sum(list_selfcons_HIGH)
        #low times
        list_selfcons_LOW = []
        list_selfcons_LOW =  df_LOW.loc [df_LOW[i + '_PV-dem'] < 0 , i + '_solar']
        sum_selfcons_LOW = sum(list_selfcons_LOW)
        
        
        if diff_prices == 1:                    #wholesale or retail electricity pricing
            if agents_info.loc[i]['demand_yearly_kWh'] >=100000:
                ewz_high = ewz_high_large       #6/100 #CHF per kWh
                ewz_low = ewz_low_large         #5/100 #CHF per kWh
            elif agents_info.loc[i]['demand_yearly_kWh'] < 100000:
                ewz_high = ewz_high_small       #24.3/100 #CHF per kWh
                ewz_low = ewz_low_small         #14.4/100 #CHF per kWh
        elif diff_prices == 0:                  #retail electricity pricing for all
            ewz_high = ewz_high_small           #24.3/100 #CHF per kWh
            ewz_low = ewz_low_small             #14.4/100 #CHF per kWh
            
        savings = (sum_extraPV_HIGH*fit_high + sum_dem_selfcons_HIGH*ewz_high + sum_selfcons_HIGH * ewz_high +
                   sum_extraPV_LOW*fit_low   + sum_dem_selfcons_LOW*ewz_low   + sum_selfcons_LOW * ewz_low)
        #print(savings)
        
        total_self_consumption = sum_dem_selfcons_HIGH + sum_dem_selfcons_LOW + sum_selfcons_HIGH + sum_selfcons_LOW
        ewz_solarsplit_costs = total_self_consumption*ewz_solarsplit_fee
        
        if total_PV_production == 0:
            scrs = 0
        else:
            scrs = total_self_consumption/total_PV_production
        
        Agents_Savings[col_name] = ""
        #Agents_OM_Costs[col_name] = ""
        Agents_EWZ_Costs[col_name] = ""
        Agents_SCRs[col_name] = ""

        list_savings.append(savings)
        #list_om_costs.append(OM_costs)
        list_ewz_costs.append(ewz_solarsplit_costs)
        list_scrs.append(scrs)
        
        #degrading PV output every year
        df_solar_AC[i] = df_solar_AC[i]*(PV_degradation)
        
    Agents_Savings[col_name] = list_savings
    #Agents_OM_Costs[col_name] = list_om_costs
    Agents_EWZ_Costs[col_name] = list_ewz_costs
    Agents_SCRs[col_name] = list_scrs

Agents_NetSavings = Agents_Savings - Agents_OM_Costs - Agents_EWZ_Costs
#end = time.time()
#print("Code Execution Time = ",end - start)   



#%% NPV Calculation
'''
small PV = < 100kW (medium is betwwen 30 and 100)
large PV = >= 100kW
'''

print("NPV Calculation")

#disc_rate = 0.05 - NOW READ FROM THE MAIN FILE

Agents_NPVs = pd.DataFrame(data = None, index = list(range(0,18)), columns = agent_list_final)
Agents_NPVs['Installation_Year'] = list(range(2018,2036))

Agents_PPs = pd.DataFrame(data = None, index = list(range(0,18)), columns = agent_list_final)
Agents_PPs['Installation_Year'] = list(range(2018,2036))
Agents_PPs_Norm = pd.DataFrame(data = None, index = list(range(0,18)), columns = agent_list_final)
Agents_PPs_Norm['Installation_Year'] = list(range(2018,2036))

Agents_Investment_Costs = pd.DataFrame(data = None, index = list(range(0,18)), columns = agent_list_final)
Agents_Investment_Costs['Installation_Year'] = list(range(2018,2036))
Agents_PV_Investment_Costs = pd.DataFrame(data = None, index = list(range(0,18)), columns = agent_list_final)
Agents_PV_Investment_Costs['Installation_Year'] = list(range(2018,2036))
Agents_Smart_Meter_Investment_Costs = pd.DataFrame(data = None, index = list(range(0,18)), columns = agent_list_final)
Agents_Smart_Meter_Investment_Costs['Installation_Year'] = list(range(2018,2036))


temp_net_yearlysavings = []
for row in Agents_NetSavings.iterrows():
    index, data = row
    temp_net_yearlysavings.append(data.tolist())
 
temp_savings_df = pd.DataFrame({'col':temp_net_yearlysavings})
temp_savings_df['Bldg_IDs'] = ""
temp_savings_df['Bldg_IDs'] = agent_list_final
temp_savings_df = temp_savings_df.set_index('Bldg_IDs')              

for i in ['B140995']:#agent_list_final:
    inv_cost_list = []
    temp_npv_list = []
    temp_payback_list = []
    smart_meter_inv_cost_list = []
    pv_inv_cost_list = []
    
    if agents_info.loc[i]["bldg_owner"] == "Homeowner":
        disc_rate_npv = disc_rate_homeown
    elif agents_info.loc[i]["bldg_owner"] == "Landlord":
        disc_rate_npv = disc_rate_landlord
    elif agents_info.loc[i]["bldg_owner"] == "Firm":
        disc_rate_npv = disc_rate_firm
    elif agents_info.loc[i]["bldg_owner"] == "Institution":
        disc_rate_npv = disc_rate_instn
    
    print(i)
    for install_year in range(18): # 0 = 2018, 17 = 2035
        #print(i)
        temp_pv_subsidy =  agents_info.loc[i]['pv_subsidy']
        if install_year >= 12:
            temp_pv_subsidy =  0
        
        temp_pv_size =   agents_info.loc[i]['pv_size_kw']
        temp_num_meters = agents_info.loc[i]['num_smart_meters']
        
        #depending on the PV system size, the investment cost per kW changes
        if temp_pv_size <= 2:
            invest_rate = PV_price_projection.loc[install_year]['Two']
        elif temp_pv_size == 3:
            invest_rate = PV_price_projection.loc[install_year]['Three']
        elif temp_pv_size == 4:
            invest_rate = PV_price_projection.loc[install_year]['Four']
        elif temp_pv_size == 5:
            invest_rate = PV_price_projection.loc[install_year]['Five']
        elif 5 < temp_pv_size < 10 :
            invest_rate = PV_price_projection.loc[install_year]['Five']
        elif 10 <= temp_pv_size < 15:
            invest_rate = PV_price_projection.loc[install_year]['Ten']
        elif 15 <= temp_pv_size < 20:
            invest_rate = PV_price_projection.loc[install_year]['Fifteen']
        elif 20 <= temp_pv_size < 30:
            invest_rate = PV_price_projection.loc[install_year]['Twenty']
        elif 30 <= temp_pv_size < 50:
            invest_rate = PV_price_projection.loc[install_year]['Thirty']
        elif 50 <= temp_pv_size < 75:
            invest_rate = PV_price_projection.loc[install_year]['Fifty']
        elif 75 <= temp_pv_size < 100:
            invest_rate = PV_price_projection.loc[install_year]['Seventy-five']
        elif 100 <= temp_pv_size < 125:
            invest_rate = PV_price_projection.loc[install_year]['Hundred']
        elif 125 <= temp_pv_size < 150:
            invest_rate = PV_price_projection.loc[install_year]['Hundred-twenty-five']
        elif temp_pv_size == 150:
            invest_rate = PV_price_projection.loc[install_year]['One-Fifty']
        elif temp_pv_size > 150:
            invest_rate = PV_price_projection.loc[install_year]['Greater']
        
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
        investment_cost         = pv_inv_cost + smart_meter_inv_cost
        
        pv_inv_cost_list.append(pv_inv_cost)
        smart_meter_inv_cost_list.append(smart_meter_inv_cost)
        inv_cost_list.append(investment_cost) 
        

        net_investment  = -1*investment_cost + temp_pv_subsidy
        cash_flows      = [net_investment]
        savings_temp    = temp_savings_df.loc[i]['col']
        
        cash_flows.extend(savings_temp)
        temp_npv = np.npv(disc_rate_npv,cash_flows)
        temp_npv_list.append(temp_npv) #only npv is stored 
        
        cf_df = pd.DataFrame(cash_flows, columns=['UndiscountedCashFlows'])
        cf_df.index.name = 'Year'
        cf_df['DiscountedCashFlows'] = np.pv(rate=pp_rate, pmt=0, nper=cf_df.index, fv=-cf_df['UndiscountedCashFlows'])
        cf_df['CumulativeDiscountedCashFlows'] = np.cumsum(cf_df['DiscountedCashFlows'])
        if any(cf_df.CumulativeDiscountedCashFlows > 0):
            final_full_year = cf_df[cf_df.CumulativeDiscountedCashFlows < 0].index.values.max()
            fractional_yr = -cf_df.CumulativeDiscountedCashFlows[final_full_year ]/cf_df.DiscountedCashFlows[final_full_year + 1]
        else:
            final_full_year = 999
            fractional_yr = 0
        
        payback_period = final_full_year + fractional_yr
        temp_payback_list.append(payback_period)   
        print(payback_period)
        
    
    #end of a building
    Agents_NPVs[i] = temp_npv_list
    Agents_PPs[i] = temp_payback_list
    Agents_PPs_Norm[i] = 1 - Agents_PPs[i]/15
    Agents_PPs_Norm.loc[Agents_PPs_Norm[i]<0,i] = 0
    Agents_Investment_Costs[i] = inv_cost_list
    Agents_PV_Investment_Costs[i] = pv_inv_cost_list
    Agents_Smart_Meter_Investment_Costs[i] = smart_meter_inv_cost_list
    

#%% writing data to pickle

#Agents_NPVs.to_pickle(r'C:\Users\iA\Dropbox\Com_Paper\06_Code\02_Results\Agents_Ind_NPVs_Years.pickle')

#Agents_PPs.to_pickle(r'C:\Users\iA\Dropbox\Com_Paper\06_Code\02_Results\Agents_Ind_PPs_Years.pickle')
    
#Agents_PPs_Norm.to_pickle(r'C:\Users\iA\Dropbox\Com_Paper\06_Code\02_Results\Agents_Ind_PPs_Norm_Years.pickle')

#Agents_SCRs.to_pickle(r'C:\Users\iA\Dropbox\Com_Paper\06_Code\02_Results\Agents_Ind_SCR_Years.pickle')

#Agents_Investment_Costs.to_pickle(r'C:\Users\iA\Dropbox\Com_Paper\06_Code\02_Results\Agents_Ind_Total_InvestmentCosts_Years.pickle')

#Agents_Savings.to_pickle(r'C:\Users\iA\Dropbox\Com_Paper\06_Code\02_Results\Agents_Ind_Savings_Years.pickle')

#Agents_NetSavings.to_pickle(r'C:\Users\iA\Dropbox\Com_Paper\06_Code\02_Results\Agents_Ind_NetSavings_Years.pickle')

#Agents_OM_Costs.to_pickle(r'C:\Users\iA\Dropbox\Com_Paper\06_Code\02_Results\Agents_Ind_OM_Costs_Years.pickle')

#Agents_PV_Investment_Costs.to_pickle(r'C:\Users\iA\Dropbox\Com_Paper\06_Code\02_Results\Agents_Ind_Investment_Costs_Years.pickle')

#Agents_Smart_Meter_Investment_Costs.to_pickle(r'C:\Users\iA\Dropbox\Com_Paper\06_Code\02_Results\Agents_Ind_Smart_Meter_Costs_Years.pickle')


