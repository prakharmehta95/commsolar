# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 14:13:01 2019

@author: Prakhar Mehta
"""

#%%
import pandas as pd
def community_combinations(data_og, same_plot_agents_positive_intention, distances, df_solar, df_demand,
                           df_solar_combos_main , df_demand_combos_main, Combos_formed_Info, uid, zone_uid,
                           no_closest_neighbors_consider, year,Agents_Ind_NPVs, 
                           disc_rate, fit_high, fit_low, ewz_high_large,ewz_low_large,
                           ewz_high_small, ewz_low_small,ewz_solarsplit_fee,
                           PV_lifetime, PV_degradation, OM_Cost_rate,
                           npv_combo, rank_combos,PV_price_projection,
                           list_hours, daylist,diff_prices):
    
    '''
    data_og                             = agents_info - Info dataframe on all agents. Contains who passed the intention stage, who has individal/community PV, IDs of the adopted PV systems  
    same_plot_agents_positive_intention = all agents in the same plot as the active agent who have passed the intention stage
    distances                           = Holds the distances from each agent to its nearest 200 agents (200 agents because dataframe size is reasonable but still satisfies all criteria for the ABM conceptual model)
    df_solar                            = Individual building solar PV potential
    df_demand                           = Individual building demands
    df_solar_combos_main                = Solar PV potential of already formed communities
    df_demand_combos_main               = Demands of already formed communities
    Combos_formed_Info                  = Info on already formed combinations
    no_closest_neighbors_consider       = how many closest neighbours must be considered to form a community together with
    uid                                 = ID of the activated agent. Eg = B123456
    year                                = Year of the simulation. 0 = 2018, 1 = 2019 and so on.
    Agents_Ind_NPVs                     = Individual NPVs of the agents 
    disc_rate                           = Discount rate used for NPV. set to 0.05 in the main script
    fit_high                            = Read in from the main script 8.5/100 #CHF per kWH
    fit_low                             = Read in from the main script 4.45/100 #CHF per kWH
    ewz_high_large                      = Read in from the main script 6/100 
    ewz_low_large                       = Read in from the main script 5/100
    ewz_high_small                      = Read in from the main script 24.3/100
    ewz_low_small                       = Read in from the main script 14.4/100
    ewz_solarsplit_fee                  = Read in from the main script 4/100 #CHF per kWH      
    PV_lifetime                         = Read in from the main script 25 #years
    PV_degradation                      = Read in from the main script 0.994 #(0.6% every year) - not allowed to degrade as NPV is calculated only for 1 year
    OM_Cost_rate                        = Read in from the main script 0.06 # CHF per kWh of solar PV production
    
    '''
    #from npv_combos_function import npv_calc_combos
    #from combos_ranking import ranking_combos
    
    data = data_og.copy()
    temp_distances                  = pd.DataFrame(data = None)
    
    #if more than 4 closest agents have positive intention, choose the closest ones and store it in combos_consider
    if len(same_plot_agents_positive_intention.index) > no_closest_neighbors_consider:
        same_plot_agents_positive_intention['dist'] = ""
        temp_str = 'dist_' + str(uid)     #i = agent uid here, hence no need of a loop to iterate over and change i
        temp_distances[uid] = distances[uid]
        temp_distances['Distances']= distances[temp_str]
        temp_distances = temp_distances.set_index(uid)
        for j in same_plot_agents_positive_intention.index:
            try:
                same_plot_agents_positive_intention.at[j,'dist'] = temp_distances.loc[j]['Distances']
            except KeyError:
                same_plot_agents_positive_intention.at[j,'dist'] = 0
                #this usually happens because the same building is referenced and we do not have distance between bldg A and bldg A = 0, duh!
        
        same_plot_agents_positive_intention = same_plot_agents_positive_intention.sort_values(by = ['dist'])
        combos_consider                     = same_plot_agents_positive_intention.head(5).copy() #closest 4 neighbours
    else:
        combos_consider                     = same_plot_agents_positive_intention.copy() #closest 4 neighbours
    
    if len(combos_consider.index) > 0:
        #only then is it worth doing all the computation in the sub-functions:
        
        #so that if someone has formed a community or individual PV it is taken in to account here
        temp_combos_list_temp = list(combos_consider.index)
        temp_combos_list_filter = [combos_consider.at[i,'Community_ID'] if combos_consider.at[i,'Adopt_COMM'] == 1 else combos_consider.at[i,'Individual_ID'] if combos_consider.at[i,'Adopt_IND'] == 1  else i for i in temp_combos_list_temp]
        
        
        #make combinations in the following lines of code--------------------------
        import itertools
        temp_combos_list = []
        constant =  uid                              #read this from the self.uid
        if constant in temp_combos_list_filter:
            temp_combos_list_filter.remove(constant) #so that the energy_champion building does not get eliminated in the combos - the combos are made without it and then this building is added to all the combos
        if '' in temp_combos_list_filter:
            temp_combos_list_filter.remove('')
        
        if '' in temp_combos_list_filter:
            temp_combos_list_filter.remove('')
        
        #print('removing blank from temp_combos_list_filter = ',temp_combos_list_filter)
        
        combos_consider_calc = temp_combos_list_filter.copy()
        combos_consider_calc = list(set(combos_consider_calc)) #remove repitition in the combos_consider_calc - will happen in case communities are already existing
        
        #create combinations without the activated agent, and then add the activated agent to all combos formed
        for j in range(0,len(combos_consider_calc)):
            for k in itertools.combinations(combos_consider_calc,len(combos_consider_calc)-j):
                temp_combos         = list(k)
                temp_combos[0:0]    = [constant]        #adding the active agent to all combos made without it
                temp_combos_list.append(temp_combos)
        
        #since we add the active agent to the list of combos, the last element in the list is just the active agent alone.
        #Since that is not a combo, it is removed in the next loop.
        #individual adoption case is taken care of in the main ABM
        for i in range(len(temp_combos_list)):
            if len(temp_combos_list[i]) == 1:
                del temp_combos_list[i] 
        
        #temp_combos_list - THIS CONTAINS ALL THE  POSSIBLE COMBINATIONS
        #combinations made!-------------------------------------------------------- 
        
        #%% COLLECTING AND STORING ALL INFO ON COMBINATIONS
        # IT IS NEEDED TO CALCULATE THE NPVs FOR THE COMBINATIONS
        #print("-------here,now----------------")
        #print(combos_consider)
        if len(temp_combos_list) > 0:
            #make the solar and demand info for all the combinations to calculate the NPVs
            df_solar_combo              = pd.DataFrame(data = None)
            df_demand_combo             = pd.DataFrame(data = None)
            df_pvsize_combo             = pd.DataFrame(data = None, index = ['Size'])
            df_pvsize_cost_combo        = pd.DataFrame(data = None, index = ['Size'])
            df_bldgs_names              = pd.DataFrame(data = None, index = ['Bldg_Names'])
            df_zones_names              = pd.DataFrame(data = None, index = ['Zone_Names'])
            df_join_individual          = pd.DataFrame(data = None, index = ['Join_Ind'])
            df_join_community           = pd.DataFrame(data = None, index = ['Join_Comm'])
            df_num_smart_meters         = pd.DataFrame(data = None, index = ['Num'])
            df_num_smart_meters_cost    = pd.DataFrame(data = None, index = ['Num'])
            df_num_members              = pd.DataFrame(data = None, index = ['Num_Members'])
            #print("temp_combos_list =========",temp_combos_list)
            set_flag_pros_cons = 0              #to make 2 cases - consumer (= 1) and prosumer (= 0)
            set_flag_pros_cons_individual = 0
            
            print(Combos_formed_Info)
            for i in range(len(temp_combos_list)):
                temp_solar = 0
                temp_demand = 0
                #maybne no need for this
                temp_name = '' #uid + '_' #because the name of the community must start with the activated agent
                temp_pv_list = []
                temp_pv_cost_list = []
                temp_pv_size = 0
                temp_bldg_og_name_list_df = []
                temp_bldg_og_name_list = []
                temp_bldg_zones_list_df = []
                #maybe no need for this
                temp_bldg_zones_list = []
                temp_join_individual_list = []
                temp_join_individual = 0
                temp_join_community_list = []
                temp_join_community = 0
                temp_num_smartmeters_list = []
                temp_num_smartmeters_cost_list = []
                temp_num_members_list = []
                temp_num_smart_meters = 0
                temp_num_smart_meters_cost = 0
                temp_num_members = 0
                temp_names_comms_list = []
                
                for j in range(len(temp_combos_list[i])):
                    try:
                        temp_bldg                       = temp_combos_list[i][j] 
                        temp_solar                      = temp_solar + df_solar[temp_bldg]
                        temp_demand                     = temp_demand + df_demand[temp_bldg]
                        temp_pv_size                    = temp_pv_size + combos_consider.at[temp_bldg,'pv_size_kw']
                        temp_pv_size_cost               = temp_pv_size + combos_consider.at[temp_bldg,'pv_size_kw']
                        temp_bldg_og_name_list.append(temp_bldg)
                        #temp_bldg_og_name_list.append([temp_bldg])
                        temp_bldg_zones_list.append(combos_consider.at[temp_bldg,'zone_id'])
                        temp_num_smart_meters           = temp_num_smart_meters + combos_consider.at[temp_bldg,'num_smart_meters']
                        temp_num_smart_meters_cost      = temp_num_smart_meters_cost + combos_consider.at[temp_bldg,'num_smart_meters'] 
                        temp_num_members                = temp_num_members + 1                                                          #always add 1 in this case as 1 agent will be considered here
                        temp_name                       = temp_name + temp_combos_list[i][j]+ '_'
                        temp_name_comms                 = temp_name 
                        
                    except KeyError: 
                        #keyError will only occur in case an existing community *OR*
                        #bldg with installed PV is one of the choices for the agent,
                        #hence read info from the combos dataframes
                        
                        #*exisiting community is an option*
                        if temp_bldg[0] == 'C' and temp_bldg != '': 
                            if len(temp_combos_list[i]) != 2:
                                #case in which new community is being formed with multiple new agents and an exisiting community.
                                #everyone installs solar on their roofs - all PROSUMERS
                                temp_bldg                       = temp_combos_list[i][j] 
                                temp_bldg_comm_contain          = temp_combos_list[i][j] 
                                temp_bldg_comm                  = temp_bldg_comm_contain.split(sep = '_')
                                temp_bldg_comm.remove('C')
                                temp_bldg_og_name_list.append(temp_bldg_comm)
                                #temp_bldg_og_name_list.extend([temp_bldg_comm])
                                print(temp_bldg)
                                temp_bldg_zones_list.extend(Combos_formed_Info.loc[temp_bldg]['combos_zone_ids'])
                                temp_solar                      = temp_solar + df_solar_combos_main[temp_bldg]
                                temp_demand                     = temp_demand + df_demand_combos_main[temp_bldg]
                                temp_pv_size                    = temp_pv_size + Combos_formed_Info.at[temp_bldg,'combos_pv_size_kw']
                                temp_pv_size_cost               = temp_pv_size_cost #since cost is only incurred for newly installed PV  - avoiding counting size for already installed PV by the existing community
                                temp_join_community             = 1
                                temp_num_smart_meters           = temp_num_smart_meters + Combos_formed_Info.at[temp_bldg,'combos_num_smart_meters']
                                temp_num_smart_meters_cost      = temp_num_smart_meters_cost #since cost is only incurred for newly installed PV  - avoiding counting smartmeters for already installed meters by the existing community
                                temp_num_members                = temp_num_members + Combos_formed_Info.at[temp_bldg,'Num_Members']#len(temp_combos_list[i])#temp_num_members + combos_consider.loc[temp_bldg]['']
                                temp_name                       = temp_name + temp_combos_list[i][j]+ '_'
                                temp_name_comms                 = temp_name
                            
                            
                            elif len(temp_combos_list[i]) == 2:
                                #THIS CAN BE REMOVED
                                #IT IS THE SAME AS THE len !=2 CASE JUST ABOVE
                                #ONLY NEED TO CHECK IF SOME ADDITIONAL INFO IS
                                #NEEDED IN THE RANKING COMBOS FUNCTION
                                
                                #case in which only the activated agent and an exisiting community is present
                                #***agent will also install solar on his own roof - PROSUMER***
                                temp_bldg                       = temp_combos_list[i][j] 
                                temp_bldg_comm_contain          = temp_combos_list[i][j] 
                                temp_bldg_comm                  = temp_bldg_comm_contain.split(sep = '_')
                                temp_bldg_comm.remove('C')
                                temp_bldg_og_name_list.append(temp_bldg_comm[0])
                                #temp_bldg_og_name_list.extend([temp_bldg_comm])
                                
                                #check if temp_bldg must be used in next line or
                                #temp_bldg_comm[0]
                                temp_bldg_zones_list.extend(Combos_formed_Info.at[temp_bldg,'combos_zone_ids'])
                                temp_solar                      = temp_solar + df_solar_combos_main[temp_bldg]
                                temp_demand                     = temp_demand + df_demand_combos_main[temp_bldg]
                                temp_pv_size                    = temp_pv_size + Combos_formed_Info.at[temp_bldg,'combos_pv_size_kw']
                                temp_pv_size_cost               = temp_pv_size_cost #since cost is only incurred for newly installed PV - avoiding counting size for already installed PV by the existing community
                                temp_join_community             = 1
                                temp_num_smart_meters           = temp_num_smart_meters + Combos_formed_Info.at[temp_bldg,'combos_num_smart_meters']
                                temp_num_smart_meters_cost      = temp_num_smart_meters_cost #since cost is only incurred for newly installed PV  - avoiding counting smartmeters for already installed meters by the existing community
                                temp_num_members                = temp_num_members + Combos_formed_Info.at[temp_bldg,'Num_Members']#len(temp_combos_list[i])#temp_num_members + combos_consider.loc[temp_bldg]['']
                                temp_name                       = temp_name + temp_combos_list[i][j]+ '_' #removed Pros from the start of the name
                                #make something to save this community name so that later it is known who is forming with community so the coop costs can be accounted for properly
                                temp_name_comms                 = temp_name
                                
                         #*existing individually installed PV is an option*
                        if temp_bldg[0] == 'P':
                            print("case with PV existing")
                            if len(temp_combos_list[i]) != 2:
                                #case in which new community is being formed with multiple new agents and an exisiting community.
                                #***everyone installs solar on their roofs - all PROSUMERS***
                                temp_bldg                       = temp_combos_list[i][j] 
                                temp_bldg_comm_contain          = temp_combos_list[i][j] 
                                temp_bldg_comm                  = temp_bldg_comm_contain.split(sep = '_')
                                temp_bldg_comm.remove('PV')
                                #temp_bldg_og_name_list.extend([temp_bldg_comm])
                                print('temp_bldg_comm = ',temp_bldg_comm[0])
                                temp_bldg_og_name_list.append(temp_bldg_comm[0])
                                temp_bldg_zones_list.extend(combos_consider.at[temp_bldg_comm[0],'zone_id'])
                                print("temp_bldg in k = 1 = ", temp_bldg)
                                temp_bldg_name_edited           = str.strip(temp_bldg, 'PV_')
                                temp_solar                      = temp_solar + df_solar[temp_bldg_name_edited]
                                temp_demand                     = temp_demand + df_demand[temp_bldg_name_edited]
                                temp_pv_size                    = temp_pv_size + combos_consider.at[temp_bldg_name_edited,'pv_size_kw']
                                temp_pv_size_cost               = temp_pv_size_cost #since cost is only incurred for newly installed PV - avoiding counting size for already installed PV by the existing community
                                temp_join_individual            = 1
                                temp_num_smart_meters           = temp_num_smart_meters + combos_consider.at[temp_bldg_name_edited,'num_smart_meters']
                                temp_num_smart_meters_cost      = temp_num_smart_meters_cost #since cost is only incurred for newly installed PV  - avoiding counting smartmeters for already installed meters by the existing community
                                temp_num_members                = temp_num_members + 1 #just add 1 coz only 1 agent will be considered here. 
                                temp_name                       = temp_name + temp_combos_list[i][j]+ '_'
                                temp_name_comms                 = temp_name
                                
                            
                            elif len(temp_combos_list[i]) == 2:
                                #case in which only the activated agent and an exisiting individually adopted agent is present
                                #***agent will also install solar on his own roof - PROSUMER***
                                temp_bldg                       = temp_combos_list[i][j] 
                                print(temp_bldg)
                                temp_bldg_comm_contain          = temp_combos_list[i][j] 
                                temp_bldg_comm                  = temp_bldg_comm_contain.split(sep = '_')
                                temp_bldg_comm.remove('PV')
                                temp_bldg_og_name_list.append(temp_bldg_comm[0])
                                temp_bldg_zones_list.extend(combos_consider.at[temp_bldg_comm[0],'zone_id'])
                                temp_bldg_name_edited           = str.strip(temp_bldg, 'PV_')
                                temp_solar                      = temp_solar + df_solar[temp_bldg_name_edited]
                                temp_demand                     = temp_demand + df_demand[temp_bldg_name_edited]
                                temp_pv_size                    = temp_pv_size + combos_consider.at[temp_bldg_name_edited,'pv_size_kw'] #CHECK!!
                                temp_pv_size_cost               = temp_pv_size_cost #since cost is only incurred for newly installed PV - avoiding counting size for already installed PV by the existing community
                                temp_join_individual            = 1
                                temp_num_smart_meters           = temp_num_smart_meters + combos_consider.at[temp_bldg_name_edited,'num_smart_meters']
                                temp_num_smart_meters_cost      = temp_num_smart_meters_cost #since cost is only incurred for newly installed PV  - avoiding counting smartmeters for already installed meters by the existing community
                                temp_num_members                = len(temp_combos_list[i])      
                                temp_name                       = temp_name + temp_combos_list[i][j]+ '_' #removed Pros from the start of the name
                                temp_name_comms                 = temp_name
                                
                                
                #populating the dataframes in the usual case - no previously installed PV systems           
                #also populating the dataframes in the existing PV case - with many other agents so all all PROSUMERS
                df_solar_combo[temp_name]               = ""
                df_solar_combo[temp_name]               = temp_solar
                df_demand_combo[temp_name]              = ""
                df_demand_combo[temp_name]              = temp_demand
                temp_pv_list.append(temp_pv_size)
                temp_pv_cost_list.append(temp_pv_size_cost)
                temp_bldg_og_name_list_df.append(temp_bldg_og_name_list)
                #temp_bldg_og_name_list_df.append(list(temp_combos_list[i]))
                temp_bldg_zones_list_df.append(temp_bldg_zones_list)
                temp_join_individual_list.append(temp_join_individual)
                temp_join_community_list.append(temp_join_community)
                temp_num_smartmeters_list.append(temp_num_smart_meters)
                temp_num_smartmeters_cost_list.append(temp_num_smart_meters_cost)
                temp_num_members_list.append(temp_num_members)
                df_pvsize_combo[temp_name]              = ""
                df_pvsize_combo[temp_name]              = temp_pv_list
                df_pvsize_cost_combo[temp_name]         = ""
                df_pvsize_cost_combo[temp_name]         = temp_pv_cost_list
                df_bldgs_names[temp_name]               = ""
                df_bldgs_names[temp_name]               = temp_bldg_og_name_list_df  
                df_zones_names[temp_name]               = ""
                df_zones_names[temp_name]               = temp_bldg_zones_list_df
                df_join_individual[temp_name]           = ""
                df_join_community[temp_name]            = ""
                df_join_individual[temp_name]           = temp_join_individual_list
                df_join_community[temp_name]            = temp_join_community_list
                df_num_smart_meters[temp_name]          = ""
                df_num_smart_meters[temp_name]          = temp_num_smartmeters_list
                df_num_smart_meters_cost[temp_name]     = ""
                df_num_smart_meters_cost[temp_name]     = temp_num_smartmeters_cost_list
                df_num_members[temp_name]               = ""
                df_num_members[temp_name]               = temp_num_members_list
                temp_names_comms_list.append(temp_name_comms)
               
        #print('sum of solar dataframe for the last combo = ',sum(df_solar_combo[temp_name]))
        #print('$$$$$bldg names dataframe =',df_bldgs_names)
        #print(df_bldgs_names.index)
        #print(df_bldgs_names.columns)
        #print(df_bldgs_names.loc['Bldg_Names'])
        #print('zones df')
        #print(df_zones_names.loc['Zone_Names'])
        
            NPV_combos = pd.DataFrame(data = None)
            
            year = year     #step_ctr from the ABM gives the current year of simulation in the model. 0 = 2018, 1 = 2019 and so on...
            
            #this calculates NPVs for all possible combinations
            #print("uid = ", uid)
            #print("before sending to NPV combos, columns of df_solar = ", df_solar_combo.columns)
            #print(temp_names_comms_list)
            
            NPV_combos = npv_combo.npv_calc_combos(df_solar_combo, df_demand_combo, year,
                                         data.at[uid,"bldg_owner"],
                                         df_pvsize_combo, df_pvsize_cost_combo,
                                         df_num_smart_meters, df_num_smart_meters_cost,
                                         df_num_members,disc_rate,
                                         fit_high, fit_low,
                                         ewz_high_large,ewz_low_large,
                                         ewz_high_small, ewz_low_small,
                                         ewz_solarsplit_fee,
                                         PV_lifetime, PV_degradation,
                                         OM_Cost_rate,PV_price_projection,
                                         list_hours, daylist,diff_prices)
            #this ranks the NPVs and then returns the best NPV. If no combination is possible then an empty dataframe is returned.
            
            Combos_Info = rank_combos.ranking_combos(NPV_combos, df_demand, combos_consider,
                                         df_join_individual, df_join_community,
                                         df_bldgs_names, df_zones_names,
                                         Agents_Ind_NPVs,year)
            
            #so that index of the Combos_Info dataframe is renamed with a C as prefix 'C_B147891_B147892_
            if len(Combos_Info.index) > 0:
                temp_name = Combos_Info.index[0] #eg = 'B147891_B147892_'
                x = temp_name.split('_')
                y = 'C'
                for i in x:
                    y = y + '_' +  i
                Combos_Info = Combos_Info.rename(index={temp_name: y})
                #temp_name = Combos_Info.index[0] #eg = 'C_B147891_B147892_' Renaming done
                Combos_Info['Num_Members'] = (len(x) -1)                
            else:
                temp_name = ''
                
        elif len(temp_combos_list) == 0:
            Combos_Info     = pd.DataFrame(data = None)
            NPV_combos      = pd.DataFrame(data = None) 
            df_solar_combo  = pd.DataFrame(data = None)
            df_demand_combo = pd.DataFrame(data = None)
            temp_name       = ''
    
    elif len(combos_consider.index) == 0:
        #meaning no combos can be formed as no neighbour has positive intention
        Combos_Info     = pd.DataFrame(data = None)
        NPV_combos      = pd.DataFrame(data = None) 
        df_solar_combo  = pd.DataFrame(data = None)
        df_demand_combo = pd.DataFrame(data = None)
        temp_name       = ''
    
    
    return Combos_Info, NPV_combos, df_solar_combo, df_demand_combo, temp_name
            
        
        
                  
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
        
 