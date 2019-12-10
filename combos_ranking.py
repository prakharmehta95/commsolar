# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 14:06:29 2019

@author: iA
"""
#%%


def ranking_combos(NPV_combos, df_demand, combos_consider, df_join_individual, df_join_community, df_bldgs_names, df_zones_names):
    
    '''
    Returns the best community combination by ranking the top three possible community formations
    
    NPV_combos          = NPVs of all possible community combinations
    df_demand           = Individual building demands
    combos_consider     = Info on the individual buildings being considered to form a community with
    df_join_individual  = Info on what community is formed with existing individual PV system. 1 = with existing PV; 0 = without existing PV
    df_join_community   = Info on what community is formed with existing communty PV system. 1 = with existing PV; 0 = without existing PV
    df_bldgs_names      = Info on names of building in the possible community formation
    df_zones_names      = Info on zones of the buildings in the possible community formations
    '''
    
    print("ranking entered")
    import pandas as pd
    npv_max_temp    = NPV_combos.copy()#.transpose()
    npv_max_temp    = npv_max_temp.sort_values(by = ['npv'], ascending = False)
    npvs_max        = npv_max_temp.head(3).copy() #top 3 combinations
    
    #add the constituent building info to the npvs_max dataframe
    npvs_max['Bldg_Names'] = ""
    npvs_max['Zone_Names'] = ""
    for i in npvs_max.index:
        npvs_max.at[i,'Bldg_Names'] = df_bldgs_names.loc['Bldg_Names'][i]
        npvs_max.at[i,'Zone_Names'] = df_zones_names.loc['Zone_Names'][i]
    
    
    #now try to compare npv shares for all buildings in the community
    npvs_max['all_agree']       = ""
    npvs_max['all_same_zones']  = ""
    for i in npvs_max.index:
        print("i = ",i)
        #first create the npv shares in an accessible dataframe...
        df_npv_shares = pd.DataFrame(data = None, index = npvs_max.loc[i]['Bldg_Names'], columns = ['yearly_demand','npv_share','npv_share_mag'])
        for j in npvs_max.loc[i]['Bldg_Names']:
            df_npv_shares.at[j,'yearly_demand'] = sum(df_demand[j]) #original individual demand
        for j in npvs_max.loc[i]['Bldg_Names']:
            df_npv_shares.at[j,'npv_share'] = df_npv_shares.at[j,'yearly_demand']/sum(df_npv_shares['yearly_demand'])
            df_npv_shares.at[j,'npv_share_mag'] = df_npv_shares.at[j,'npv_share']*NPV_combos.loc[npvs_max.index[0]]['npv']
        
        
        #need to store these NPV shares so that they can be used later to compare in case an agent wants to join an existing community
        
        #actual loop for comparison with individual NPVs
        ctr = 0
        dummy_for_actual_npv = -2020202 #will be read from Agents_Ind_NPVs.loc[year][i]!!!****
        #ccc = []
        for j in npvs_max.loc[i]['Bldg_Names']:
            if dummy_for_actual_npv < df_npv_shares.loc[j]['npv_share_mag']: #IT MUST BE THIS WAY - if Agents_Ind_NPVs.loc[year][i] > df_npv_shares.loc[i]['npv_share_mag']:
                ctr = ctr + 1
            
            if ctr == len(npvs_max.loc[i]['Bldg_Names']): #this means that for all buildings the community npv is better than the individual npv
                npvs_max.at[i,'all_agree'] = 'Y'
            elif ctr != len(npvs_max.loc[i]['Bldg_Names']):
                npvs_max.at[i,'all_agree'] = 'N'
        
        #this if-else is to note if all buildings in the community are in the same zone or not
        temp_list = npvs_max.loc[i]['Zone_Names']
        if temp_list.count(temp_list[0]) == len(temp_list):
            npvs_max.at[i,'all_same_zones'] = 'Y'
        else:
            npvs_max.at[i,'all_same_zones'] = 'N'
                
    #finding the best community combination; best = highest positive NPV and all have said yes and all buildings in the same zone
    npvs_max_best_temp  = npvs_max.loc[npvs_max.all_agree == 'Y']
    npvs_max_best_temp  = npvs_max_best_temp.sort_values(by = ['npv'], ascending = False)
    npvs_max_best       = npvs_max_best_temp.loc[npvs_max_best_temp.all_same_zones == 'Y']
    npvs_max_best       = npvs_max_best.sort_values(by = ['npv'], ascending = False)
    if len(npvs_max_best.index) > 0:
        community = npvs_max_best.index[0]      #best npv, all agree, same zones
    elif len(npvs_max_best_temp.index) > 0:
        community = npvs_max_best_temp.index[0] #best npv, all agree, different zones
    else:
        community = ""                          #no community is formed
    
    #storing information on the community formed
    if community != "":
        comm_bldgs      = npvs_max_best.loc[community]['Bldg_Names']#str.split(community,'_') #split string to get name of the buildings in the community
        en_champ_agent  = comm_bldgs[0] #the first building in the community is the energy champion
    
        if len(comm_bldgs) > 0: #meaning that there is indeed a community formed. Else comm_bldgs = 0
            temp_bldg_names         = []
            temp_bldg_names_list    = []
            temp_egids              = []
            temp_egids_list         = []
            temp_types_list         = []
            temp_types              = []
            temp_owners_list        = []
            temp_owners             = []
            temp_zone_ids           = []
            temp_zone_ids_list      = []
            temp_plot_ids           = []
            temp_plot_ids_list      = []
            temp_total_persons      = 0
            temp_num_sm_meters      = 0
            temp_roof_area          = 0
            temp_pv_yearly          = 0
            temp_pv_size            = 0
            temp_dem                = 0
            temp_dem_area           = 0
            
            combos_info = pd.DataFrame(data = None, index = [community])
            
            for i in comm_bldgs:
                #maybe here add a field so that the building names are also saved
                temp_bldg_names.append(i)
                temp_egids.append(combos_consider.loc[i]['bldg_EGID']) 
                temp_types.append(combos_consider.loc[i]['bldg_type'])
                temp_owners.append(combos_consider.loc[i]['bldg_owner'])
                temp_zone_ids.append(combos_consider.loc[i]['zone_id'])
                temp_plot_ids.append(combos_consider.loc[i]['plot_id'])
                temp_total_persons  = temp_total_persons + combos_consider.loc[i]['total_persons']
                temp_num_sm_meters  = temp_num_sm_meters + combos_consider.loc[i]['num_smart_meters']
                temp_roof_area      = temp_roof_area + combos_consider.loc[i]['roof_area_m2']
                temp_pv_yearly      = temp_pv_yearly + combos_consider.loc[i]['pv_yearly_kWh']
                temp_pv_size        = temp_pv_size + combos_consider.loc[i]['pv_size_kw']
                temp_dem            = temp_dem + combos_consider.loc[i]['demand_yearly_kWh']
                temp_dem_area       = temp_dem_area + combos_consider.loc[i]['demand_areas_nutzflache_m2']
            
            temp_bldg_names_list.append(temp_bldg_names)
            temp_egids_list.append(temp_egids)
            temp_types_list.append(temp_types)
            temp_owners_list.append(temp_owners)
            temp_zone_ids_list.append(temp_zone_ids)
            temp_plot_ids_list.append(temp_plot_ids)
            combos_info['combos_bldg_names']        = temp_bldg_names_list
            combos_info['combos_EGIDs']             = temp_egids_list
            combos_info['combos_types']             = temp_types_list
            combos_info['combos_owners']            = temp_owners_list
            combos_info['combos_zone_ids']          = temp_zone_ids_list
            combos_info['combos_plot_ids']          = temp_plot_ids_list
            combos_info['combos_total_persons']     = temp_total_persons
            combos_info['combos_num_smart_meters']  = temp_num_sm_meters
            combos_info['combos_roof_area_m2']      = temp_roof_area
            combos_info['combos_pv_yearly_kWh']     = temp_pv_yearly
            combos_info['combos_pv_size_kw']        = temp_pv_size
            combos_info['combos_demand_yearly_kWh'] = temp_dem
            combos_info['combos_demand_areas_m2']   = temp_dem_area
            combos_info['npv_share_en_champ']       = df_npv_shares.loc[en_champ_agent]['npv_share_mag']
        else:
            combos_info = pd.DataFrame(data = None)#, index = [community]) #empty combos info dataframe
    
    elif community == "":
        combos_info = pd.DataFrame(data = None)#, index = [community]) #empty combos info dataframe
    
    return combos_info
