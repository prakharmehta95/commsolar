# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 14:06:29 2019

@author: iA
"""
#%%


def ranking_combos(NPV_combos, df_demand, combos_consider, df_join_individual, df_join_community, df_bldgs_names):
    print("ranking entered")
    import pandas as pd
    npv_max_temp = NPV_combos.copy()#.transpose()
    npv_max_temp =npv_max_temp.sort_values(by = ['npv'], ascending = False)
    npvs_max = npv_max_temp.head(3).copy() #top 3 combinations
    
    #add the constituent building info to the npvs_max dataframe
    npvs_max['Bldg_Names'] = ""
    for i in npvs_max.index:
        npvs_max.at[i,'Bldg_Names'] = df_bldgs_names.loc['Bldg_Names'][i]
    
    #print(npv_max)
    print("npvs max dataframe = ",npvs_max)
    bldgs_comm_bestnpv = str.split(npvs_max.index[0],'_') #split string to get name of the buildings in the community
    while '' in bldgs_comm_bestnpv:
        bldgs_comm_bestnpv.remove('') #get the buildings of community in a list
    en_champ_agent = bldgs_comm_bestnpv[0]
    '''
    now try to compare npv shares for all buildings in the community
    '''
    #first create the npv shares in an accessible dataframe...
    
    '''
    now try to compare npv shares for all buildings in the community
    in a loop, test all combos, starting from the best npv...if a community is formed then exit the loop
    '''
    npvs_max['all_agree'] = ""
    for i in npvs_max.index:
        print(i)
        bldgs_comm_bestnpv = str.split(i,'_') #split string to get name of the buildings in the community - this will fuck everything up!!!
        while '' in bldgs_comm_bestnpv:
            bldgs_comm_bestnpv.remove('') #get the buildings of community in a list
        print("bldgs_comm_best_npv = ",bldgs_comm_bestnpv)
        #first create the npv shares in an accessible dataframe...
        df_npv_shares = pd.DataFrame(data = None, index = bldgs_comm_bestnpv, columns = ['yearly_demand','npv_share','npv_share_mag'])
        print('$$$$', df_npv_shares)
        for j in bldgs_comm_bestnpv:
            print(j)
            df_npv_shares.at[j,'yearly_demand'] = sum(df_demand[j])
        for j in bldgs_comm_bestnpv:
            print(j)
            df_npv_shares.at[j,'npv_share'] = df_npv_shares.at[j,'yearly_demand']/sum(df_npv_shares['yearly_demand'])
            df_npv_shares.at[j,'npv_share_mag'] = df_npv_shares.at[j,'npv_share']*NPV_combos_temp.loc[npvs_max.index[0]]['npv']
        print('$$$$', df_npv_shares)
        
        #need to store these NPV shares so that they can be used later to compare in case an agent wants to join an existing community
        
        #actual loop for comparison with individual NPVs
        ctr = 0
        #ccc = []
        for j in bldgs_comm_bestnpv:
            if -2000578 < df_npv_shares.loc[j]['npv_share_mag']:#Agents_Ind_NPVs.loc[year][i] > df_npv_shares.loc[i]['npv_share_mag']:
                ctr = ctr + 1
                print('ctr = ', ctr)
            if ctr == len(bldgs_comm_bestnpv):
                #for all buildings the community npv is better than the individual npv
                # --> form the community, exit this process
                #ccc.append(ctr)
                npvs_max.at[i,'all_agree'] = 'Y'
            elif ctr != len(bldgs_comm_bestnpv):
                npvs_max.at[i,'all_agree'] = 'N'
                
    #again sort in descending order
    npvs_max = npvs_max.sort_values(by = ['npv'], ascending = False)
    
    #
    if npvs_max.loc[npvs_max.index[0]]['all_agree'] == 'Y':
        print('condition 1')
        community = npvs_max.index[0]
    elif (npvs_max.loc[npvs_max.index[0]]['all_agree'] != 'N') and (npvs_max.loc[npvs_max.index[1]]['all_agree'] == 'Y'):
        print('condition 2')
        community = npvs_max.index[1]
    elif (npvs_max.loc[npvs_max.index[0]]['all_agree'] != 'N') and (npvs_max.loc[npvs_max.index[1]]['all_agree'] != 'Y') and (npvs_max.loc[npvs_max.index[2]]['all_agree'] == 'Y'):
        print('condition 3')
        community = npvs_max.index[2]
    #elif for more than 3 conditions...
    else: 
        community = ""
            
    print("community = ",community)
    comm_bldgs = str.split(community,'_') #split string to get name of the buildings in the community
    comm_bldgs.remove('')
    
    temp_bldg_names     = ""
    temp_egids          = ""
    temp_types          = ""
    temp_owners         = ""
    temp_zone_ids       = ""
    temp_plot_ids       = ""
    temp_total_persons  = 0
    temp_num_sm_meters  = 0
    temp_roof_area      = 0
    temp_pv_yearly      = 0
    temp_pv_size        = 0
    temp_pv_category    = ""
    temp_pv_sub         = 0
    temp_dem            = 0
    temp_dem_area       = 0
    
    combos_info = pd.DataFrame(data = None, index = [community])
    
    for i in comm_bldgs:
        #maybe here add a field so that the building names are also saved
        temp_bldg_names = temp_bldg_names + i + '_'
        temp_egids = temp_egids + str(combos_consider.loc[i]['bldg_EGID']) + '_'
        temp_types = temp_types + combos_consider.loc[i]['bldg_type'] + '_'
        temp_owners = temp_owners + combos_consider.loc[i]['bldg_owner'] + '_'
        temp_zone_ids = temp_zone_ids + combos_consider.loc[i]['zone_id'] + '_'
        temp_plot_ids = temp_plot_ids + combos_consider.loc[i]['plot_id'] + '_'
        temp_total_persons = temp_total_persons + combos_consider.loc[i]['total_persons']
        temp_num_sm_meters = temp_num_sm_meters + combos_consider.loc[i]['num_smart_meters']
        temp_roof_area = temp_roof_area + combos_consider.loc[i]['roof_area_m2']
        temp_pv_yearly = temp_pv_yearly + combos_consider.loc[i]['pv_yearly_kWh']
        temp_pv_size = temp_pv_size + combos_consider.loc[i]['pv_size_kw']
        #temp_pv_category = temp_pv_category + combos_consider.loc[i]['total_persons']
        #temp_pv_sub = temp_pv_sub + combos_consider.loc[i]['total_persons']
        temp_dem = temp_dem + combos_consider.loc[i]['demand_yearly_kWh']
        print("temp_dem at comm_bldg = ",i,temp_dem)
        temp_dem_area = temp_dem_area + combos_consider.loc[i]['demand_areas_nutzflache_m2']
    
    combos_info['combos_bldg_names'] = temp_bldg_names
    combos_info['combos_EGIDs'] = temp_egids
    combos_info['combos_types'] = temp_types
    combos_info['combos_owners'] = temp_owners
    combos_info['combos_zone_ids'] = temp_zone_ids
    combos_info['combos_plot_ids'] = temp_plot_ids
    combos_info['combos_total_persons'] = temp_total_persons
    combos_info['combos_num_smart_meters'] = temp_num_sm_meters
    combos_info['combos_roof_area_m2'] = temp_roof_area
    combos_info['combos_pv_yearly_kWh'] = temp_pv_yearly
    combos_info['combos_pv_size_kw'] = temp_pv_size
    #combos_info['combos_pv_size_category'] = 
    #combos_info['combos_pv_subsidy'] = 
    combos_info['combos_demand_yearly_kWh'] = temp_dem
    combos_info['combos_demand_areas_m2'] = temp_dem_area
    combos_info['npv_share_en_champ'] = df_npv_shares.loc[en_champ_agent]['npv_share_mag']
    
    return combos_info
