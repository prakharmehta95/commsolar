# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 14:06:29 2019

@author: iA
"""
#%%


def ranking_combos(NPV_combos, df_demand, combos_consider, df_join_individual, df_join_community, df_bldgs_names, df_zones_names, Agents_Ind_NPVs,year):
    
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
    
    import pandas as pd
    df_join_individual  = df_join_individual.transpose()
    df_join_community   = df_join_community.transpose()
    
    NPV_combos['Join_Exist_Ind']    = df_join_individual['Join_Ind']
    NPV_combos['Join_Exist_Comm']   = df_join_community['Join_Comm'] 
    
    npv_max_temp    = NPV_combos.copy()
    npv_max_temp    = npv_max_temp.sort_values(by = ['npv'], ascending = False)
    npvs_max        = npv_max_temp.head(3).copy() #top 3 combinations
    #print(npv_max_temp)
    
    #add the constituent building info to the npvs_max dataframe
    npvs_max['Bldg_Names'] = ""
    npvs_max['Zone_Names'] = ""
    for i in npvs_max.index:
        npvs_max.at[i,'Bldg_Names'] = df_bldgs_names.loc['Bldg_Names'][i]
        npvs_max.at[i,'Zone_Names'] = df_zones_names.loc['Zone_Names'][i]
    
    
    #now try to compare npv shares for all buildings in the community
    npvs_max['all_agree']       = ""
    npvs_max['all_same_zones']  = ""
    npvs_max['npv_share_en_champ']  = ""
    
    print('npvs_max = ',npvs_max)
    #print('npvs_max columns = ',npvs_max.columns)
    #print('npvs_max index = ',npvs_max.index)
    #print('npvs_max Bldg Names column = ',npvs_max.Bldg_Names)
    print(df_bldgs_names)
    
    for i in npvs_max.index:
        #dataframe to hold npv shares of the agents temporarily
        df_npv_shares = pd.DataFrame(data = None, index = npvs_max.loc[i]['Bldg_Names'], columns = ['yearly_demand','npv_share','npv_share_mag'])
        print('i = ',i)
        for j in npvs_max.loc[i]['Bldg_Names']:
            print('j = ',j)
            if npvs_max.loc[i]['Join_Exist_Ind'] == 1 or npvs_max.loc[i]['Join_Exist_Comm'] == 1:
                #existing individual PV or community
                #this means that as they have already installed PV, for the new
                #PV system, they do not pay any money.
                #so the npv shares must not include them
                #hence assign it zero for that building/group of buildings in
                #case of a community
                df_npv_shares.at[j,'yearly_demand'] = 0
            else:
                #no existing individual PV or community
                #normal case 
                df_npv_shares.at[j,'yearly_demand'] = sum(df_demand[j]) #original individual demand
        
        en_champ_temp = npvs_max.loc[i]['Bldg_Names'][0]
        
        for j in npvs_max.loc[i]['Bldg_Names']:
            try:
                #npv shares for all buildings
                df_npv_shares.at[j,'npv_share'] = df_npv_shares.at[j,'yearly_demand']/sum(df_npv_shares['yearly_demand'])
                #use the npv share to calculate the magnitude of the npv share
                df_npv_shares.at[j,'npv_share_mag'] = df_npv_shares.at[j,'npv_share']*NPV_combos.loc[i]['npv']
            except ZeroDivisionError:
                #for now so that it runs. Check properly later - I know the problem!:
                #occurs because some demands are zero
                #those buildings need to be removed from the datasets
                df_npv_shares.at[j,'npv_share']     = 0 #df_npv_shares.at[j,'yearly_demand']/sum(df_npv_shares['yearly_demand'])
                df_npv_shares.at[j,'npv_share_mag'] = 0 #df_npv_shares.at[j,'npv_share']*NPV_combos.loc[npvs_max.index[0]]['npv']
        
        #saving the npv share for the energy champiion agent
        npvs_max.at[i,'npv_share_en_champ'] = df_npv_shares.loc[en_champ_temp]['npv_share_mag']
        
        #need to store these NPV shares so that they can be used later to compare
        #in case an agent wants to join an existing community
        
        #@Alejandro - I have not stored the NPV shares for all agents when 
        #they form a community. So when an agent wishes to join an exisiting
        #community, the agents within the existing community again compare
        #with their individual  NPVs.this may lead to a worse community
        #combination. The only way this is 
        #avoided in the current script is that the energy champion's NPV is 
        #actually saved, so at least the energy champion agent can compare 
        #new community npv share with old community npv share. 
        #This will still need to be added in the following lines here as an 
        #alternative if-else statement!
        
        #update on the previous point:
        #I decided to not do it that the energy champion of exisiting 
        #community compares his old share with his new share. I have another 
        #reasoning, as explained below
        
        #loop for comparison with individual NPVs
        ctr = 0
        for j in npvs_max.loc[i]['Bldg_Names']:
            #if NPV share and individual NPV are equal, we give preference to
            #community formation. Hence the lessthan-equal to sign is used
            if npvs_max.loc[i]['Join_Exist_Ind'] != 1 and npvs_max.loc[i]['Join_Exist_Comm'] != 1: 
                #no existing individual PV or community PV
                if Agents_Ind_NPVs.loc[year][j] <= df_npv_shares.loc[j]['npv_share_mag']:
                    #NPV shares in the community are better than Individual PV
                    ctr += 1
            elif npvs_max.loc[i]['Join_Exist_Ind'] == 1:
                #case when there is an existing individual PV in the prospective
                #community. Then, for that individual building with existing PV,
                #just say YES - ASSUMPTION!
                #Ideally - the bldg with PV installed does not pay any money
                #for the others installing a community PV system
                #In that case, bldg with installed PV must reassess how much 
                #electricity is sold to the new community instead of grid.
                #Complicated problem and needs recalculation of NPV for that
                #bldg again. However, at least in CH, FiT is low and we can 
                #assume that this bldg can sell to the community for a price
                #higher than the FiT. So exisiting individual PV will always 
                #say YES to being part of a community.
                #no need to compare with any NPV here hence the elif statement 
                #has no NPV term. 
                ctr += 1
                
            elif npvs_max.loc[i]['Join_Exist_Comm'] == 1:
                #case when there is an existing individual PV in the prospective
                #community. Then, for that individual building with existing PV,
                #just say YES - ASSUMPTION!
                #Again, same argument as above
                ctr += 1
                
        #assigning 'Y' or 'N' if all agents agree i.e NPV share better or worse than individual NPV
        if ctr == len(npvs_max.loc[i]['Bldg_Names']): #this means that for all buildings the community npv is better than the individual npv
            npvs_max.at[i,'all_agree'] = 'Y'
        elif ctr != len(npvs_max.loc[i]['Bldg_Names']):
            npvs_max.at[i,'all_agree'] = 'N'
    
        #note if all buildings in the community are in the same zone or not
        temp_list = npvs_max.loc[i]['Zone_Names']
        if temp_list.count(temp_list[0]) == len(temp_list):
            npvs_max.at[i,'all_same_zones'] = 'Y'
        else:
            npvs_max.at[i,'all_same_zones'] = 'N'
    
    
    
    #finding the best community combination; best = highest positive NPV and all have said yes and all buildings in the same zone
    npvs_max_best_temp  = npvs_max.loc[npvs_max.all_agree == 'Y']
    npvs_max_best_temp  = npvs_max_best_temp.sort_values(by = ['npv'], ascending = False) #holds combinations in which all community members agree AND all are in different zones
    npvs_max_best       = npvs_max_best_temp.loc[npvs_max_best_temp.all_same_zones == 'Y']
    npvs_max_best       = npvs_max_best.sort_values(by = ['npv'], ascending = False) #holds combinations in which all community members agree AND all are in the same zone
    diff_zones = 0 #initializing a variable for info on whether all bldgs in same zone or not 
    if len(npvs_max_best.index) > 0:
        diff_zones = 0
        community = npvs_max_best.index[0]      #best npv, all agree, same zones
    elif ((len(npvs_max_best.index) == 0) and (len(npvs_max_best_temp.index) > 0)) == 1:
        diff_zones = 1
        community = npvs_max_best_temp.index[0] #best npv, all agree, different zones
    else:
        community = ""                          #no community is formed
        
    #storing information on the community formed
    if community != "":
        if diff_zones == 0:
            comm_bldgs      = npvs_max_best.loc[community]['Bldg_Names']    #get names of the buildings in the community
        elif diff_zones == 1:
            comm_bldgs      = npvs_max_best_temp.loc[community]['Bldg_Names']    #get names of the buildings in the community
        en_champ_agent  = comm_bldgs[0]                                 #the first building in the community is the energy champion
    
        if len(comm_bldgs) > 0: 
            #meaning that there is indeed a community formed. Else comm_bldgs = 0
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
                #print('i just before the error = ',i)
                if i[0] == 'P':
                    #building has exisitng PV hence it's name starts with PV_B123456
                    i = i.split('_')
                    i.remove('PV')
                    i = i[0]
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
            combos_info['npv_share_en_champ']       = npvs_max.loc[community]['npv_share_en_champ']
        else:
            combos_info = pd.DataFrame(data = None)#, index = [community]) #empty combos info dataframe
    
    elif community == "":
        combos_info = pd.DataFrame(data = None)#, index = [community]) #empty combos info dataframe
    
    return combos_info
