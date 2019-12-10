# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 14:13:01 2019

@author: iA
"""

#%%
import pandas as pd
def community_combinations(data_og, same_plot_agents_positive_intention, distances, df_solar, df_demand, df_solar_combos_main , df_demand_combos_main, Combos_formed_Info, uid, year):
    
    '''
    data                                = agents_info - Info dataframe on all agents. Contains who passed the intention stage, who has individal/community PV, IDs of the adopted PV systems  
    same_plot_agents_positive_intention = all agents in the same plot as the active agent who have passed the intention stage
    distances                           = Holds the distances from each agent to its nearest 200 agents (200 agents because dataframe size is reasonable but still satisfies all criteria for the ABM conceptual model)
    df_solar                            = Individual building solar PV potential
    df_demand                           = Individual building demands
    df_solar_combos_main                = Solar PV potential of all possible combinations
    df_demand_combos_main               = Demands of all possible combinations
    Combos_formed_Info                  = Info on already formed combinations
    uid                                 = ID of the activated agent. Eg = B123456
    year                                = Year of the simulation. 0 = 2018, 1 = 2019 and so on.
    '''
    
    data = data_og.copy()

# =============================================================================
#     
#     #since cooperation costs are considered as 0, no need for these now
#
#     admin_costs      = 1000 #CHF, let's say. Read from main ABM
#     rate_cooperation = 50   #CHF per agent, read from main ABM
# 
# =============================================================================

# =============================================================================
#    
#     #this will be done within the ABM. Done now just for testing purposes
#     

      #will read agents_info in place of 'data'
#     #same_plot_agents_positive_intention dataframe can be read directly from within the ABM
#     data['intention'] = ""
#     data['intention'] = [ 1 if i%2 == 1 else 1 for i in range(len(data.index))] # temporarily setting intention so that I can replicate what happens in the ABM
#     
#     data['individual'] = ""
#     data['individual'] =  [ 0 if i%2 == 1 else 0 for i in range(len(data.index))] # temporarily setting adoption so that I can replicate what happens in the ABM
#     data.at['B147890','individual'] = 0
#     data.at['B147891','individual'] = 0
#     data.at['B147892','individual'] = 0
#     #data.at['B147893','individual'] = 0
#     data.at['B2371142','individual'] = 1
#     
#     
#     data['community'] = ""
#     data['community'] =  [ 0 if i%2 == 1 else 0 for i in range(len(data.index))] # temporarily setting community adoption so that I can replicate what happens in the ABM
#     #data.at['B147891','community'] = 1
#     data.at['B147889','community'] = 1
#     data.at['B147890','community'] = 1
#     #data.at['B147893','community'] = 1
#     data.at['B2371142','community'] = 0
#     #data.at['B147889','community'] = 0
#     
#    '''
#     lets say that the filtering is already done and I have the building which is being considered at the moment
#     This will also be done by tge ABM
#     '''
#     data = data.set_index('bldg_name', drop = False)
#     for i in uid:#['B147890']: #data.index: THIS IS THE BUILDING BEING CONSIDERED IN THE ABM VIA THE self.uid...
#         if data.loc[i]['intention'] == 1: #find other buildings in same plot with positive intention
#             temp_plot_id = data.loc[i]['plot_id']
#             same_plot_agents = data.loc[data['plot_id']==temp_plot_id].copy()
#             same_plot_agents_positive_intention = same_plot_agents.loc[same_plot_agents['intention'] == 1].copy() #available to form community
#     #until here will actually be done in the ABM code. Now just so that I can test these functions.
#     #same_plot_agents_positive_intention dataframe can be read directly from within the ABM! - this will be individual buildings with info on individual/community adoptions
#     #------------------------------------------
#     
#     #to find the closest 'n' neighbours available for community formation - even if they have already adopted individual/formed a community?
#     #do not have distance info on the communities which will be formed - then, take distance of the first agent in that community (i.e. the energy champion) - just easier to code this way I think
#     
#    
# =============================================================================
    #EVERYTHING BEFORE^^ IS ONLY FOR TESTING THIS FUNCTION SEPARATELY
    
    no_closest_neighbors_consider   = 4
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
    
    
    #READ FROM THE MAIN ABM A FILE WHICH CONTAINS INFO ON ALREADY FORMED COMMUNITIES, THEN FILTER FOR THE BUILDINGS IN THIS CASE AND ASSIGN TO THE COLUMNS IN combos_consider
    combos_consider['Comm_formed']  = ""
    combos_consider['Ind_formed']   = ""
    combos_consider['Community']  = ""
    combos_consider['Individual']  = ""
    for i in combos_consider.index:
        combos_consider.at[i,'Community']   = data.loc[i]['Adopt_COMM']
        combos_consider.at[i,'Individual']  = data.loc[i]['Adopt_IND']
        combos_consider.at[i,'Comm_formed'] = data.loc[i]['Community_ID']
        combos_consider.at[i,'Ind_formed']  = data.loc[i]['Individual_ID']


#     
# =============================================================================
#     FOR TESTING THIS FUNCTION SEPARATELY
        
#     combos_consider.at['B147890','Comm_formed'] = "C_B147890_B147889_"
#     combos_consider.at['B147889','Comm_formed'] = "C_B147890_B147889_"
#     #combos_consider.at['B147893','Comm_formed'] = "C_B147891_B147893_"
#     #in case there are communities already formed, consider them for an agent to join a community later    
#     #temp_combos_list_temp = list(combos_consider.index)
#     #temp_combos_list_filter = [combos_consider.loc[i]['Comm_formed'] if combos_consider.loc[i]['community'] == 1 else combos_consider.loc[i]['Ind_formed'] if combos_consider.loc[i]['individual'] == 1  else i for i in temp_combos_list_temp]
#     #temp_combos_list_filter = temp_combos_list_temp
#     #%make the possible combinations    
#     
#     #-------------
#     #do a similar thing as above for buildings with individual PV installed
#     combos_consider.at['B2371142','Ind_formed'] = "PV_B2371142"
#     #temp_combos_list_filter = [combos_consider.loc[i]['Ind_formed'] if combos_consider.loc[i]['individual'] == 1 else i for i in temp_combos_list_temp]
#     #temp_combos_list_filter = temp_combos_list_temp
#     
# =============================================================================
    
    temp_combos_list_temp = list(combos_consider.index)
    temp_combos_list_filter = [combos_consider.loc[i]['Comm_formed'] if combos_consider.loc[i]['community'] == 1 else combos_consider.loc[i]['Ind_formed'] if combos_consider.loc[i]['individual'] == 1  else i for i in temp_combos_list_temp]
    #temp_combos_list_filter has the names of the agents who are available to form a community. 
    #Includes existing communitities (= C_B12345_B6789) and individual PVs (= PV_B456789)
    
    #make combinations in the following lines of code--------------------------
    import itertools
    temp_combos_list = []
    constant =  uid                              #read this from the self.uid
    if constant in temp_combos_list_filter:
        temp_combos_list_filter.remove(constant) #so that the energy_champion building does not get eliminated in the combos - the combos are made without it and then this building is added to all the combos
    if '' in temp_combos_list_filter:
        temp_combos_list_filter.remove('')
    
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
    
    
    #make the solar and demand info for all the combinations to calculate the NPVs
    df_solar_combo      = pd.DataFrame(data = None)
    df_demand_combo     = pd.DataFrame(data = None)
    df_pvsize_combo     = pd.DataFrame(data = None, index = ['Size'])
    df_bldgs_names      = pd.DataFrame(data = None, index = ['Bldg_Names'])
    df_zones_names      = pd.DataFrame(data = None, index = ['Zone_Names'])
    df_join_individual  = pd.DataFrame(data = None, index = ['Join_Ind'])
    df_join_community   = pd.DataFrame(data = None, index = ['Join_Comm'])
    df_num_smart_meters = pd.DataFrame(data = None, index = ['Num'])
    df_num_members      = pd.DataFrame(data = None, index = ['Num_Members'])
    #print("temp_combos_list =========",temp_combos_list)
    set_flag_pros_cons = 0              #to make 2 cases - consumer (= 1) and prosumer (= 0)
    set_flag_pros_cons_individual = 0
    
    for i in range(len(temp_combos_list)):
        temp_solar = 0
        temp_demand = 0
        temp_name = ""
        temp_pv_list = []
        temp_pv_size = 0
        temp_bldg_og_name_list_df = []
        temp_bldg_og_name_list = []
        temp_bldg_zones_list_df = []
        temp_bldg_zones_list = []
        temp_join_individual_list = []
        temp_join_individual = 0
        temp_join_community_list = []
        temp_join_community = 0
        temp_num_smartmeters_list = []
        temp_num_members_list = []
        temp_num_smart_meters = 0
        temp_num_members = 0
        temp_names_comms_list = []
        
        #for the consumer case of the 2 agent situation with existing community installed PV
        temp_solar_2 = 0
        temp_demand_2 = 0
        temp_join_individual_2 = 0
        temp_join_community_2 = 0
        temp_name_2 = ""
        temp_pv_list_2 = []
        temp_pv_size_2 = 0
        temp_bldg_og_name_list_df_2 = []
        temp_bldg_og_name_list_2 = [] 
        temp_bldg_zones_list_df_2 = []
        temp_bldg_zones_list_2 = []
        temp_join_individual_list_2 = []
        temp_join_individual_2 = 0
        temp_join_community_list_2 = []
        temp_join_community_2 = 0
        temp_num_smartmeters_list_2 = []
        temp_num_members_list_2 = []
        temp_num_smart_meters_2 = 0
        temp_num_members_2 = 0
        temp_names_comms_list_2 = []
        
        #for the consumer case of the 2 agent situation with individually installed PV of another agent
        temp_solar_3 = 0
        temp_demand_3 = 0
        temp_join_individual_3 = 0
        temp_join_community_3 = 0
        temp_name_3 = ""
        temp_pv_list_3 = []
        temp_pv_size_3 = 0
        temp_bldg_og_name_list_df_3 = []
        temp_bldg_og_name_list_3 = [] 
        temp_bldg_zones_list_df_3 = []
        temp_bldg_zones_list_3 = []
        temp_join_individual_list_3 = []
        temp_join_individual_3 = 0
        temp_join_community_list_3 = []
        temp_join_community_3 = 0
        temp_num_smartmeters_list_3 = []
        temp_num_members_list_3 = []
        temp_num_smart_meters_3 = 0
        temp_num_members_3 = 0
        temp_names_comms_list_3 = []
        
        for j in range(len(temp_combos_list[i])):
            
            try:
                temp_bldg               = temp_combos_list[i][j] 
                temp_solar              = temp_solar + df_solar[temp_bldg]
                temp_demand             = temp_demand + df_demand[temp_bldg]
                temp_pv_size            = temp_pv_size + combos_consider.loc[temp_bldg]['pv_size_kw']
                temp_bldg_og_name_list.append(temp_bldg)
                temp_bldg_zones_list.append(combos_consider.loc[temp_bldg]['zone_id'])
                temp_num_smart_meters   = temp_num_smart_meters + combos_consider.loc[temp_bldg]['num_smart_meters']
                temp_num_members        = temp_num_members + 1                                                          #always add 1 in this case as 1 agent will be considered here
                temp_name               = temp_name + temp_combos_list[i][j]+ '_'
                temp_name_comms         = temp_name 
            
            except KeyError: 
                #keyError will only occur in case an existing community *OR* bldg with installed PV is one of the choices for the agent, hence read info from the combos dataframes
                
                #*exisiting community is an option*
                if temp_bldg[0] == 'C': 
                    if len(temp_combos_list[i]) != 2:
                        #case in which new community is being formed with multiple new agents and an exisiting community.
                        #everyone installs solar on their roofs - all PROSUMERS
                        temp_bldg               = temp_combos_list[i][j] 
                        temp_bldg_comm_contain  = temp_combos_list[i][j] 
                        temp_bldg_comm          = temp_bldg_comm_contain.split(sep = '_')
                        temp_bldg_comm.remove('C')
                        temp_bldg_og_name_list.extend(temp_bldg_comm)
                        temp_bldg_zones_list.extend(Combos_formed_Info.loc[temp_bldg]['combos_zone_ids'])
                        temp_solar              = temp_solar + df_solar_combos_main[temp_bldg]
                        temp_demand             = temp_demand + df_demand_combos_main[temp_bldg]
                        temp_pv_size            = temp_pv_size + Combos_formed_Info.loc[temp_bldg]['combos_pv_size_kw']
                        temp_join_community     = 1
                        temp_num_smart_meters   = temp_num_smart_meters + Combos_formed_Info.loc[temp_bldg]['combos_num_smart_meters']
                        temp_num_members        = temp_num_members + Combos_formed_Info.loc[temp_bldg]['Num_Members']#len(temp_combos_list[i])#temp_num_members + combos_consider.loc[temp_bldg]['']
                        temp_name               = temp_name + temp_combos_list[i][j]+ '_'
                        temp_name_comms         = temp_name
                    
                    
                    elif len(temp_combos_list[i]) == 2:
                        #case in which only the activated agent and an exisiting community is present
                        #***agent will also install solar on his own roof - PROSUMER***
                        temp_bldg               = temp_combos_list[i][j] 
                        temp_bldg_comm_contain  = temp_combos_list[i][j] 
                        temp_bldg_comm          = temp_bldg_comm_contain.split(sep = '_')
                        temp_bldg_comm.remove('C')
                        temp_bldg_og_name_list.extend(temp_bldg_comm)
                        temp_bldg_zones_list.extend(Combos_formed_Info.loc[temp_bldg]['combos_zone_ids'])
                        temp_solar              = temp_solar + df_solar_combos_main[temp_bldg]
                        temp_demand             = temp_demand + df_demand_combos_main[temp_bldg]
                        temp_pv_size            = temp_pv_size + Combos_formed_Info.loc[temp_bldg]['combos_pv_size_kw']
                        temp_join_community     = 1
                        temp_num_smart_meters   = temp_num_smart_meters + Combos_formed_Info.loc[temp_bldg]['combos_num_smart_meters']
                        temp_num_members        = temp_num_members + Combos_formed_Info.loc[temp_bldg]['Num_Members']#len(temp_combos_list[i])#temp_num_members + combos_consider.loc[temp_bldg]['']
                        temp_name               = temp_name + temp_combos_list[i][j]+ '_' #removed Pros from the start of the name
                        #make something to save this community name so that later it is known who is forming with community so the coop costs can be accounted for properly
                        temp_name_comms         = temp_name
                        
# =============================================================================
#                         CONSUMER CASE - NEGLECTED FOR BASELINE ABM SCENARIOS
                        
#                         #***agent will not install solar on own roof, only join as a CONSUMER***
#                         temp_bldg               = temp_combos_list[i][j] 
#                         temp_bldg_comm_contain_2= temp_combos_list[i][j] 
#                         temp_bldg_comm_2        = temp_bldg_comm_contain_2.split(sep = '_')
#                         temp_bldg_comm_2.remove('C')
#                         temp_bldg_og_name_list_2.append(temp_combos_list[i][j-1]) #to add the name of the first agent (active agent)
#                         temp_bldg_og_name_list_2.extend(temp_bldg_comm_2)
#                         temp_bldg_zones_list_2 = temp_bldg_zones_list
#                         temp_solar_2            = df_solar_combos_main[temp_bldg]
#                         temp_demand_2           = temp_demand           #only temp_demand must be considered as the KeyError only happens with the second agent; I have hardcoded it this way. Anyway calculated in the previous lines of code!
#                         temp_pv_size_2          = temp_pv_size          #Calculated in the previous lines of code!
#                         temp_join_community_2   = 1
#                         temp_num_smart_meters_2 = temp_num_smart_meters #Calculated in the previous lines of code! 
#                         temp_num_members_2      = temp_num_members      #Calculated in the previous lines of code!
#                         temp_name_2             = 'Cons_' + temp_combos_list[i][j-1] + '_' + temp_combos_list[i][j]+ '_'
#                         temp_name_comms_2       = temp_name_2
#                         set_flag_pros_cons      = 1                     #indicates that one agent combines with one existing community
#                 
# =============================================================================
                
                #*existing individually installed PV is an option*
                if temp_bldg[0] == 'P': 
                    if len(temp_combos_list[i]) != 2:
                        #case in which new community is being formed with multiple new agents and an exisiting community.
                        #***everyone installs solar on their roofs - all PROSUMERS***
                        temp_bldg               = temp_combos_list[i][j] 
                        temp_bldg_comm_contain  = temp_combos_list[i][j] 
                        temp_bldg_comm          = temp_bldg_comm_contain.split(sep = '_')
                        temp_bldg_comm.remove('PV')
                        temp_bldg_og_name_list.extend(temp_bldg_comm)
                        temp_bldg_zones_list.extend(combos_consider.loc[temp_bldg_comm]['zone_id'])
                        print("temp_bldg in k = 1 = ", temp_bldg)
                        temp_bldg_name_edited   = str.strip(temp_bldg, 'PV_')
                        temp_solar              = temp_solar + df_solar[temp_bldg_name_edited]
                        temp_demand             = temp_demand + df_demand[temp_bldg_name_edited]
                        temp_pv_size            = temp_pv_size + combos_consider.loc[temp_bldg_name_edited]['pv_size_kw']
                        temp_join_individual    = 1
                        temp_num_smart_meters   = temp_num_smart_meters + combos_consider.loc[temp_bldg_name_edited]['num_smart_meters']
                        temp_num_members        = temp_num_members + 1 #just add 1 coz only 1 agent will be considered here. 
                        temp_name               = temp_name + temp_combos_list[i][j]+ '_'
                        temp_name_comms         = temp_name
                        
                    
                    elif len(temp_combos_list[i]) == 2:
                        #case in which only the activated agent and an exisiting individually adopted agent is present
                        #***agent will also install solar on his own roof - PROSUMER***
                        temp_bldg               = temp_combos_list[i][j] 
                        temp_bldg_comm_contain  = temp_combos_list[i][j] 
                        temp_bldg_comm          = temp_bldg_comm_contain.split(sep = '_')
                        temp_bldg_comm.remove('PV')
                        temp_bldg_og_name_list.extend(temp_bldg_comm)
                        temp_bldg_zones_list.extend(combos_consider.loc[temp_bldg_comm]['zone_id'])
                        temp_bldg_name_edited   = str.strip(temp_bldg, 'PV_')
                        temp_solar              = temp_solar + df_solar[temp_bldg_name_edited]
                        temp_demand             = temp_demand + df_demand[temp_bldg_name_edited]
                        temp_pv_size            = temp_pv_size + combos_consider.loc[temp_bldg_name_edited]['pv_size_kw'] #CHECK!!
                        temp_join_individual    = 1
                        temp_num_smart_meters   = temp_num_smart_meters + combos_consider.loc[temp_bldg_name_edited]['num_smart_meters']
                        temp_num_members        = len(temp_combos_list[i])      
                        temp_name               = temp_name + temp_combos_list[i][j]+ '_' #removed Pros from the start of the name
                        temp_name_comms         = temp_name
                        
# =============================================================================
#                         CONSUMER CASE - NEGLECTED FOR BASELINE ABM SCENARIOS
                        
#                         #***agent will not install solar on own roof, only join as a CONSUMER***
#                         temp_bldg               = temp_combos_list[i][j] 
#                         temp_bldg_comm_contain_3= temp_combos_list[i][j] 
#                         temp_bldg_comm_3        = temp_bldg_comm_contain_3.split(sep = '_')
#                         temp_bldg_comm_3.remove('PV')
#                         temp_bldg_og_name_list_3.append(temp_combos_list[i][j-1])                               #to add the name of the first agent (active agent)
#                         temp_bldg_og_name_list_3.extend(temp_bldg_comm_3)
#                         temp_bldg_zones_list_3.append(combos_consider.loc[temp_combos_list[i][j-1]]['zone_id']) #to add the zone of the first agent (active agent)
#                         temp_bldg_zones_list_3.extend(combos_consider.loc[temp_bldg_comm_3]['zone_id'])
#                         temp_bldg_name_edited   = str.strip(temp_bldg, 'PV_')
#                         temp_solar_3            = df_solar[temp_bldg_name_edited]                               #only the existing agent with the PV
#                         temp_demand_3           = temp_demand                                                   #Calculated in the previous lines of code!
#                         temp_pv_size_3          = combos_consider.loc[temp_bldg_name_edited]['pv_size_kw']      #Calculated in the previous lines of code! - CHECK!!
#                         temp_join_individual_3  = 1
#                         temp_num_smart_meters_3 = temp_num_smart_meters                                         #Calculated in the previous lines of code!
#                         temp_num_members_3      = len(temp_combos_list[i])                                      #temp_num_members + combos_consider.loc[temp_bldg][''] if this is wromg, hardcode to 2
#                         temp_name_3             = 'Cons_' + temp_combos_list[i][j-1] + '_' + temp_combos_list[i][j]+ '_'
#                         temp_name_comms_3       = temp_name_3
#                         set_flag_pros_cons_individual = 1                                                       #indicates that one agent + one exisitng individual combine to form a community
#         
# =============================================================================
                        
        #populating the dataframes in the usual case - no previously installed PV systems           
        #also populating the dataframes in the existing PV case - with many other agents so all all PROSUMERS
        df_solar_combo[temp_name]       = ""
        df_solar_combo[temp_name]       = temp_solar
        df_demand_combo[temp_name]      = ""
        df_demand_combo[temp_name]      = temp_demand
        temp_pv_list.append(temp_pv_size)
        temp_bldg_og_name_list_df.append(temp_bldg_og_name_list)
        temp_bldg_zones_list_df.append(temp_bldg_zones_list)
        temp_join_individual_list.append(temp_join_individual)
        temp_join_community_list.append(temp_join_community)
        temp_num_smartmeters_list.append(temp_num_smart_meters)
        temp_num_members_list.append(temp_num_members)
        df_pvsize_combo[temp_name]      = ""
        df_pvsize_combo[temp_name]      = temp_pv_list
        df_bldgs_names[temp_name]       = ""
        df_bldgs_names[temp_name]       = temp_bldg_og_name_list_df  
        df_zones_names[temp_name]       = ""
        df_zones_names[temp_name]       = temp_bldg_zones_list_df
        df_join_individual[temp_name]   = ""
        df_join_community[temp_name]    = ""
        df_join_individual[temp_name]   = temp_join_individual_list
        df_join_community[temp_name]    = temp_join_community_list
        df_num_smart_meters[temp_name]  = ""
        df_num_smart_meters[temp_name]  = temp_num_smartmeters_list
        df_num_members[temp_name]       = ""
        df_num_members[temp_name]       = temp_num_members_list
        temp_names_comms_list.append(temp_name_comms)
       
# =============================================================================
#         SAVING DATA FOR THE CONSUMER CASES - NEGLECTED FOR BASELINE ABM SCENARIOS
#         PLEASE KEEP A COPY OF THIS SOMEWHERE BEFORE TAKING IT OUT OF THIS SCRIPT.
#         WILL BE USEFUL 

#         #populating the dataframes in the 1 agent + 1 existing COMMUNITY case - join as CONSUMER
#         if set_flag_pros_cons == 1:
#             df_solar_combo[temp_name_2]         = ""
#             df_solar_combo[temp_name_2]         = temp_solar_2
#             df_demand_combo[temp_name_2]        = ""
#             df_demand_combo[temp_name_2]        = temp_demand_2
#             temp_pv_list_2.append(temp_pv_size_2)
#             temp_bldg_og_name_list_df_2.append(temp_bldg_og_name_list_2)
#             temp_bldg_zones_list_df_2.append(temp_bldg_zones_list_2)
#             temp_join_individual_list_2.append(temp_join_individual_2)
#             temp_join_community_list_2.append(temp_join_community_2)
#             temp_num_smartmeters_list_2.append(temp_num_smart_meters_2)
#             temp_num_members_list_2.append(temp_num_members_2)
#             df_pvsize_combo[temp_name_2]        = ""
#             df_pvsize_combo[temp_name_2]        = temp_pv_list_2
#             df_bldgs_names[temp_name_2]         = ""
#             df_bldgs_names[temp_name_2]         = temp_bldg_og_name_list_df_2
#             df_zones_names[temp_name_2]         = ""
#             df_zones_names[temp_name_2]         = temp_bldg_zones_list_df_2
#             df_join_individual[temp_name_2]     = ""
#             df_join_community[temp_name_2]      = ""
#             df_join_individual[temp_name_2]     = temp_join_individual_list_2
#             df_join_community[temp_name_2]      = temp_join_community_list_2
#             df_num_smart_meters[temp_name_2]    = ""
#             df_num_smart_meters[temp_name_2]    = temp_num_smartmeters_list_2
#             df_num_members[temp_name_2]         = ""
#             df_num_members[temp_name_2]         = temp_num_members_list_2
#             temp_names_comms_list_2.append(temp_name_comms_2)
#             set_flag_pros_cons = 0 #reset flag so that erroneous columns in the dataframe are not created
#         
#         #populating the dataframes in the 1 agent + 1 existing INDIVIDUAL case - join as CONSUMER
#         if set_flag_pros_cons_individual == 1:
#             print('temp_name_3 case entered', temp_name_3)
#             df_solar_combo[temp_name_3]         = ""
#             df_solar_combo[temp_name_3]         = temp_solar_3
#             df_demand_combo[temp_name_3]        = ""
#             df_demand_combo[temp_name_3]        = temp_demand_3
#             temp_pv_list_3.append(temp_pv_size_3)
#             temp_bldg_og_name_list_df_3.append(temp_bldg_og_name_list_3)
#             temp_bldg_zones_list_df_3.append(temp_bldg_zones_list_3)
#             temp_join_individual_list_3.append(temp_join_individual_3)
#             temp_join_community_list_3.append(temp_join_community_3)
#             temp_num_smartmeters_list_3.append(temp_num_smart_meters_3)
#             temp_num_members_list_3.append(temp_num_members_3)
#             df_pvsize_combo[temp_name_3]        = ""
#             df_pvsize_combo[temp_name_3]        = temp_pv_list_3
#             df_bldgs_names[temp_name_3]         = ""
#             df_bldgs_names[temp_name_3]         = temp_bldg_og_name_list_df_3
#             df_zones_names[temp_name_3]         = ""
#             df_zones_names[temp_name_3]         = temp_bldg_zones_list_df_3
#             df_join_individual[temp_name_3]     = ""
#             df_join_community[temp_name_3]      = ""
#             df_join_individual[temp_name_3]     = temp_join_individual_list_3
#             df_join_community[temp_name_3]      = temp_join_community_list_3
#             df_num_smart_meters[temp_name_3]    = ""
#             df_num_smart_meters[temp_name_3]    = temp_num_smartmeters_list_3
#             df_num_members[temp_name_3]         = ""
#             df_num_members[temp_name_3]         = temp_num_members_list_3
#             temp_names_comms_list_3.append(temp_name_comms_3)
#             set_flag_pros_cons_individual = 0 #reset flag so that erroneous columns in the dataframe are not created
#         
# =============================================================================
    NPV_combos = pd.DataFrame(data = None)
    
    year = year     #step_ctr from the ABM gives the current year of simulation in the model. 0 = 2018, 1 = 2019 and so on...
    
    #this calculates NPVs for all possible combinations
    from npv_combos_function import npv_calc_combos
    NPV_combos = npv_calc_combos(df_solar_combo, df_demand_combo, year, data.loc[i]["bldg_owner"],
                                 df_pvsize_combo, df_num_smart_meters,df_num_members) #admin_costs, rate_cooperation, temp_names_comms_list) #year should be the current year in the model! 2018, 2019... so that the correct PV price is taken
    
    #this ranks the NPVs and then returns the best NPV. If no combination is possible then an empty dataframe is returned.
    from combos_ranking import ranking_combos
    print("ranking called here")
    Combos_Info = ranking_combos(NPV_combos, df_demand, combos_consider,df_join_individual, df_join_community, df_bldgs_names, df_zones_names)
    
    #so that index of the Combos_Info dataframe is renamed 
    if len(Combos_Info.index) > 0:
        temp_name = Combos_Info.index[0] #eg = 'C_B147891_B147892_'
        x = temp_name.split('_')
        y = 'C'
        for i in x:
            y = y + '_' +  i
        Combos_Info = Combos_Info.rename(index={temp_name: y})
        Combos_Info['Num_Members'] = (len(x) -1)                
    
    return Combos_Info, NPV_combos, df_solar_combo, df_demand_combo, temp_name, combos_consider

            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
               
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
        
 