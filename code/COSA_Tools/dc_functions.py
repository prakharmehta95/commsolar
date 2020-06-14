# -*- coding: utf-8 -*-
"""
Created on Sat May  2 13:22:23 2020

@author: prakh
"""

class functions:
    """
    FUNCTIONS for agent attributes, counting installed capacity by building typologies
    
    @Alejandro: most of these functions here are for data collection.
    Check the datacollector definition and you will see. Can be simply modified
    so that we can collect data based on ownership/building-use type separately
    """
        
#-----GYM-----    
    def agent_type_gym(self):
        '''
        to find total number of GYM individual adoptions
        '''
        sum_agent_type_gym_ind = 0
        for i in self.schedule.agents:
            if i.bldg_type == 'GYM':
                sum_agent_type_gym_ind = sum_agent_type_gym_ind + i.adopt_ind
        return sum_agent_type_gym_ind
    
    
    def agent_type_gym_CAP(self):
        '''
        to find total CAPACITY of GYM individual adoptions
        '''
        sum_agent_type_gym_ind_cap = 0
        for i in self.schedule.agents:
            if i.bldg_type == 'GYM' and i.adopt_ind == 1:
                sum_agent_type_gym_ind_cap = sum_agent_type_gym_ind_cap + i.pv_size
        return sum_agent_type_gym_ind_cap

#-----HOSPITAL-----    
    def agent_type_hospital(self):
        '''
        to find total number of HOSPITAL individual adoptions
        '''
        sum_agent_type_hospital_ind = 0
        for i in self.schedule.agents:
            if i.bldg_type == 'HOSPITAL':
                sum_agent_type_hospital_ind = sum_agent_type_hospital_ind + i.adopt_ind
        return sum_agent_type_hospital_ind
    
    def agent_type_hospital_CAP(self):
        '''
        to find total CAPACITY of HOSPITAL individual adoptions
        '''
        sum_agent_type_hospital_ind_cap = 0
        for i in self.schedule.agents:
            if i.bldg_type == 'HOSPITAL' and i.adopt_ind == 1:
                sum_agent_type_hospital_ind_cap = sum_agent_type_hospital_ind_cap + i.pv_size
        return sum_agent_type_hospital_ind_cap

#-----HOTEL-----    
    def agent_type_hotel(self):
        '''
        to find total number of HOTEL individual adoptions
        '''
        sum_agent_type_hotel_ind = 0
        for i in self.schedule.agents:
            if i.bldg_type == 'HOTEL':
                sum_agent_type_hotel_ind = sum_agent_type_hotel_ind + i.adopt_ind
        return sum_agent_type_hotel_ind
    
    def agent_type_hotel_CAP(self):
        '''
        to find total CAPACITY of HOTEL individual adoptions
        '''
        sum_agent_type_hotel_ind_cap = 0
        for i in self.schedule.agents:
            if i.bldg_type == 'HOTEL' and i.adopt_ind == 1:
                sum_agent_type_hotel_ind_cap = sum_agent_type_hotel_ind_cap + i.pv_size
        return sum_agent_type_hotel_ind_cap

#-----INDUSTRIAL-----
    def agent_type_industrial(self):
        '''
        to find total number of INDUSTRIAL individual adoptions
        '''
        sum_agent_type_industrial_ind = 0
        for i in self.schedule.agents:
            if i.bldg_type == 'INDUSTRIAL':
                sum_agent_type_industrial_ind = sum_agent_type_industrial_ind + i.adopt_ind
        return sum_agent_type_industrial_ind
    
    def agent_type_industrial_CAP(self):
        '''
        to find total CAPACITY of INDUSTRIAL individual adoptions
        '''
        sum_agent_type_industrial_ind_cap = 0
        for i in self.schedule.agents:
            if i.bldg_type == 'INDUSTRIAL' and i.adopt_ind == 1:
                sum_agent_type_industrial_ind_cap = sum_agent_type_industrial_ind_cap + i.pv_size
        return sum_agent_type_industrial_ind_cap

#-----LIBRARY-----    
    def agent_type_library(self):
        '''
        to find total number of LIBRARY individual adoptions
        '''
        sum_agent_type_library_ind = 0
        for i in self.schedule.agents:
            if i.bldg_type == 'LIBRARY':
                sum_agent_type_library_ind = sum_agent_type_library_ind + i.adopt_ind
        return sum_agent_type_library_ind
    
    def agent_type_library_CAP(self):
        '''
        to find total CAPACITY of LIBRARY individual adoptions
        '''
        sum_agent_type_library_ind_cap = 0
        for i in self.schedule.agents:
            if i.bldg_type == 'LIBRARY' and i.adopt_ind == 1:
                sum_agent_type_library_ind_cap = sum_agent_type_library_ind_cap + i.pv_size
        return sum_agent_type_library_ind_cap

#-----MULTI_RES-----    
    def agent_type_multi_res(self):
        '''
        to find total number of MULTI_RES individual adoptions
        '''
        sum_agent_type_multi_res_ind = 0
        for i in self.schedule.agents:
            if i.bldg_type == 'MULTI_RES':
                sum_agent_type_multi_res_ind = sum_agent_type_multi_res_ind + i.adopt_ind
        return sum_agent_type_multi_res_ind
    
    def agent_type_multi_res_CAP(self):
        '''
        to find total CAPACITY of MULTI_RES individual adoptions
        '''
        sum_agent_type_multi_res_ind_cap = 0
        for i in self.schedule.agents:
            if i.bldg_type == 'MULTI_RES' and i.adopt_ind == 1:
                sum_agent_type_multi_res_ind_cap = sum_agent_type_multi_res_ind_cap + i.pv_size
        return sum_agent_type_multi_res_ind_cap

#-----OFFICE-----    
    def agent_type_office(self):
        '''
        to find total number of OFFICE individual adoptions
        '''
        sum_agent_type_office_ind = 0
        for i in self.schedule.agents:
            if i.bldg_type == 'OFFICE':
                sum_agent_type_office_ind = sum_agent_type_office_ind + i.adopt_ind
        return sum_agent_type_office_ind
    
    def agent_type_office_CAP(self):
        '''
        to find total CAPACITY of OFFICE individual adoptions
        '''
        sum_agent_type_office_ind_cap = 0
        for i in self.schedule.agents:
            if i.bldg_type == 'OFFICE' and i.adopt_ind == 1:
                sum_agent_type_office_ind_cap = sum_agent_type_office_ind_cap + i.pv_size
        return sum_agent_type_office_ind_cap

#-----PARKING-----    
    def agent_type_parking(self):
        '''
        to find total number of PARKING individual adoptions
        '''
        sum_agent_type_parking_ind = 0
        for i in self.schedule.agents:
            if i.bldg_type == 'PARKING':
                sum_agent_type_parking_ind = sum_agent_type_parking_ind + i.adopt_ind
        return sum_agent_type_parking_ind
    
    def agent_type_parking_CAP(self):
        '''
        to find total CAPACITY of PARKING individual adoptions
        '''
        sum_agent_type_parking_ind_cap = 0
        for i in self.schedule.agents:
            if i.bldg_type == 'PARKING' and i.adopt_ind == 1:
                sum_agent_type_parking_ind_cap = sum_agent_type_parking_ind_cap + i.pv_size
        return sum_agent_type_parking_ind_cap

#-----RESTAURANT-----    
    def agent_type_restaurant(self):
        '''
        to find total number of RESTAURANT individual adoptions
        '''
        sum_agent_type_restaurant_ind = 0
        for i in self.schedule.agents:
            if i.bldg_type == 'RESTAURANT':
                sum_agent_type_restaurant_ind = sum_agent_type_restaurant_ind + i.adopt_ind
        return sum_agent_type_restaurant_ind
    
    def agent_type_restaurant_CAP(self):
        '''
        to find total CAPACITY of RESTAURANT individual adoptions
        '''
        sum_agent_type_restaurant_ind_cap = 0
        for i in self.schedule.agents:
            if i.bldg_type == 'RESTAURANT' and i.adopt_ind == 1:
                sum_agent_type_restaurant_ind_cap = sum_agent_type_restaurant_ind_cap + i.pv_size
        return sum_agent_type_restaurant_ind_cap

#-----RETAIL-----   
    def agent_type_retail(self):
        '''
        to find total number of RETAIL individual adoptions
        '''
        sum_agent_type_retail_ind = 0
        for i in self.schedule.agents:
            if i.bldg_type == 'RETAIL':
                sum_agent_type_retail_ind = sum_agent_type_retail_ind + i.adopt_ind
        return sum_agent_type_retail_ind
    
    def agent_type_retail_CAP(self):
        '''
        to find total CAPACITY of RETAIL individual adoptions
        '''
        sum_agent_type_retail_ind_cap = 0
        for i in self.schedule.agents:
            if i.bldg_type == 'RETAIL' and i.adopt_ind == 1:
                sum_agent_type_retail_ind_cap = sum_agent_type_retail_ind_cap + i.pv_size
        return sum_agent_type_retail_ind_cap

#-----SCHOOL-----   
    def agent_type_school(self):
        '''
        to find total number of SCHOOL individual adoptions
        '''
        sum_agent_type_school_ind = 0
        for i in self.schedule.agents:
            if i.bldg_type == 'SCHOOL':
                sum_agent_type_school_ind = sum_agent_type_school_ind + i.adopt_ind
        return sum_agent_type_school_ind
    
    def agent_type_school_CAP(self):
        '''
        to find total CAPACITY of SCHOOL individual adoptions
        '''
        sum_agent_type_school_ind_cap = 0
        for i in self.schedule.agents:
            if i.bldg_type == 'SCHOOL' and i.adopt_ind == 1:
                sum_agent_type_school_ind_cap = sum_agent_type_school_ind_cap + i.pv_size
        return sum_agent_type_school_ind_cap

#-----SINGLE_RES-----
    def agent_type_single_res(self):
        '''
        to find total number of SINGLE_RES individual adoptions
        '''
        sum_agent_type_single_res_ind = 0
        for i in self.schedule.agents:
            if i.bldg_type == 'SINGLE_RES':
                sum_agent_type_single_res_ind = sum_agent_type_single_res_ind + i.adopt_ind
        return sum_agent_type_single_res_ind
    
    def agent_type_single_res_CAP(self):
        '''
        to find total CAPACITY of SINGLE_RES individual adoptions
        '''
        sum_agent_type_single_res_ind_cap = 0
        for i in self.schedule.agents:
            if i.bldg_type == 'SINGLE_RES' and i.adopt_ind == 1:
                sum_agent_type_single_res_ind_cap = sum_agent_type_single_res_ind_cap + i.pv_size
        return sum_agent_type_single_res_ind_cap

#--------------------------
        
    def cumulate_solar_ind(self):
        """
        To find the cumulative INDIVIDUAL installations at the end of every time step
        """
        solar_sum_ind = 0
        for i in self.schedule.agents:
            solar_sum_ind = solar_sum_ind + i.adopt_ind
        return solar_sum_ind
            
    def cumulate_solar_comm(self):
        """
        To find the cumulative COMMUNITY "ALL buildings with installations" at the end of every time step
        """
        solar_sum_comm = 0
        for i in self.schedule.agents:
            solar_sum_comm = solar_sum_comm + i.adopt_comm
        return solar_sum_comm  
    
    def cumulate_solar_champions(self):
        """
        To find the cumulative COMMUNITY installations at the end of every time step
         = total number of communities formed
        """
        solar_sum_champ = 0
        for i in self.schedule.agents:
            solar_sum_champ = solar_sum_champ + i.en_champ
        return solar_sum_champ        
    
    def cumulate_solar_ind_sizes(self):
        """
        To find the cumulative INDIVIDUAL solar capacity at the end of every time step
        """
        solar_sum_sizes = 0
        
        for i in self.schedule.agents:
            if i.adopt_ind == 1:
                solar_sum_sizes = solar_sum_sizes + i.pv_size
        return solar_sum_sizes
     
    def cumulate_solar_comm_sizes(self):
        """
        To find the cumulative COMMUNITY solar capacity at the end of every time step
        """
        solar_comm_sizes = 0
        for i in self.schedule.agents:
            if i.adopt_comm == 1:
                solar_comm_sizes = solar_comm_sizes + i.pv_size
        return solar_comm_sizes