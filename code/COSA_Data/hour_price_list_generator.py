#%% IMPORT PACKAGES
import numpy as np
import pandas as pd
import sys, os

#%% COMPUTE HOUR PRICE LIST

# Create a list of 8760 items indicating hour of day for a whole year
hourday = list(range(24)) * 365

# Create a list of 8736 hours containing day-of-week for each hour in year
weeekdays = list(np.repeat(["Sat","Sun","Mon","Tues","Wed","Thurs","Fri"],24))*52
# Note: The reference year is 2005, which started in Saturday

# Add the missing 24 hours to complete the 8760 hours of the year
weeekdays.extend(list(["Sat"]*24))

# Create a list of tuples for all hours of the year where each item 
# (weekday, hour-of-day) for example ("Sat", 14)
wd_hd = [(weeekdays[hour], hourday[hour]) for hour in range(8760)]

# Create list of high and low price hours in the year
# All hours of the year are "low" except from Mon-Sat from 6-21
hour_price = ["high" if ((a[0] != "Sun") and (a[1] > 5) and (a[1] < 22)) else "low" for a in wd_hd]

# %% STORE LIST IN CSV

# Set out directory
files_dir = os.path.dirname(os.path.abspath(__file__))

# Put list in dataframe
out_data_df = pd.DataFrame(hour_price)

# Save it
out_data_df.to_csv(files_dir + "\\hour_price.csv", mode='w', sep=';')
