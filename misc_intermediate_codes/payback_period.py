# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 13:23:10 2019

@author: iA
"""
#%%
import numpy as np

a = np.pv(0.05, 3, -100.0, fv = 0, when = 'begin')
print(a)


#%%
import pandas as pd

rate = 10/100 # 10%
cash_flows = [-50000] + [10000] * 70 # Amounts in millions
cf_df = pd.DataFrame(cash_flows, columns=['UndiscountedCashFlows'])
cf_df.index.name = 'Year'
cf_df['DiscountedCashFlows'] = np.pv(rate=rate, pmt=0, nper=cf_df.index, fv=-cf_df['UndiscountedCashFlows'])
cf_df['CumulativeDiscountedCashFlows'] = np.cumsum(cf_df['DiscountedCashFlows'])
if any(cf_df.CumulativeDiscountedCashFlows > 0):
    final_full_year = cf_df[cf_df.CumulativeDiscountedCashFlows < 0].index.values.max()
    fractional_yr = -cf_df.CumulativeDiscountedCashFlows[final_full_year ]/cf_df.DiscountedCashFlows[final_full_year + 1]
else:
    final_full_year = 'na'#999
    fractional_yr = 'na'#1000

payback_period = final_full_year + fractional_yr
print(payback_period)
#%%
import pandas as pd

cash_flows = [-5000] + [500] * 7

def discounted_payback_period(rate, cash_flows=list()):
    print(cash_flows)
    cf_df = pd.DataFrame(cash_flows, columns=['UndiscountedCashFlows'])
    cf_df.index.name = 'Year'
    cf_df['DiscountedCashFlows'] = np.pv(rate=rate, pmt=0, nper=cf_df.index, fv=-cf_df['UndiscountedCashFlows'])
    cf_df['CumulativeDiscountedCashFlows'] = np.cumsum(cf_df['DiscountedCashFlows'])
    
    if any(cf_df.CumulativeDiscountedCashFlows > 0):
        final_full_year = cf_df[cf_df.CumulativeDiscountedCashFlows < 0].index.values.max()
        fractional_yr = -cf_df.CumulativeDiscountedCashFlows[final_full_year ]/cf_df.DiscountedCashFlows[final_full_year + 1]
    else:
        final_full_year = 'nope,'#999
        fractional_yr = 'no payback'#1000
    
    payback_period = final_full_year + fractional_yr
    return payback_period

a= discounted_payback_period(0.0,cash_flows)
print(a)
