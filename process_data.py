import numpy as np
# Copyright © 2005-2019, NumPy Developers.

import pandas as pd
# Copyright (c) 2008-2012, AQR Capital Management, LLC, Lambda Foundry, Inc. and PyData Development Team

from tqdm import tqdm # Showing progress bar in for-loops

##################################################################
##  Load data
##################################################################
data = pd.read_csv('../data/data_compustat.csv',sep=',',low_memory=False)

# Restricting on fyear 1960 and later because World Bank data start from then
data = data[(data['fyear']>=1960)&(data['fyear']<=2022)].reset_index(drop=True)

# Removing 11 financial statements because of duplicates in the data
data = data.drop_duplicates(subset=['fyear','gvkey'])

# Sorting data
data = data.sort_values(by=['fyear','gvkey']).reset_index(drop=True)

##################################################################
##  Define variables
##################################################################

# See table 2 of:
# Lyngstadaas, H. (2020). Packages or systems? Working capital management 
# and financial performance among listed U.S. manufacturing firms. Journal of 
# Management Control, 31(4), 403–450. https://doi.org/10.1007/s00187-020-00306-z
data = data.rename(columns={
    'gvkey':'company_id',
    'fyear':'fyear',
    'sale': 'sales', # SALE -- Sales/Turnover (Net) (SALE)
    'xopr': 'operating_costs', # XOPR -- Operating Expenses Total (XOPR)
    'cogs': 'cost_of_goods', # COGS -- Cost of Goods Sold (COGS)
    'at': 'assets', # AT -- Assets - Total (AT)
    'emp': 'employees', # EMP -- Employees (EMP)
    'invt': 'inventories', # INVT -- Inventories - Total (INVT)
    'rectr': 'receivables', # RECTR -- Receivables - Trade (RECTR)
    'ap': 'payable', # AP -- Accounts Payable - Trade (AP)
    'sic': 'industry', # SIC -- Standard Industry Classification Code (SIC)
    'act': 'current_assets', # ACT -- Current Assets - Total (ACT)
    'lct': 'current_liabilities', # LCT -- Current Liabilities - Total (LCT)
    })


##################################################################
##  GDP and CPI data
##################################################################
columns = []
for c in list(range(1960,2022+1)):
    columns = columns + [str(c)]

# GDP
gdp_data    = pd.read_csv('../data/GDP/API_NY.GDP.MKTP.CD_DS2_en_csv_v2_5728855.csv',sep=',',header=2)
gdp_data = gdp_data[gdp_data['Country Name']=='United States'][columns].reset_index(drop=True)
gdp_data.columns = gdp_data.columns.astype(int)
gdp_data = gdp_data.T[0]

# CPI
CPI_data = pd.read_csv('../data/CPI/API_FP.CPI.TOTL_DS2_en_csv_v2_5728960.csv',sep=',',header=2)
CPI_data = CPI_data[CPI_data['Country Name']=='United States'][columns].reset_index(drop=True)
CPI_data.columns = CPI_data.columns.astype(int)
CPI_data = CPI_data.T[0]


##################################################################
## Deflating Nominal Values to Real Values
##################################################################
# Financial data
columns_to_deflate = [
    'sales',
    'operating_costs',
    'cost_of_goods',
    'assets',
    'inventories',
    'receivables',
    'payable',
    'current_assets',
    'current_liabilities',
]
data = data.reset_index(drop=True)
CPIs = CPI_data.loc[data['fyear']].values
for c in columns_to_deflate:
    data[c] = data[c] / CPIs

# GDP data
CPIs = CPI_data.loc[gdp_data.index].values
gdp_data = gdp_data / CPIs

##################################################################
##  Filtering  data
##################################################################
# Our sample start at 1983
ind = data['fyear']>=1983

##################################################################
##  Making data                                                 ##
##################################################################
data['gdp']                     = None
data['gdp_prev']                = None

data_for_saving_variables = pd.DataFrame(index=data.index)
data_for_saving_variables['sales_prev']              = None
data_for_saving_variables['sales_prev_prev']         = None
data_for_saving_variables['operating_costs_prev']    = None
data_for_saving_variables['cost_of_goods_prev']      = None
data_for_saving_variables['inventories_prev']        = None

for fyear in tqdm(np.sort(data.loc[ind, 'fyear'].unique())):
    
    # Filter data for the current 'fyear' and previous 'fyear'
    curr_data = data[ind & (data['fyear'] == fyear)]
    prev_data = data[data['fyear'] == fyear - 1]
    prev_prev_data = data[data['fyear'] == fyear - 2]
    
    # Merge data on 'company_id' to get accounting 
    # numbers from one and two years ago
    columns_prev = [
        'sales',
        'operating_costs',
        'cost_of_goods',
        'inventories',
    ]
    columns_prev_prev = [
        'sales',
    ]
    merged_data = curr_data.merge(prev_data[['company_id']+columns_prev], on='company_id', how='left', suffixes=('', '_prev'))
    merged_data = merged_data.merge(prev_prev_data[['company_id']+columns_prev_prev], on='company_id',how='left', suffixes=('', '_prev_prev'))
    merged_data.index = curr_data.index

    # Assign to data
    for c in columns_prev:
        data_for_saving_variables.loc[merged_data.index,c+'_prev'] = merged_data[c+'_prev']
    for c in columns_prev_prev:
        data_for_saving_variables.loc[merged_data.index,c+'_prev_prev'] = merged_data[c+'_prev_prev']

    # Assign values to 'gdp' and 'gdp_prev'
    data.loc[curr_data.index,'gdp']       = gdp_data.loc[fyear]
    data.loc[curr_data.index,'gdp_prev']  = gdp_data.loc[fyear - 1]

data = pd.concat([data,data_for_saving_variables],axis=1)

columns_to_keep = [
    'company_id',
    'fyear',
    'industry',
    'sales',
    'sales_prev',
    'sales_prev_prev',
    'operating_costs',
    'operating_costs_prev',
    'cost_of_goods',
    'cost_of_goods_prev',
    'assets',
    'employees',
    'gdp',
    'gdp_prev',
    'payable',
    'inventories',
    'inventories_prev',
    'receivables',
    'current_assets',
    'current_liabilities',
]
data = data.loc[ind,columns_to_keep].reset_index(drop=True)

folder_name = '../data_processed'
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# Save data 
data.to_csv(folder_name+'/data_processed.csv',index=False,sep=';')

