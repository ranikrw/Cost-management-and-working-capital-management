import numpy as np
# Copyright © 2005-2019, NumPy Developers.

import pandas as pd
# Copyright (c) 2008-2012, AQR Capital Management, LLC, Lambda Foundry, Inc. and PyData Development Team

import statsmodels.api as sm

import os

import sys
from functions import *

num_decimals = 3 # For results tables

do_standardize = True # Standardized coefficients

fixed_effects_year = True
fixed_effects_industry = True

##############################################
##  Define response variable
##############################################
# The start and end year of the sample
year_start = 1983
year_end = 2022

# Make folder for saving results
folder_name = 'results/'
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

test_settings = [
    ('operating_costs', 'CCC'),
    ('operating_costs', 'NTC'),
    ('operating_costs', 'WCTA'),
    ('cost_of_goods', 'CCC'),
    ('cost_of_goods', 'NTC'),
    ('cost_of_goods', 'WCTA'),
]

for costs_for_response, WCM_proxy in test_settings:

    string_for_save = '_'+costs_for_response+'_'+WCM_proxy

    ##############################################
    ##  Creating data frames for inserting results
    ##############################################
    results_df = pd.DataFrame()
    results_df.index.name = costs_for_response

    sample_selection_table = pd.DataFrame()

    ##############################################
    ##  Model 1
    ##############################################
    model_number   = '1'

    # Load data
    data = load_data(year_start,year_end)

    # Sample selection
    num_prev = 1
    var_log = [
        costs_for_response,
        costs_for_response+'_prev',
        'sales',
        'sales_prev',
    ]
    var_CCC=[]
    data,sample_selection_table = sample_selection(data,num_prev,var_log,var_CCC,model_number,sample_selection_table,WCM_proxy)

    # Making variables
    data['lnCost_'+costs_for_response]  = np.log((data[costs_for_response]/data[costs_for_response+'_prev']).astype(float))
    data['lnSales']     = np.log((data['sales']/data['sales_prev']).astype(float))
    data['DlnSales']    = (data['lnSales']<0)*data['lnSales']

    # Defining variables for regression
    var = [
        'lnCost_'+costs_for_response,
        'lnSales',
        'DlnSales',
    ]

    # Regression
    results_df = regression_rrw(var,data,results_df,model_number,do_standardize,num_decimals,fixed_effects_year,fixed_effects_industry,WCM_proxy)

    ##############################################
    ##  Model 2
    ##############################################
    model_number   = '2'

    # Load data
    data = load_data(year_start,year_end)

    # Sample selection
    num_prev = 1
    var_log = [
        costs_for_response,
        costs_for_response+'_prev',
        'sales',
        'sales_prev',
        'assets',  
        'employees', 
    ]

    var_CCC = {}
    if WCM_proxy == 'CCC':
        data['purchases']   = data['cost_of_goods'] - data['inventories_prev'] + data['inventories']
        var_CCC['num'] = [
            'inventories',
            'receivables',
            'payable',
        ]
        var_CCC['denum'] = [
            'sales',
            'cost_of_goods',
            'purchases',
        ]
    elif WCM_proxy == 'NTC':
        var_CCC['num'] = [
            'inventories',
            'receivables',
            'payable',
        ]
        var_CCC['denum'] = [
            'sales',
        ]
    elif WCM_proxy == 'WCTA':
        var_CCC['num'] = [
            'current_assets',
            'current_liabilities',
        ]
        var_CCC['denum'] = [
            'assets',
        ]
    data,sample_selection_table = sample_selection(data,num_prev,var_log,var_CCC,model_number,sample_selection_table,WCM_proxy)

    # Making variables
    data['lnCost_'+costs_for_response]  = np.log((data[costs_for_response]/data[costs_for_response+'_prev']).astype(float))
    data['lnSales']     = np.log((data['sales']/data['sales_prev']).astype(float))
    data['DlnSales']    = (data['lnSales']<0)*data['lnSales']

    data['AINT']    = np.log((data['assets']/data['sales']).astype(float))
    data['EINT']    = np.log((data['employees']/data['assets']).astype(float))
    data['GDP']     = (data['gdp']/data['gdp_prev']).astype(float)-1

    interval_winsorizing_ratios = [0.01,0.99] # In numbers, so 0.01 = restricting at 1%
    if WCM_proxy == 'CCC':
        data['INV'] = 365 * make_ratio(data['inventories'],data['cost_of_goods'])
        data['ACR'] = 365 * make_ratio(data['receivables'],data['sales'])
        data['ACP'] = 365 * make_ratio(data['payable'].astype(float),data['purchases'])
        data = winsorize_rrw(data,['INV','ACR','ACP'],interval_winsorizing_ratios)
        data[WCM_proxy]         = data['INV'] + data['ACR'] - data['ACP'] 

    elif WCM_proxy == 'NTC':
        data[WCM_proxy] = 365*(data['inventories'].astype(float) + data['receivables'].astype(float) - data['payable'].astype(float))/data['sales']
        data = winsorize_rrw(data,[WCM_proxy],interval_winsorizing_ratios)

    elif WCM_proxy == 'WCTA':
        data[WCM_proxy] = (data['current_assets'].astype(float) - data['current_liabilities'].astype(float))/(data['assets'].astype(float))
        data = winsorize_rrw(data,[WCM_proxy],interval_winsorizing_ratios)

    data['lnSalesAINT'] = data['lnSales']*data['AINT']
    data['lnSalesEINT'] = data['lnSales']*data['EINT']
    data['lnSalesGDP']  = data['lnSales']*data['GDP']
    data['lnSales'+WCM_proxy]  = data['lnSales']*data[WCM_proxy]

    data['DlnSalesAINT'] = data['DlnSales']*data['AINT']
    data['DlnSalesEINT'] = data['DlnSales']*data['EINT']
    data['DlnSalesGDP']  = data['DlnSales']*data['GDP']
    data['DlnSales'+WCM_proxy]  = data['DlnSales']*data[WCM_proxy]


    # Defining variables for regression
    var = [
        'lnCost_'+costs_for_response,
        'lnSales',
        'DlnSales',
        'lnSalesAINT',
        'lnSalesEINT',
        'lnSalesGDP',
        'lnSales'+WCM_proxy,
        'DlnSalesAINT',
        'DlnSalesEINT',
        'DlnSalesGDP',
        'DlnSales'+WCM_proxy,
    ]

    # Descriptives
    if WCM_proxy == 'CCC':
        make_descriptives(data[var+[WCM_proxy,'INV','ACR','ACP']].astype(float)).to_excel(folder_name+'Descriptives_'+model_number+string_for_save+'.xlsx')
    elif (WCM_proxy == 'NTC') | (WCM_proxy == 'WCTA'):
        make_descriptives(data[var+[WCM_proxy]].astype(float)).to_excel(folder_name+'Descriptives_'+model_number+string_for_save+'.xlsx')

    # Regression
    results_df = regression_rrw(var,data,results_df,model_number,do_standardize,num_decimals,fixed_effects_year,fixed_effects_industry,WCM_proxy)

    ##############################################
    ##  Model 3
    ##############################################
    model_number   = '3'

    # Load data
    data = load_data(year_start,year_end)

    # Sample selection
    num_prev = 2
    var_log = [
        costs_for_response,
        costs_for_response+'_prev',
        'sales',
        'sales_prev',
        'sales_prev_prev',
    ]
    var_CCC=[]
    data,sample_selection_table = sample_selection(data,num_prev,var_log,var_CCC,model_number,sample_selection_table,WCM_proxy)

    # Making variables
    data['lnCost_'+costs_for_response]  = np.log((data[costs_for_response]/data[costs_for_response+'_prev']).astype(float))
    data['lnSales']     = np.log((data['sales']/data['sales_prev']).astype(float))
    data['DlnSales']    = (data['lnSales']<0)*data['lnSales']
    data['lnSalesPrev']     = np.log((data['sales_prev']/data['sales_prev_prev']).astype(float))
    data['DlnSalesPrev']    = (data['lnSalesPrev']<0)*data['lnSales']

    # Defining variables for regression
    var = [
        'lnCost_'+costs_for_response,
        'lnSales',
        'DlnSales',
        'lnSalesPrev',
        'DlnSalesPrev',
    ]

    # Regression
    results_df = regression_rrw(var,data,results_df,model_number,do_standardize,num_decimals,fixed_effects_year,fixed_effects_industry,WCM_proxy)


    ##############################################
    ##  Model 4
    ##############################################
    model_number   = '4'

    # Load data
    data = load_data(year_start,year_end)

    # Sample selection
    num_prev = 2
    var_log = [
        costs_for_response,
        costs_for_response+'_prev',
        'sales',
        'sales_prev',
        'sales_prev_prev',
    ]
    var_CCC=[]
    data,sample_selection_table = sample_selection(data,num_prev,var_log,var_CCC,model_number,sample_selection_table,WCM_proxy)

    # Making variables
    data['lnCost_'+costs_for_response]  = np.log((data[costs_for_response]/data[costs_for_response+'_prev']).astype(float))

    data['lnSales']     = np.log((data['sales']/data['sales_prev']).astype(float))
    data['DlnSales']    = (data['lnSales']<0)*data['lnSales']

    data['lnSalesPrev'] = np.log((data['sales_prev']/data['sales_prev_prev']).astype(float))
    data['I_prev']      = (data['lnSalesPrev']>0)
    data['D_prev']      = (data['lnSalesPrev']<0)

    data['beta_1I']    = data['I_prev']*data['lnSales']
    data['beta_2I']    = data['I_prev']*data['DlnSales']

    data['beta_1D']    = data['D_prev']*data['lnSales']
    data['beta_2D']    = data['D_prev']*data['DlnSales']

    # Defining variables for regression
    var = [
        'lnCost_'+costs_for_response,
        'beta_1I',
        'beta_2I',
        'beta_1D',
        'beta_2D',
    ]

    # Regression
    results_df = regression_rrw(var,data,results_df,model_number,do_standardize,num_decimals,fixed_effects_year,fixed_effects_industry,WCM_proxy)

    ##############################################
    ##  Rearranging results table
    ##############################################
    results_df = shift_row_to_bottom('beta0',results_df)
    results_df = shift_row_to_bottom('Year fixed effects',results_df)
    results_df = shift_row_to_bottom('Industry fixed effects',results_df)
    results_df = shift_row_to_bottom('R2',results_df)
    results_df = shift_row_to_bottom('No. of obs.',results_df)

    results_df.index.name = string_for_save

    ##############################################
    ##  Save results
    ##############################################
    results_df.to_excel(folder_name+'Results'+string_for_save+'.xlsx')
    sample_selection_table.to_excel(folder_name+'Sample_selection'+string_for_save+'.xlsx')

