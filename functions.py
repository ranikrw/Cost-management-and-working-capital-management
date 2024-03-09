import numpy as np
import pandas as pd


from statsmodels.formula.api import ols

import os

from scipy.stats import pearsonr

from sklearn import preprocessing

def load_data(year_start,year_end):
    data = pd.read_csv('../data_processed/data_processed.csv',sep=';',low_memory=False)

    data = data[(data['fyear']>=year_start)&(data['fyear']<=year_end)].reset_index(drop=True)

    # Checking if all is unique
    unique_company_id = data.groupby(['company_id','fyear']).size().reset_index()
    temp = unique_company_id[0].unique()
    if len(temp)!=1:
        print('WARNING: not all company_id-fyear unique')
    
    return data

# https://stackoverflow.com/questions/25571882/pandas-columns-correlation-with-statistical-significance
def calculate_pvalues(df):
    df = df.dropna()._get_numeric_data()
    dfcols = pd.DataFrame(columns=df.columns)
    pvalues = dfcols.transpose().join(dfcols, how='outer')
    for r in df.columns:
        for c in df.columns:
            pvalues[r][c] = round(pearsonr(df[r], df[c])[1], 4)
    return pvalues


def make_descriptives(temp_df):
    df_descriptives = pd.DataFrame()
    df_descriptives['Mean']     = np.mean(temp_df,axis=0)
    df_descriptives['Std']      = np.std(temp_df,axis=0)
    df_descriptives['Min']      = np.min(temp_df,axis=0)

    df_descriptives['1 per']    = np.nanquantile(temp_df,0.01,axis=0)
    df_descriptives['5 per']    = np.nanquantile(temp_df,0.05,axis=0)
    df_descriptives['25 per']   = np.nanquantile(temp_df,0.25,axis=0)
    df_descriptives['50 per']   = np.nanquantile(temp_df,0.50,axis=0)
    df_descriptives['75 per']   = np.nanquantile(temp_df,0.75,axis=0)
    df_descriptives['95 per']   = np.nanquantile(temp_df,0.95,axis=0)
    df_descriptives['99 per']   = np.nanquantile(temp_df,0.99,axis=0)
    df_descriptives['Max']      = np.max(temp_df,axis=0)
    return df_descriptives


def add_tailing_zeros_decimals(num,num_decimals):
    while len(num[num.rfind('.')+1:])!=num_decimals:
        num = num+'0'
    return num

def shift_row_to_bottom(col_to_shift,df):
  idx = df.index.tolist()
  idx.remove(col_to_shift)
  df = df.reindex(idx + [col_to_shift])
  return df

def model_preparing(X,y,data):
    df = pd.concat([X,data['company_id']],axis=1)
    df = pd.concat([df,y],axis=1)
    for i in X.columns:
        if i == X.columns[0]:
            string_formula = str(y.name) +' ~ '+i
        else:
            string_formula = string_formula+' + '+str(i)
    return df,string_formula

def regression_rrw(var,data,results_df,model_number,do_standardize,num_decimals,fixed_effects_year,fixed_effects_industry,WCM_proxy):

    y = data[var[0]]
    X = data[var[1:]]

    if do_standardize:
        if model_number == '2':
            coefs_to_standardize = [
                'lnSalesAINT',
                'lnSalesEINT',
                'lnSalesGDP',
                'lnSales'+WCM_proxy,
                'DlnSalesAINT',
                'DlnSalesEINT',
                'DlnSalesGDP',
                'DlnSales'+WCM_proxy]
            num_in_list = 0
            for i in coefs_to_standardize:
                num_in_list = num_in_list+int(i in var)
            if num_in_list!=len(coefs_to_standardize):
                print('ERROR: not all coefficients to standardize is present')
            X.loc[:,coefs_to_standardize] = pd.DataFrame(preprocessing.scale(X[coefs_to_standardize]), columns = coefs_to_standardize)

    # Add fixed effects
    if fixed_effects_year:
        temp = pd.get_dummies(data['fyear'].astype(int)).iloc[:,:-1]
        for i in temp.columns:
            temp=temp.rename(columns = {i:'dy'+str(i)})
        X = pd.concat([X,temp],axis=1)
    if fixed_effects_industry:
        temp = pd.get_dummies(data['industry']).iloc[:,:-1]
        for i in temp.columns:
            temp=temp.rename(columns = {i:'di'+str(i)})
        X = pd.concat([X,temp],axis=1)

    df,string_formula = model_preparing(X,y,data)

    model = ols(formula=string_formula,data=df).fit(cov_type='cluster', cov_kwds={'groups': df['company_id']})

    list_variables = ['Intercept'] + var[1:]
    
    series_coef     = pd.Series(dtype=object)
    series_t_val    = pd.Series(dtype=object)
    for i in list_variables:
        coef = np.round(model.params[i],num_decimals).astype(str)
        coef = add_tailing_zeros_decimals(coef,num_decimals)
        series_coef[i] = coef

        tval = np.round(model.tvalues[i],num_decimals).astype(str)
        tval = add_tailing_zeros_decimals(tval,num_decimals)
        pval = model.pvalues[i]
        if pval<0.001:
            tval = tval+'****'
        elif pval<0.01:
            tval = tval+'***'
        elif pval<0.05:
            tval = tval+'**'
        elif pval<0.10:
            tval = tval+'*'
        series_t_val[i] = tval

    # Number of observations
    series_coef['No. of obs.'] = thousand_seperator(X.shape[0])

    # R squared
    series_coef['R2'] = np.round(model.rsquared,num_decimals).astype(str)

    # Fixed effects
    if fixed_effects_year:
        series_coef['Year fixed effects']  = 'Yes'
    else:
        series_coef['Year fixed effects']  = 'No'
    if fixed_effects_industry:
        series_coef['Industry fixed effects']  = 'Yes'
    else:
        series_coef['Industry fixed effects']  = 'No'

    # Merging coefficient and pvalue results
    series_results = pd.concat([series_coef,series_t_val],axis=1)
    series_results.columns = ['Model ('+model_number+')','T-test']

    # Renaming coefficients
    series_results = series_results.rename(index={'Intercept': 'beta0'})
    series_results = series_results.rename(index={'lnSales': 'beta1'})
    series_results = series_results.rename(index={'DlnSales': 'beta2'})

    series_results = series_results.rename(index={'lnSalesAINT': 'gamma_1^1'})
    series_results = series_results.rename(index={'lnSalesEINT': 'gamma_1^2'})
    series_results = series_results.rename(index={'lnSalesGDP': 'gamma_1^3'})
    series_results = series_results.rename(index={'lnSales'+WCM_proxy: 'gamma_1^4'})

    series_results = series_results.rename(index={'DlnSalesAINT': 'gamma_2^1'})
    series_results = series_results.rename(index={'DlnSalesEINT': 'gamma_2^2'})
    series_results = series_results.rename(index={'DlnSalesGDP': 'gamma_2^3'})
    series_results = series_results.rename(index={'DlnSales'+WCM_proxy: 'gamma_2^4'})

    series_results = series_results.rename(index={'lnSalesPrev': 'beta3'})
    series_results = series_results.rename(index={'DlnSalesPrev': 'beta4'})

    series_results = series_results.rename(index={'beta_1I': 'beta_1^I'})
    series_results = series_results.rename(index={'beta_2I': 'beta_2^I'})
    series_results = series_results.rename(index={'beta_1D': 'beta_1^D'})
    series_results = series_results.rename(index={'beta_2D': 'beta_2^D'})

    results_df = pd.concat([results_df,series_results],axis=1)

    return results_df

def make_ratio(numerator,denumerator):
    ratio = [None]*len(numerator)
    for i in range(len(numerator)):
        if denumerator[i]==0:
            if numerator[i]>0:
                ratio[i] = np.inf
            elif numerator[i]<0:
                ratio[i] = -np.inf
            else: # numerator[i]==0:
                ratio[i] = 0
        else:
            ratio[i] = numerator[i] / denumerator[i]
    
    ratio = pd.Series(ratio)

    # Setting inf and -inf to maximum and minimum, respectively, values
    ratio = ratio.replace(np.inf,np.max(ratio[ratio != np.inf]))
    ratio = ratio.replace(-np.inf,np.max(ratio[ratio != -np.inf]))

    return ratio

def winsorize_rrw(data,variables_to_winsorize,interval_winsorizing_ratios):
    
    for var in variables_to_winsorize:
        lower = data[var].quantile(interval_winsorizing_ratios[0])
        upper = data[var].quantile(interval_winsorizing_ratios[1])
        data[var] = data[var].clip(lower=lower, upper=upper)

    return data


def exclude_missing_prev_year(num_prev,data,sample_selection_series):
    col_name_1 = 'Excluding if no firm-year observation the previous year'
    col_name_2 = 'Excluding if no firm-year observation the two previous years'
    if (num_prev==1)|(num_prev==2):
        ind = pd.isnull(data['sales_prev'])==False
        data = data[ind]
        data = data.reset_index(drop=True) # Reset index
        sample_selection_series[col_name_1] = thousand_seperator(np.sum(ind==False))
    if num_prev==2:
        ind = pd.isnull(data['sales_prev_prev'])==False
        data = data[ind]
        data = data.reset_index(drop=True) # Reset index
        sample_selection_series[col_name_2] = thousand_seperator(np.sum(ind==False))
    else:
        sample_selection_series[col_name_2] = None
    if (num_prev!=1)&(num_prev!=2):
        print('ERROR defining num_prev in function exclude_missing_prev_year()')
    return data,sample_selection_series


def thousand_seperator(number):
    return "{:,.0f}".format(number)

def removing_zero_and_negative_ratios(var_log,var_CCC,data,sample_selection_series,WCM_proxy):
    obs_before = data.shape[0]

    for var in var_log:
        ind = data[var]<=0
        ind = ind | (data[var].isnull())
        data = data[ind==False]
        data = data.reset_index(drop=True) # Reset index 

    col_name = 'Excluding zero for accounting items used in log-ratios'
    sample_selection_series[col_name] = thousand_seperator(obs_before-data.shape[0])

    col_name = 'Excluding zero for accounting items used as denominator in '+WCM_proxy
    if len(var_CCC)>0:
        obs_before = data.shape[0]

        for var in var_CCC['num']:
            ind = data[var].isnull()
            data = data[ind==False]
            data = data.reset_index(drop=True) # Reset index 
        for var in var_CCC['denum']:
            ind = data[var]<=0
            ind = ind | (data[var].isnull())
            data = data[ind==False]
            data = data.reset_index(drop=True) # Reset index 
        
        sample_selection_series[col_name] = thousand_seperator(obs_before-data.shape[0])
    else:
        sample_selection_series[col_name] = None

    return data,sample_selection_series


def sample_selection(data,num_prev,var_log,var_CCC,model_number,sample_selection_table,WCM_proxy):
    sample_selection_series = pd.Series(dtype=float)
    
    text_initial_sample = 'All firm-years of non-financial firms'
    sample_selection_series[text_initial_sample] = thousand_seperator(data.shape[0])

    # we include only firm-year observations of firms where 
    # also a firm-year from the previous accounting year is available
    data,sample_selection_series = exclude_missing_prev_year(num_prev,data,sample_selection_series)

    # Removing zero and negative
    data,sample_selection_series = removing_zero_and_negative_ratios(var_log,var_CCC,data,sample_selection_series,WCM_proxy)

    # Adding final sample size and merging with table
    sample_selection_series['Final sample'] = thousand_seperator(data.shape[0])
    sample_selection_table['Model '+model_number] = sample_selection_series

    return data,sample_selection_table