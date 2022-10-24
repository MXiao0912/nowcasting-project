# -*- coding: utf-8 -*-
'''
2022-09-19 created by Ando & Xiao

'''
import imf_datatools.ecos_sdmx_utilities as ecos
import numpy as np
import pandas as pd
idx = pd.IndexSlice

from ax_package import ax_forecast

#%% define functions

# Generate unknowns and data up to year t
def setunknown(df_in, k, t):
    
    ind_u = ['M'+str(i) for i in range(k,13)]
    ind_u.extend(['A'])
    df = df_in.loc[:t,:].copy()
    df.loc[df.index[-1],ind_u] = np.nan
    df_u = df.loc[:,df.isna().any()]
    df_k = df.loc[:,~df.isna().any()]    
    df_out = pd.concat([df_u, df_k], axis=1)
    
    return df_out

# Different methods
def plain(df0):
    
    # constraints
    C = np.repeat(-1, len(df0.columns))
    C[df0.columns == 'A'] = 1
    C = np.reshape(C, (1, -1))
    d = 0
    C_dict = {}
    d_dict = {}
    for i, t in enumerate(df0.index):
        C_dict[i] = C
        d_dict[i] = d
        
    df2, df1, df0aug_coef = ax_forecast(df0, lag, Tin, C_dict, d_dict)
    
    return [df2, df1, df0aug_coef]

def collapsed(df0):
    
    # collapse variables
    df_u = df0.loc[:,df0.isna().any()]
    df_k = df0.loc[:,~df0.isna().any()]
    df = df0[['A']].copy()
    df['known_months'] = df_k.sum(axis=1)
    df['unknown_months'] = df_u.filter(like='M').sum(axis=1)
    df.loc[df.index[-1],'unknown_months'] = np.nan # sum of nan will be 0
    
    # constraints
    C = np.repeat(-1, len(df.columns))
    C[df.columns == 'A'] = 1
    C = np.reshape(C, (1, -1))
    d = 0
    C_dict = {}
    d_dict = {}
    for i, t in enumerate(df.index):
        C_dict[i] = C
        d_dict[i] = d
        
    df2, df1, df0aug_coef = ax_forecast(df, lag, Tin, C_dict, d_dict)
    
    return [df2, df1, df0aug_coef]
    
def combo(df0):
    
    # combine collapsed variables
    df_u = df0.loc[:,df0.isna().any()]
    df_k = df0.loc[:,~df0.isna().any()]
    df = df0.copy()
    df['known_months'] = df_k.sum(axis=1)
    df['unknown_months'] = df_u.filter(like='M').sum(axis=1)
    df.loc[df.index[-1],'unknown_months'] = np.nan # sum of nan will be 0
    
    # constraint 1: A = M1 + ... + M12
    C1 = pd.Series(-1,index=df.columns)
    C1['A'] = 1
    C1[['known_months','unknown_months']] = 0
    
    # constraint 2: known_months = M1 + ... + Mk
    C2 = pd.Series(0,index=df.columns)
    C2['known_months']   = 1
    C2[df_k.columns] = -1

    # constraint 3: unknown_months = Mk+1 + ... + M12
    C3 = pd.Series(0,index=df.columns)
    C3['unknown_months']   = 1
    C3[df_u.filter(like='M').columns] = -1
    
    # constraint 4: A = known_months + unknown_months, this is implied by C1,...,C3
    C4 = pd.Series(0,index=df.columns)
    C4['A'] = 1
    C4[['known_months','unknown_months']] = -1
    
    C = pd.concat([C1,C2,C3,C4],axis=1).values.T
    d = np.repeat(0,4).reshape(-1,1)
    C_dict = {}
    d_dict = {}
    for i, t in enumerate(df.index):
        C_dict[i] = C
        d_dict[i] = d
    
    df2, df1, df0aug_coef = ax_forecast(df, lag, Tin, C_dict, d_dict)
    
    return [df2, df1, df0aug_coef]

#%% download data
c = '138' # Netherlands

# WEO
weo_db = ecos.get_weo_databases()
weo_db = weo_db[ (weo_db['year'] >= 2017) & (weo_db['month']!=1)]
db_list = weo_db.index.tolist()
frequency = 'A'
v = 'PCPI' # PCPIHA_IX
weo = pd.DataFrame([])
for db in db_list:
    print(db)    
    df = ecos.get_ecos_sdmx_data(db,
                                 c,
                                 v,
                                 freq=frequency)
    df['vintage'] = pd.Timestamp(year  = weo_db.loc[db,'year'],
                                 month = weo_db.loc[db,'month'],
                                 day = 1)
    df['year'] = df.index.year
    weo = pd.concat([weo,df],axis=0,ignore_index=True)
weo['inflation'] = weo.groupby('vintage')[c + v + '.A'].pct_change()*100
weo_inflation = weo.loc[ (weo.year == weo.vintage.dt.year-1),:]

# CPI
v = 'PCPIHA_IX'
db_list = ecos.get_databases()
cpi_db = db_list.loc[db_list.index.str.contains('CPI'),:]
cpi_db = cpi_db.iloc[[0,2],:]
ds = ecos.get_data_structure('ECDATA_CPI_LATEST_PUBLISHED')
df = ecos.get_ecos_sdmx_data('ECDATA_CPI_LATEST_PUBLISHED',
                             c,
                             v, 
                             freq = ['A', 'M'])
df_A = df[[c + v + '.A']]
df_A.index = df_A.index.year
df_A['type'] = 'A'
df_A = df_A.dropna()
df_A.rename(columns = {c + v + '.A': 'value'}, inplace = True)
df_M = df[[c + v + '.M']]
df_M.index = pd.to_datetime(df_M.index)
df_M['type'] = ['M' + str(m) for m in df_M.index.month]
df_M.index = df_M.index.year
df_M.rename(columns = {c + v + '.M': 'value'}, inplace = True)
cpi = pd.concat([df_A, df_M], axis = 0)
cpi = cpi.pivot(columns = 'type', values = 'value')

cont = cpi.drop(columns='A')
cont = (cont-cont.shift(1)).div(cpi.A.shift(1),axis=0)/12 * 100 # transform data to yoy contribution
cont['A'] = cpi.A.pct_change() * 100
cont.loc[cont.index[-1],'A'] = np.nan # pct_change with nan gives 0
cont = cont.dropna()
cont.plot()

# check data consistency, weo = cpi?
should_be_zero = cpi.filter(like='M').mean(axis=1) - cpi['A'] # cpi annual is the average of monthly
cont.loc[cont.index>=2017,'A']
weo.loc[ (weo.year == weo.vintage.dt.year-1) & (weo.vintage.dt.month == 4),:]

#%% forecast
lag = 2
Tin = 5

methods = ['plain','collapsed','combo','weo']
stages  = ['1st','2nd']
fe = pd.DataFrame(index   = pd.MultiIndex.from_arrays([weo_db['year'],weo_db['month']]),
                  columns = pd.MultiIndex.from_product([methods,stages])
                 ).drop(('weo','1st'),axis=1).drop(2022,axis=0) # forecast error, weo only has 2nd stage
coef = []

for (t,k) in fe.index:
    print(t,k)
    
    # create forecast input
    df_orig = setunknown(cont, k-1, t)
    forecast_year = df_orig.index[-1]
    df_u = df_orig.loc[:,df_orig.isna().any()]
    df_k = df_orig.loc[:,~df_orig.isna().any()]
    df0 = pd.concat([df_u, df_k], axis=1)
    
    # run methods
    plain2,     plain1,     plain_coef     = plain(df0)
    collapsed2, collapsed1, collapsed_coef = collapsed(df0)
    combo2,     combo1,     combo_coef     = combo(df0)
    weo2 = weo.loc[(weo['year'].isin(df0.index)) &\
                   (weo.vintage.dt.year  == forecast_year) &\
                   (weo.vintage.dt.month == k),
                   :].set_index('year')
    
    # record forecast error and coefficients
    fe.loc[idx[t,k], ('plain',      '1st')] = cont.loc[forecast_year,'A'] - plain1.loc[forecast_year,'A']
    fe.loc[idx[t,k], ('plain',      '2nd')] = cont.loc[forecast_year,'A'] - plain2.loc[forecast_year,'A']
    fe.loc[idx[t,k], ('collapsed',  '1st')] = cont.loc[forecast_year,'A'] - collapsed1.loc[forecast_year,'A']
    fe.loc[idx[t,k], ('collapsed',  '2nd')] = cont.loc[forecast_year,'A'] - collapsed2.loc[forecast_year,'A']
    fe.loc[idx[t,k], ('combo',      '1st')] = cont.loc[forecast_year,'A'] - combo1.loc[forecast_year,'A']
    fe.loc[idx[t,k], ('combo',      '2nd')] = cont.loc[forecast_year,'A'] - combo2.loc[forecast_year,'A']
    fe.loc[idx[t,k], ('weo',        '2nd')] = cont.loc[forecast_year,'A'] - weo2.loc[forecast_year,'inflation']
    
    coef.append({'plain':       plain_coef,
                 'collapsed':   collapsed_coef,
                 'combo':       combo_coef
                 })

fe = fe.T.swaplevel()
fe_1st = fe.loc['1st'].T
fe_1st.plot()
fe_2nd = fe.loc['2nd'].T
fe_2nd.plot(title = 'forecast error')
fe_2nd.abs().plot(title = 'forecast error')
fe_2nd.abs().mean(axis=0).plot.bar(title = 'mean absolute forecast error')
