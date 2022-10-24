# -*- coding: utf-8 -*-
'''
2022-09-19 created by Ando & Xiao

'''
import imf_datatools.ecos_sdmx_utilities as ecos
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
idx = pd.IndexSlice

from ax_package import ax_forecast


#%% download data

# =============================================================================
# ##### countries #####
# =============================================================================
cs_list = ['France','Estonia','Spain','Germany','Latvia','Italy','Denmark',
           'Portugal','The Netherlands']
countries = ecos.get_ebv_country_info()
c_list = [countries.loc[countries['ShortForm'] == cs,'Code'].item() for cs in cs_list]

# =============================================================================
# ##### WEO #####
# =============================================================================
weo_db = ecos.get_weo_databases()
weo_db = weo_db[ (weo_db['year'] >= 2017) & (weo_db['month']!=1)]
db_list = weo_db.index.tolist()
frequency = 'A'
v = 'PCPI' # PCPIHA_IX
weo = pd.DataFrame([])
for db in db_list:
    print(db)    
    df = ecos.get_ecos_sdmx_data(db,
                                 c_list,
                                 v,
                                 freq=frequency)
    df['vintage'] = pd.Timestamp(year  = weo_db.loc[db,'year'],
                                 month = weo_db.loc[db,'month'],
                                 day = 1)
    df['year'] = df.index.year
    weo = pd.concat([weo,df],axis=0,ignore_index=True)
weo = weo.set_index(['vintage','year'])
weo.columns = pd.MultiIndex.from_arrays([
    [countries.loc[countries['Code'] == v[:3],'ShortForm'].item()  for v in weo.filter(like=".A").columns],
    [v[3:] for v in weo.columns]],
    names=['country','indicator'])
weo = weo.stack('country').swaplevel(i=1,j=2,axis=0).sort_index()
weo['inflation'] = weo.groupby(['vintage','country']).pct_change(1)*100

# =============================================================================
# ##### CPI #####
# =============================================================================
db_list = ecos.get_databases()
cpi_db = db_list.loc[db_list.index.str.contains('CPI'),:]
cpi_db = cpi_db.iloc[[0,2],:]
ds = ecos.get_data_structure('ECDATA_CPI_LATEST_PUBLISHED')
dsi = ds['Consumer Price Index (CPI) Indicator']
v_list = [k for k in dsi.keys() if re.search('Harmonized',dsi[k]) and
                                  (not re.search('Percent',dsi[k])) and
                                  (not re.search('Overlap',dsi[k])) and
                                  (not re.search('Standard',dsi[k]))]
v_list.sort()

# table to explain indicator
v_table = pd.DataFrame([v_list,[dsi[v] for v in v_list]],index=['indicator','explanation']).T
v_table['short_indicator'] = [v[re.search('PCPI',v).end():] for v in v_table.indicator]
v_list_short = [*set([v[re.search('PCPIHA',v).end():v.find('_')] for v in v_list])]
v_list_short.remove('')
v_list_short.sort()

df = ecos.get_ecos_sdmx_data('ECDATA_CPI_LATEST_PUBLISHED',
                             c_list,
                             v_list, 
                             freq = ['A', 'M'])
df.columns = pd.MultiIndex.from_arrays([[countries.loc[countries['Code'] == v[:3],'ShortForm'].item()  for v in df.columns],
                                        [v[3:] for v in df.columns]],
                                       names=['country','indicator'])
# annual data
df_A = df.filter(like='.A').dropna().copy()
df_A.index = df_A.index.year
df_A.index = df_A.index.rename('year')
df_A = df_A.stack('country').swaplevel(i=0,j=1,axis=0).sort_index()
df_A.columns = ['A_' + v[6:v.find('.A')] for v in df_A.columns]

# monthly data
df_M = df.filter(like='.M').copy()
df_M.index = pd.MultiIndex.from_arrays([df_M.index.year,df_M.index.month],names=['year','month'])
df_M = df_M.stack('country').dropna()
df_M.columns = [v[6:v.find('.M')] for v in df_M.columns]
df_M = df_M.unstack('month').swaplevel(i=0,j=1,axis=0).sort_index()
colname = []
for coli,col in enumerate(df_M.columns):
    colname.append('M' + str(df_M.columns.get_level_values(1)[coli]) + '_' + df_M.columns.get_level_values(0)[coli])
df_M.columns = colname

# combine anniual and transposed monthly data
cpi = pd.concat([df_A, df_M], axis = 1)
cpi.to_csv('cpi.csv')

# check monthly aggregation
# chap 8 https://ec.europa.eu/eurostat/documents/3859598/9479325/KS-GQ-17-015-EN-N.pdf/d5e63427-c588-479f-9b19-f4b4d698f2a2
# https://ec.europa.eu/eurostat/web/hicp/faq
df_M_check = df_M.copy()
df_M_check.columns = pd.MultiIndex.from_arrays([[v[1:v.find('_')] for v in df_M.columns],
                                                [v[v.find('_')+1:] for v in df_M.columns]],
                                         names = ['month','indicator'])
df_M_check = df_M_check.stack('month')
df_M_check['sum_weight'] = df_M_check.filter(like='_WT').sum(axis=1)

for col in v_list_short:
    df_M_check[col+'_step1'] = np.nan
    for row in df_M_check.index:
        if row[1] >= 1997:
            df_M_check.loc[row,col+'_step1'] = df_M_check.loc[row,col+'_IX']/df_M_check.loc[idx[row[0],row[1]-1,'12'],col+'_IX']

for v in v_list_short:
    df_M_check.loc[:,v+'_con'] = df_M_check.loc[:,v+'_step1'] * df_M_check.loc[:,v+'_WT']/df_M_check.sum_weight
df_M_check['step2'] = df_M_check.filter(like='_con').sum(axis=1)

for row in df_M_check.index:
    if row[1] >= 1997:
        df_M_check.loc[row,'step3'] = df_M_check.loc[row,'step2'] * df_M_check.loc[idx[row[0],row[1]-1,'12'],'_IX']
should_be_0 = df_M_check['_IX'] - df_M_check['step3']
should_be_0.unstack('country').plot(title = 'All-item index - aggregation')

# check annual aggregation
cpi_check = cpi.filter(like='__IX').copy()
cpi_check['monthly_average'] = cpi_check.filter(like='M').mean(axis=1)
cpi_check.unstack('country')
(cpi_check[idx['A__IX']] - cpi_check[idx['monthly_average']]).unstack('country').plot()

# =============================================================================
# ##### GAS (to be added) #####
# =============================================================================
db_list = ecos.get_databases()
db_list = db_list[db_list['dbpath'].str.contains('GAS')]
db_list = db_list[db_list['dbpath'].str.contains('(?=.*Databases/RES/GAS/Archives/)(?=.*Pub)')]
db_list['vintage'] = pd.to_datetime( db_list.index.str[10:14]+db_list.index.str[7:10] , format = '%Y%b')
db_list = db_list.sort_values(['vintage'])

# =============================================================================
# ##### market forecast (to be added) #####
# =============================================================================
db = ecos.get_databases()
db = db[db.index.str.contains('BLOOMBERG')]
ds = ecos.get_data_structure('ECDATA_BLOOMBERG')
ticker = pd.DataFrame(ds['Bloomberg (BLBG) Ticker'],index=['ticker']).T
inflation_linked = ticker[ticker['ticker'].str.contains('Inflation')]

#%% define functions

# Generate unknowns and data up to year t
def setunknown(df_in, k, t):
    
    # creat variable names to nullify
    ind_u = ['M'+str(i) for i in range(k,13)]
    ind_u.extend(['A'])
    ind_u = [v for v in df_in.columns if (v[:v.find("_")] in ind_u) and (v[-2:] == 'IX')]
    
    df = df_in.loc[:t,:].copy()
    df.loc[df.index[-1],ind_u] = np.nan
    df_u = df.loc[:,df.isna().any()]
    df_k = df.loc[:,~df.isna().any()]    
    df_out = pd.concat([df_u, df_k], axis=1)
    
    return df_out

# Different methods
def plain(df0):
    
    forecast_year = df0.index[-1]
    gr = df0.filter(like='_IX').pct_change(1)*100
    gr[np.isnan(df0)] = np.nan
    gr = gr.dropna(how='all')
    
    w = df0.filter(like='_WT')
    w.columns = pd.MultiIndex.from_arrays([[v[:v.find('_')] for v in w.columns],
                                           [v[v.find('_')+1:]  for v in w.columns]
                                           ],names=['month','indicator'])
    w = w/w.groupby('month',axis=1).sum()
    for col in w.columns:
        v = col[1]
        v = v[:v.find('_')]
        w.loc[forecast_year,col] = df0.loc[forecast_year-1,'M12__IX']/ \
                                   df0.loc[forecast_year-1,'M12_'+v+'_IX']
    
    C = pd.DataFrame(0,index=range(13),
                     columns = gr.columns)
    C.loc[0,'A__IX'] = 1
    v_list_short = [v[v.find('_')+1:] for v in df0.columns]
    v_list_short = [v[:v.find('_')] for v in v_list_short]
    v_list_short = list(set(v_list_short))
    v_list_short.remove('')
    
    for m in range(1,13):
        # constraint for year
        C.loc[0,'M' + str(m) + '__IX'] = - df0.loc[forecast_year-1,'M' + str(m) + '__IX']/ \
                                           df0.loc[forecast_year-1,'A__IX']/12
        
        # constraint for each month
        C.loc[m,'M' + str(m) + '__IX'] = 1
        for v in v_list_short:
            C.loc[m,'M'+str(m)+'_'+v+'_IX'] = - w.loc[forecast_year,idx['M'+str(m),v+'_WT']]* \
                df0.loc[forecast_year-1,'M'+str(m)+'_'+v+'_IX']/ \
                df0.loc[forecast_year-1,'M'+str(m)+'__IX']
    C = C.values
    
    # constant for each month
    d = w.diff(1)
    for col in d.columns:
        m = col[0]
        v = col[1]
        v = v[:v.find('_')]
        d.loc[forecast_year,col] = d.loc[forecast_year,col]* \
                                   df0.loc[forecast_year-1,m+'_'+v+'_IX']/ \
                                   df0.loc[forecast_year-1,m+'__IX']
    d = d.groupby('month',axis=1).sum().loc[forecast_year,:]
    d.index= [int(str(v)[1:]) for v in d.index]
    d = np.append(0,d.sort_index().values)

    C_dict = {}
    d_dict = {}
    for i, t in enumerate(df0.index):
        C_dict[i] = C
        d_dict[i] = d
        
    df2, df1, df0aug_coef = ax_forecast(gr, lag, Tin, C_dict, d_dict)
    
    return [df2, df1, df0aug_coef]

#%% forecast
lag = 2
Tin = 5

cpi = pd.read_csv('cpi.csv',index_col=[0,1])

c = "The Netherlands"
cpi_c = cpi.loc[idx[c,:],:]
cpi_c = cpi_c.dropna()
cpi_c.index = cpi_c.index.get_level_values('year')

gr = cpi_c.filter(like='_IX').pct_change(1)*100
gr[np.isnan(cpi_c)] = np.nan
gr = gr.dropna(how='all')

methods = ['plain','collapsed','combo','weo']
stages  = ['1st','2nd']
fe = pd.DataFrame(index   = pd.MultiIndex.from_arrays([weo_db['year'],weo_db['month']]),
                  columns = pd.MultiIndex.from_product([methods,stages])
                 ).drop(('weo','1st'),axis=1).drop(2022,axis=0) # forecast error, weo only has 2nd stage

for (weo_year,weo_month) in fe.index:
    print(weo_year,weo_month)
    
    # create forecast input
    df0 = setunknown(cpi_c, weo_month-2, weo_year)
    forecast_year = df0.index[-1]
    
    # run methods
    plain2,     plain1,     plain_coef     = plain(df0)
    weo2 = weo.loc[(weo.index.get_level_values('vintage').year  == weo_year) & \
                   (weo.index.get_level_values('vintage').month == weo_month) & \
                   (weo.index.get_level_values('country') == c),
                   :]
    weo2.index = weo2.index.get_level_values('year')
    
    # record forecast error and coefficients
    fe.loc[idx[weo_year,weo_month], ('plain','1st')] = plain1.loc[forecast_year,'A__IX'] - gr.loc[forecast_year,'A__IX']
    fe.loc[idx[weo_year,weo_month], ('plain','2nd')] = plain2.loc[forecast_year,'A__IX'] - gr.loc[forecast_year,'A__IX']
    fe.loc[idx[weo_year,weo_month], ('weo','2nd')] = weo2.loc[forecast_year,'inflation'] - gr.loc[forecast_year,'A__IX']
    
# plot results
fe[['plain','weo']].abs().plot(title=c + 'absolute error')
fe[['plain','weo']].abs().mean().plot.bar()