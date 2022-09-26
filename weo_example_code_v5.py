# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
print("\014")

#%% import packages

import os
cwd = r"\\ecnswn06p\data\SAndo\conditional_forecast_of_hierarchical_time_series"
os.chdir(cwd)

results = 'weo_example_results/'

import imf_datatools.ecos_sdmx_utilities as ecos
import matplotlib.pyplot as plt
import seaborn as sns
import numpy  as np
import os
import pandas as pd
import re
import sympy  as sy
import time

from numpy.linalg import inv
from numpy.linalg import matrix_rank

from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

import plotly.express as px
import plotly.io as pio
pio.renderers.default = "browser"

idx = pd.IndexSlice

#%% download data
# =============================================================================
# Country info
# =============================================================================
# get economies from weo published
economies = ecos.get_countries('WEO_WEO_PUBLISHED').\
    reset_index().\
        rename(columns={'countryname':'WEO_name'})
        
# keep only those starting with 3 digits number
economies = economies[economies['IMFcode'].str.contains('^\d{3}$')].\
    sort_values(['IMFcode'])

# get countries from enterprice business vocabulary
countries = ecos.get_ebv_country_info().\
    reset_index().\
        rename(columns={'ShortForm':'EBV_name',
                        'Code':'EBVcode'})

# change name for merge
countries['EBV_name'] = countries['EBV_name'].\
    replace(['Russian Federation','The Netherlands'],
            ['Russia',            'Netherlands'])

economies = economies.merge(countries[['EBVcode','EBV_name','Name']],
                            how='outer',
                            left_on='IMFcode',
                            right_on='EBVcode',
                            indicator=True)
economies['Name_ambiguous'] = ( economies['WEO_name'] != economies['EBV_name'] ) &\
                              ( economies['WEO_name'] != economies['Name'] ) # 118 is G8 in WEO and isle of Man in EBV

# country list
countries = economies[(economies['_merge']=='both') &\
                      (economies['Name_ambiguous']==False)].\
    reset_index(drop=True)[['IMFcode','WEO_name']].\
        merge(countries,
              how      = 'left',
              left_on  ='IMFcode',
              right_on = 'EBVcode')

# economy list
economies['code'] = economies['IMFcode'].fillna(economies.EBVcode)
economies = economies[['code','WEO_name','EBV_name','IMFcode','EBVcode']].\
    reset_index(drop=True)

economies.to_csv(r'weo_example_data/economies.csv',index=False)
countries.to_csv(r'weo_example_data/countries.csv',index=False)

# =============================================================================
# # WEO data, current desk-input vars
# =============================================================================
db_list = ecos.get_weo_databases()
db_list['vintage'] = pd.to_datetime( db_list[['year','month']].assign(Day=1) )
db_list = db_list[(db_list['year']>=2016) & (db_list['month']==4)].sort_values(['vintage'])

# variables with explanation
var = pd.read_excel(r'weo_example_data/WEO Indicators.xlsx', sheet_name = None,header=None)
var = [ var[v] for v in list(filter(re.compile('WEO Series').match, var.keys())) ]
for i,sheet in enumerate(var):
    start_loc = np.where(sheet.values=='Mnemonics')
    var[i] = sheet.iloc[start_loc[0][0]:,start_loc[1][0]:]
    var[i].columns = var[i].iloc[0,:]
var = pd.concat(var)
var = var[((var['Formulas'] == 'Desk Input') |
           (var['Scale']    != 'Units'))     &
          (var['Formulas'].isna() == False) &
          (var['Mnemonics'].isna() == False)]\
         [['Mnemonics','Descriptors','Scale','Unit','Formulas']].\
             reset_index(drop=True).\
                 drop_duplicates('Mnemonics',keep='first')
var.iloc[-1] = ['EDNA','U.S. dollars per national currency units, period average', \
                'Units','U.S. Dollar','Desk Input'] # 2002 Sep vintage doesn't have ENDA
var.sort_values('Mnemonics').to_csv(r'weo_example_data/weo_varexp.csv',index=False)

# download data
c = ['132','718','941']
frequency = 'A'
df_weo  = []
country = []
for i,db in enumerate(db_list.index):
    print(db)
    df = ecos.get_ecos_sdmx_data(db, c, var['Mnemonics'],freq=frequency)
    df.index = pd.MultiIndex.from_arrays([
         [db_list['vintage'][i].to_period('M')] * df.shape[0],
         df.index.to_period(frequency)],
         names = ['vintage','time'])
    fname = 'weo_example_data/weo' + \
            str(db_list['vintage'][i].year)   + '_' + \
            str(db_list['vintage'][i].month)  + '_' + frequency + '.csv'
    df.to_csv(fname)

#%% define functions

def cond_forecast_step1(r,Tin,model=ElasticNetCV):
     """
     Parameters
     ----------
     r : numpy matrix
         (T+h) x m matrix representing input data
         the first m-k columns of T:T+h rows are nan, and the rest are not nan
         r should not contain constant column, otherwise division by std gives error
     C_dict : dictionary of numpy matrix
         T+h length dictionary, each element of which is (m-n) x m matrix representing constraints Cr = d
         n is the number of constraints
     d_dict : dictionary of numpy array
         T+h length dictionary, each element of which is (m-n) x 1 column vector representing constraints Cr = d
         n is the number of constraints
     model: sklearn linear model
         default is ElasticNetCV, but can be LassoCV, LinearRegression, etc
     Returns
     -------
     ru : numpy matrix
         (T+h) x m-k matrix representing unknown portion of the original data
     ruh_coef : dataframe
         (T+h) x m-k dataframe storing the list of non-zero coefficients names
     ruh_coefval: dataframe
         (T+h) x m-k dataframe storing the list of non-zero coefficients values
     """
     
     T = sum(~np.isnan(r).any(axis=1))
     h = r.shape[0] - T
     m = r.shape[1]
     k = sum(~np.isnan(r).any(axis=0))
     
     r = pd.DataFrame(r)
     ru = r.iloc[:,:m-k].to_numpy()
     rk = r.iloc[:,-k:].to_numpy() 
     
     # normalize data, forecast ru, normalize back ru
     ru_mean = np.nanmean(ru,axis=0)
     ru_std  = np.nanstd(ru,axis=0,ddof=1)
     ru_std[ru_std == 0] = 1 # if std is 0, division creats error, so replace it by 1
     ru_z    = (ru-ru_mean) /ru_std
     
     rk_mean = np.nanmean(rk,axis=0)
     rk_std  = np.nanstd(rk,axis=0,ddof=1)
     rk_std[rk_std == 0] = 1 # if std is 0, division creats error, so replace it by 1
     rk_z    = (rk-rk_mean) /rk_std
     
     # forecast ru
     ruh = ru_z.copy()
     ruh_coef = pd.DataFrame([],index=r.index, columns=r.columns[:ruh.shape[1]])
     ruh_coefval = pd.DataFrame([],index=r.index, columns=r.columns[:ruh.shape[1]])
     for ui in range(ru.shape[1]):   
        X_data = rk_z[:-h,:]
        y_data = ru_z[:-h,ui]
        X_pred = rk_z[-h:,:]
        #tscv = TimeSeriesSplit(X_data.shape[0]//2).split(X_data)
        tscv = TimeSeriesSplit(n_splits=Tin).split(X_data)
        fit = model(cv=tscv,max_iter=100000,n_jobs=-1).fit(X_data,y_data)
        ruh[-h:,ui] = fit.predict(X_pred)
        ruh_coef.iloc[-h:,ui] = [r.columns[ru.shape[1]:][abs(fit.coef_)>0.01]]
        ruh_coefval.iloc[-h:,ui] = [fit.coef_[abs(fit.coef_)>0.01]]
        # one can check which varibles are chosen
        # r.columns[18:][abs(fit.coef_)>10**(-17)]
        for t in range(T-Tin+1,T):
            X_data = rk_z[:t,:]
            y_data = ru_z[:t,ui]
            X_pred = rk_z[t:t+1,:]
            #tscv = TimeSeriesSplit(n_splits=X_data.shape[0]//2).split(X_data)
            tscv = TimeSeriesSplit(n_splits=Tin).split(X_data)
            fit = model(cv=tscv,max_iter=100000,n_jobs=-1).fit(X_data,y_data)
            ruh[t:t+1,ui] = fit.predict(X_pred)
            ruh_coef.iloc[t:t+1,ui] = [r.columns[ru.shape[1]:][abs(fit.coef_)>0.01].astype('object')]
            ruh_coefval.iloc[t:t+1,ui] = [fit.coef_[abs(fit.coef_)>0.01].astype('object')]
     ruh = ruh * ru_std + ru_mean
     rh  = np.concatenate((ruh,rk),axis=1)
     np.savetxt("rh.csv",rh,delimiter=",")

     return rh,ruh_coef,ruh_coefval

def cond_forecast_step2(r,rh,Tin,C_dict,d_dict):
    """
    Parameters
    ----------
    ru : numpy matrix
        (T+h) x m-k matrix representing input data
    ruh : numpy matrix
        (T+h) x m-k matrix representing stage 1 forecast
    Tin : int
        represents the length of insample forecast in ruh
    C_dict : dictionary of numpy matrix
        T+h length dictionary, each element of which is (m-n) x m matrix representing constraints Cr = d
        n is the number of constraints
    d_dict : dictionary of numpy array
        T+h length dictionary, each element of which is (m-n) x 1 column vector representing constraints Cr = d
        n is the number of constraints
    Returns
    -------
    rtil : numpy matrix
        (T+h) x m matrix, rtil fills nan of r and satisfy Cr = d
    rh : numpy matrix
        (T+h) x m matrix, rh fills nan of r but may not satisfy Cr = d
    """
    T = sum(~np.isnan(r).any(axis=1))
    h = r.shape[0] - T
    m = r.shape[1]
    k = ((r-rh)==0).all(axis=0).sum() # columns of r and rh are the same for kvar
    
    r = pd.DataFrame(r).to_numpy()
    ru = r[:,:m-k]
    rk = r[:,-k:]

    rh = pd.DataFrame(rh).to_numpy()
    ruh = rh[:,:m-k]
    
    # weight matrix
    eh  = ruh[T-Tin+1:T,:] - ru[T-Tin+1:T,:] # in-sample one-step ahead forecast error
    ehb = np.mean(eh,axis=0)
    V   = np.cov(eh,rowvar=False) # each column is a variable
    diagV = np.diag(V).copy()
    diagV[diagV == 0] = np.array(1) # replace 0 by 1 if variance = 0
    z   = ( eh-ehb ) @ np.diag( 1/np.sqrt( diagV ) ) # z = standardized e_hat
    rho = z.T @ z /(Tin-1) # same as np.corrcoef(eh,rowvar=False)
    Vrho=  np.multiply(z,z).T @ np.multiply(z,z) * Tin/(Tin-1)**3 - \
           np.multiply( rho , rho )              / (Tin*(Tin-1))
    lam = np.sum( Vrho - np.diag(np.diag(Vrho)) ) / np.sum( np.square( rho - np.diag(np.diag(rho)) ) )
    W   = lam * np.diag(np.diag(V)) + (1-lam) * V
    
    # reconcili rh by projecting it on constraint
    rutil = ru.copy()
    for hi in range(h):
        C = C_dict[T+hi]
        U = C.T[:m-k,:]
        d = d_dict[T+hi]
        rutil[T+hi:T+1+hi,:] = ( ruh[T+hi:T+1+hi,:].T - \
                                 W @ U @ inv(U.T @ W @ U) @ \
             (C @ np.concatenate( (ruh[T+hi:T+1+hi,:],rk[T+hi:T+1+hi,:]), axis=1 ).T -d) \
                                ).T
    rtil = np.concatenate((rutil,rk),axis=1)
    
    return rtil

#%% country example, france

# country names
economies = pd.read_csv(r'weo_example_data/economies.csv')
countries = pd.read_csv(r'weo_example_data/countries.csv')

# weo data, bulk if weo = pd.read_csv(r'weo_example_data/weo.csv', index_col=[0,1])
weo_var = pd.read_csv(r'weo_example_data/weo_varexp.csv')
db_list = os.listdir(r'weo_example_data')
weo = []
for i in range(len(db_list)):
    # match = re.search('weo(\d{4})(_9|_10).*_A',db_list[i]) # fall weo
    match = re.search('weo(\d{4})(_4|_5).*_A',db_list[i]) # spring weo
    if match:
        print(db_list[i])
        weo.append( pd.read_csv('weo_example_data/' + db_list[i]) )
weo = pd.concat(weo,axis=0).set_index(['vintage','time'])

df  = weo.\
        sort_index(axis=1).\
            filter( regex=('^\d{3}[\s\S]*(.A)$') ).\
                filter( regex = ('^((?!_PCH).)*$')) # drop series with .AA and _PCH
df = df.loc[:,~df.columns.duplicated()] # remove duplication440NGDP_R is still duplicated
df.index = df.index.set_levels(pd.to_datetime(df.index.levels[0]).year, level=0) # vintage to year
df.columns = [v[:v.find('.')] for v in df.columns]

# create variables from definition
df['132bca'] = df['132BCA_BP6'].multiply(df['132ENDA']) # ca in dom currency

# define variables
c = '132' # France
uvar =  [c + i for i in ['NTDD',
                        'NC','NCP',
                        'NI','NIP',
                        'NFI','NFIP',
                        'NINV',
                        'NFB',
                        'NX','NXG','NXS',
                        'NM','NMG','NMS',
                        'NSDGDP',
                        'NGS','NGSP']]
kvar = [c + i for i in ['NGDP_R','NGDP',
                        'NCG','NIG','NFIG','NGSG',
                        'bca']
        ] +\
       df.columns.to_series().filter(regex = '^' + c + '(GG|GC)').to_list()
kvar.remove(c+'GGSBP_GDP') # GGSBP is already included
allvar = uvar + kvar

fe = pd.DataFrame()
frtrue = pd.DataFrame()
first_sample = 1980
h = 1
estlag = 2 # est with lag = 0 often doesn't converge, best lag is 4
Tin = 5

uvar_num = []
kvar_num = []
kvar_list = []
ruh_coef_list = []
ruh_coefval_list = []
rctuh_coef_list = []
rctuh_coefval_list = []
# 2015 is the first year bp6 starts but the path is different from later years
# df.loc[idx[2015:,:2015],'132BCA_BP6'].unstack('vintage').loc[2000:,:]plot()
for v in range(2016,2022):
    print(v)
    vtrue = v+1
    
    # vintages above v, fix time periods
    y_vp = df.loc[idx[v:,first_sample:v],allvar].\
        dropna(axis=1)                            # drop axis with at least one nan
    y_vp = y_vp.loc[:,~ (y_vp.std() == 0)]
    
    y = y_vp.loc[idx[v,:],:]
    y.index = y.index.get_level_values('time')
    uvar_avai = list(set(uvar).intersection(y_vp.columns))
    kvar_avai = list(set(kvar).intersection(y_vp.columns))
    
    # check number of u vars are constant
    uvar_num.append(len(uvar_avai))
    kvar_num.append(len(kvar_avai))
    kvar_list.append(kvar_avai)
    
    # true
    ytrue = y_vp.loc[idx[vtrue,:],:]
    ytrue.index = y.index
   
    # weo estimator
    yweo = y.copy()
    
    # naive estimator
    ynai = y.copy()
    ynai.iloc[-h,:] = ynai.iloc[-h-1,:]

    # transform y to r
    def tf_level_data(y,uvar_avai,kvar_avai):        
        ru = y[uvar_avai].div(y[c+'NGDP'],axis=0)
        ru.columns = [vn+'_NGDP' for vn in ru.columns]
            
        pctvar = [c+vn for vn in ['NGDP','NGDP_R']]
        rk_pct = y[pctvar].pct_change()
        rk_pct.columns = [vn+'_PCH' for vn in rk_pct.columns]
        
        rk_ratio = y[list(set(kvar_avai)-set(pctvar))].div(y[c+'NGDP'],axis=0)
        rk_ratio.columns = [vn+'_NGDP' for vn in rk_ratio.columns]
        
        rk = pd.concat([rk_ratio,rk_pct],axis=1)
        
        ruvar = ru.columns
        rkvar = rk.columns        
        r = pd.concat([ru,rk],axis=1) # mix of ratio and pct_change
        
        # ratio vars take diff
        difvar = r.filter(regex='_NGDP').columns
        rdif = pd.concat([r[difvar].diff(periods =1),
                       r[list(set(r.columns)-set(difvar))]
                       ],axis=1)
            
        rdif = pd.concat([rdif[ruvar],rdif[rkvar]],axis=1)
        
        # first row is nan due to pct_ch and diff
        r = r.iloc[1:,]
        rdif = rdif.iloc[1:,]
        
        return r,rdif,ruvar,rkvar,difvar
    
    r,rdif          ,ruvar,rkvar,difvar  = tf_level_data(y,uvar_avai,kvar_avai)
    rtrue,rtruedif  ,ruvar,rkvar,difvar  = tf_level_data(ytrue,uvar_avai,kvar_avai)
    rweo,rweodif    ,ruvar,rkvar,difvar  = tf_level_data(yweo,uvar_avai,kvar_avai)
    rnai,rnaidif    ,ruvar,rkvar,difvar  = tf_level_data(ynai,uvar_avai,kvar_avai)
    
    # replace weo data by nan for estimation
    rct = rtrue.copy()
    rctdif = rtruedif.copy()
    for ri in ['r','rdif','rct','rctdif']:
        globals()[ri].loc[v,ruvar] = np.nan
    
    # augment data, and drop country code (otherwise can't diff symbol)
    def augment_lags(r,lag):
        # add Lags
        r_list = [r]
        for Li in range(1,lag+1):
            Lr = r.shift(Li)
            Lr.columns = ['L'+str(Li)+'_'+vn for vn in r.columns]
            r_list.append(Lr)
        raug = pd.concat(r_list,axis=1).iloc[lag:,:]
        return raug
    
    rdifaug   = augment_lags(rdif,estlag)
    rctdifaug = augment_lags(rctdif,estlag)
    
    # 1st stage estimation
    rdifaugh,ruh_coef,ruh_coefval = cond_forecast_step1(rdifaug,Tin)
    ruh_coef_list.append(ruh_coef.dropna(axis=0, how='all'))
    ruh_coefval_list.append(ruh_coefval.dropna(axis=0, how='all'))
    rdifaugh = pd.DataFrame(rdifaugh,
                           index   = rdifaug.index,
                           columns = rdifaug.columns)
    rctdifaugh,rctuh_coef,rctuh_coefval = cond_forecast_step1(rctdifaug,Tin)
    rctuh_coef_list.append(rctuh_coef.dropna(axis=0, how='all'))
    rctuh_coefval_list.append(rctuh_coefval.dropna(axis=0, how='all'))
    rctdifaugh = pd.DataFrame(rctdifaugh,
                           index   = rctdifaug.index,
                           columns = rctdifaug.columns)
    
    # transform data to apply constraints
    def tf_diff_data(r,rdifaugh):
        rdifh = rdifaugh[r.columns]
        rh = r.copy()
        for ci in range(v-Tin+1,v+1):
            rh.loc[ci,difvar] = rh.loc[ci-1,difvar] + rdifh.loc[ci,difvar]
        return rh
        
    rh = tf_diff_data(r,rdifaugh)
    rcth = tf_diff_data(rct,rctdifaugh)
    
    # 2nd stage estimation       
    # constraints
    sy.init_printing(pretty_print = False)
    sym_v = sy.var([vn[3:] for vn in rh.columns])
    hoge = sy.Symbol(' '.join(uvar_avai))
    C = sy.derive_by_array([
        NC_NGDP   - (NCG_NGDP  + NCP_NGDP),
        NI_NGDP   - (NIG_NGDP  + NIP_NGDP),
        NFI_NGDP  - (NFIG_NGDP + NFIP_NGDP),
        NI_NGDP   - (NFI_NGDP  + NINV_NGDP),
        NTDD_NGDP - (NC_NGDP   + NI_NGDP),
        NFB_NGDP  - (NX_NGDP   - NM_NGDP),
        NX_NGDP   - (NXG_NGDP  + NXS_NGDP),
        NM_NGDP   - (NMG_NGDP  + NMS_NGDP),
        NGS_NGDP  - (NI_NGDP   + bca_NGDP),
        NGS_NGDP  - (NGSG_NGDP + NGSP_NGDP),
        NTDD_NGDP + NFB_NGDP   + NSDGDP_NGDP
                            ], sym_v)
    C = np.array(C,dtype='float').T  
    d = np.array([0,0,0,0,0,0,0,0,0,0,1]).reshape(-1,1)
    C_dict = {}
    d_dict = {}
    for i, t in enumerate(rh.index.get_level_values('time')):
        C_dict[i] = C
        d_dict[i] = d
    
    rtil = cond_forecast_step2(r,rh,Tin,C_dict,d_dict)   
    rtil = pd.DataFrame(rtil,
                        index = r.index,
                        columns= r.columns)

    rcttil = cond_forecast_step2(rct,rcth,Tin,C_dict,d_dict)   
    rcttil = pd.DataFrame(rcttil,
                        index = r.index,
                        columns= r.columns)

    def tf_ratio_data(r,y):
        
        ru_N = r.filter(regex = '_NGDP$')
        ru_N.columns = [ vn[:vn.find('_NGDP')] for vn in ru_N.columns]
        ru_N = ru_N[list(set(ru_N.columns).intersection(uvar_avai))] # drop NCG
        
        ynew = y.copy()
        ynew.loc[v,ru_N.columns] = ru_N.loc[v,:].multiply(ynew.loc[v,c+'NGDP'])
        yu = ynew[uvar_avai]

        return yu
        
    ytil   = pd.concat([tf_ratio_data(rtil,yweo),      yweo[kvar_avai]],axis=1)
    yh     = pd.concat([tf_ratio_data(rh,yweo),        yweo[kvar_avai]],axis=1)
    ycttil = pd.concat([tf_ratio_data(rcttil,ytrue),  ytrue[kvar_avai]],axis=1)
    ycth   = pd.concat([tf_ratio_data(rcth,ytrue),    ytrue[kvar_avai]],axis=1)
    
    rtrue = rtrue
    forecast_rtrue = rtrue.iloc[-h,:].to_frame()
    frtrue = pd.concat([frtrue,forecast_rtrue.T],axis=0)
    
    all_estname = ['weo','til','h','cttil','cth','nai']
    for estname in all_estname:
        rest = globals()['r'+estname]
        forecast_error = (rest-rtrue).iloc[-h,:].to_frame()
        forecast_error.columns = pd.MultiIndex.from_arrays([[v],[estname]],
                                                           names = ['time','estimator'])
        fe = pd.concat([fe,forecast_error.T],axis=0)

# forecast error over time
fe.columns.set_names('indicator',inplace=True)
feu = fe[ruvar].unstack('estimator')
frtrue.columns.set_names('indicator',inplace=True)
fru = frtrue[ruvar] + feu

# forecast error plot for selected series
for vs in feu.columns.get_level_values('indicator').unique().sort_values():
    feu.loc[:,idx[vs,['til','weo']]].\
        droplevel(0,axis=1).\
            rename(columns = {'til':r'$\tilde r$: $2^{nd}$ step',
                              'weo':r'$r^{WEO}$: WEO'}).\
                rename_axis('',axis=1).\
                    plot(xlabel='')
    plt.axhline(y=0,color='black')
    plt.title('Forecast Error of ' + vs[3:].replace('_','/'))
    plt.savefig(results+'FR_national_accounts_cond_NG_G_'+vs[3:]+'.png',bbox_inches='tight')

print(feu.describe().loc[['mean','std'],idx[:,['cttil','til','h','weo']]].stack('estimator'))
print(fe.loc[idx[:,'weo'],:].abs().describe().loc['mean'].sort_values())

# RMSE bar charts by method
rmse = pd.DataFrame((feu**2).mean()**(1/2),columns=['RMSE']).loc[idx[:,['cttil','til','h','weo']],:]
mRMSE = rmse.\
    groupby(level='estimator').\
        mean().\
            sort_values(by = 'RMSE').\
                rename(index = {'weo':r'$r^{WEO}$: WEO',
                                'h':r'$\hat r$: $1^{st}$ step',
                                'til':r'$\tilde r$: $2^{nd}$ step',
                                'cttil':r'$\tilde r^{*}$: $2^{nd}$ step (true)'},
                       columns = {'RMSE':'mean RMSE'})
plt.bar(mRMSE.index, mRMSE['mean RMSE'], color = ['blue','red','blue','blue'])
plt.savefig(results+'FR_national_accounts_cond_NG_G.png',bbox_inches='tight')

# RMSE bar charts by indicator 
rmse_nice = rmse.unstack('estimator').\
    droplevel(0,axis=1)\
        [['til','weo']].\
            rename(columns={'til':r'$\tilde r$: $2^{nd}$ step',
                            'weo':r'$r^{WEO}$: WEO'}).\
                rename_axis('',axis=1).\
                    reset_index().\
                        sort_values('indicator').\
                            reset_index(drop=True)
rmse_nice['indicator'] = [vn[3:].replace('_','/') for vn in rmse_nice['indicator']]
rmse_nice.plot(x='indicator', xlabel='', kind='bar')
plt.savefig(results+'FR_national_accounts_cond_NG_G_all_series.png',bbox_inches='tight')

# absolute error over time
abs_err = pd.DataFrame(feu.stack('indicator').unstack('time').abs().mean(),columns=['abs_err']).unstack('estimator')
abs_err.loc[:,idx['abs_err',['weo','til']]].plot()
plt.savefig(results+'FR_national_accounts_cond_NG_G_over_time.png',bbox_inches='tight')

#%% country example, seychelles contribution

# country names
economies = pd.read_csv(r'weo_example_data/economies.csv')
countries = pd.read_csv(r'weo_example_data/countries.csv')

# weo data, bulk if weo = pd.read_csv(r'weo_example_data/weo.csv', index_col=[0,1])
weo_var = pd.read_csv(r'weo_example_data/weo_varexp.csv')
db_list = os.listdir(r'weo_example_data')
weo = []
for i in range(len(db_list)):
    # match = re.search('weo(\d{4})(_9|_10).*_A',db_list[i]) # fall weo
    match = re.search('weo(20[1-2][0-9])(_4|_5).*_A',db_list[i]) # spring weo
    if match:
        print(db_list[i])
        weo.append( pd.read_csv('weo_example_data/' + db_list[i]) )
weo = pd.concat(weo,axis=0).set_index(['vintage','time'])

df  = weo.\
        sort_index(axis=1).\
            filter( regex=('^\d{3}[\s\S]*(.A)$') ).\
                filter( regex = ('^((?!_PCH).)*$')) # drop series with .AA and _PCH
df = df.loc[:,~df.columns.duplicated()] # remove duplication440NGDP_R is still duplicated
df.index = df.index.set_levels(pd.to_datetime(df.index.levels[0]).year, level=0) # vintage to year
df.columns = [v[:v.find('.')] for v in df.columns]

# create variables from definition
df['718bca'] = df['718BCA_BP6'].multiply(df['718ENDA']) # ca in dom currency
df['718bxs'] = df['718BXS_BP6'].multiply(df['718ENDA']) # ca in dom currency

# define variables
c = '718' # seychelles
uvar =  [c + i for i in ['NTDD',
                        'NC','NCP',
                        'NI','NIP',
                        'NFI','NFIP',
                        'NINV',
                        'NFB','NGS',
                        'NX','NXG','NXS',
                        'NM','NMG','NMS',
                        'NSDGDP']]
kvar = [c + i for i in ['NGDP',
                        'NCG',
                        'NIG',
                        'NFIG',
                        'bca','bxs']] +\
       df.columns.to_series().filter(regex = '^' + c + '(GG|GC)').to_list()
allvar = uvar + kvar

fe = pd.DataFrame()
frtrue = pd.DataFrame()
first_sample = 1977
h = 1
estlag = 2 # est with lag = 0 often doesn't converge, best lag is 4
Tin = 5

uvar_num = []
kvar_num = []
kvar_list = []
ruh_coef_list = []
ruh_coefval_list = []
rctuh_coef_list = []
rctuh_coefval_list = []
for v in range(2016,2022): # 2015 is the first year bp6 starts
    print(v)
    vtrue = v+1
    
    # vintages above v, fix time periods
    y_vp = df.loc[idx[v:,first_sample:v],allvar].\
        dropna(axis=1)                            # drop axis with at least one nan
    y_vp = y_vp.loc[:,~ (y_vp.std() == 0)]
    
    y = y_vp.loc[idx[v,:],:]
    y.index = y.index.get_level_values('time')
    uvar_avai = list(set(uvar).intersection(y_vp.columns))
    kvar_avai = list(set(kvar).intersection(y_vp.columns))
    
    # check number of u vars are constant
    uvar_num.append(len(uvar_avai))
    kvar_num.append(len(kvar_avai))
    kvar_list.append(kvar_avai)
    
    # true
    ytrue = y_vp.loc[idx[vtrue,:],:]
    ytrue.index = y.index
   
    # weo estimator
    yweo = y.copy()
    
    # naive estimator
    ynai = y.copy()
    ynai.iloc[-h,:] = ynai.iloc[-h-1,:]

    def tf_level_to_contribution(y):
        r = (y - y.shift(1)).div(y[c+'NGDP'].shift(1),axis=0)
        r.columns = [v+'_con' for v in y.columns]
        return r
    
    r     = tf_level_to_contribution(y)
    rtrue = tf_level_to_contribution(ytrue)
    rweo  = tf_level_to_contribution(yweo)
    rnai  = tf_level_to_contribution(ynai)
    
    # replace weo data by nan for estimation
    rct = rtrue.copy()
    for ri in ['r','rct']:
        globals()[ri].loc[v,[v+'_con' for v in uvar_avai]] = np.nan
    
    # augment data, and drop country code (otherwise can't diff symbol)
    def augment_lags(r,lag):
        # add Lags
        r_list = [r]
        for Li in range(1,lag+1):
            Lr = r.shift(Li)
            Lr.columns = ['L'+str(Li)+'_'+vn for vn in r.columns]
            r_list.append(Lr)
        raug = pd.concat(r_list,axis=1).iloc[lag+1:,:]
        return raug
    
    raug   = augment_lags(r,estlag)
    rctaug = augment_lags(rct,estlag)
    
    # 1st stage estimation
    raugh,ruh_coef,ruh_coefval = cond_forecast_step1(raug,Tin)
    ruh_coef_list.append(ruh_coef.dropna(axis=0, how='all'))
    ruh_coefval_list.append(ruh_coefval.dropna(axis=0, how='all'))
    raugh = pd.DataFrame(raugh,
                         index   = raug.index,
                         columns = raug.columns)
    rctaugh,rctuh_coef,rctuh_coefval = cond_forecast_step1(rctaug,Tin)
    rctuh_coef_list.append(rctuh_coef.dropna(axis=0, how='all'))
    rctuh_coefval_list.append(rctuh_coefval.dropna(axis=0, how='all'))
    rctaugh = pd.DataFrame(rctaugh,
                           index   = rctaug.index,
                           columns = rctaug.columns)
            
    rh = raugh[r.columns]
    rcth = rctaugh[r.columns]

    r = r.loc[rh.index,rh.columns]
    rct = rct.loc[rcth.index,rcth.columns]
    
    # 2nd stage estimation       
    # constraints
    sy.init_printing(pretty_print = False)
    sym_v = sy.var([vn[3:] for vn in rh.columns])
    hoge = sy.Symbol(' '.join(uvar_avai))
    C = sy.derive_by_array([
        NC_con   - (NCG_con  + NCP_con),
        NI_con   - (NIG_con  + NIP_con),
        NFI_con  - (NFIG_con + NFIP_con),
        NI_con   - (NFI_con  + NINV_con),
        NTDD_con - (NC_con   + NI_con),
        NFB_con  - (NX_con   - NM_con),
        NX_con   - (NXG_con  + NXS_con),
        NM_con   - (NMG_con  + NMS_con),
        NGS_con  - (NI_con   + bca_con),
        NTDD_con + NFB_con   + NSDGDP_con - NGDP_con
                            ], sym_v)
    C = np.array(C,dtype='float').T  
    d = np.array([0,0,0,0,0,0,0,0,0,0]).reshape(-1,1)
    C_dict = {}
    d_dict = {}
    for i, t in enumerate(rh.index.get_level_values('time')):
        C_dict[i] = C
        d_dict[i] = d
    
    rtil = cond_forecast_step2(r,rh,Tin,C_dict,d_dict)   
    rtil = pd.DataFrame(rtil,
                        index = r.index,
                        columns= r.columns)

    rcttil = cond_forecast_step2(rct,rcth,Tin,C_dict,d_dict)   
    rcttil = pd.DataFrame(rcttil,
                        index = r.index,
                        columns= r.columns)

    def tf_ratio_data(r,y):
        
        ydiff = r.multiply( y[c+'NGDP'].shift(1) , axis = 0)
        ydiff.columns = [ vn[:vn.find('_')] for vn in ydiff.columns]
        ynew = ydiff + y.shift(1)
        yu = ynew[uvar_avai]

        return yu
        
    ytil   = pd.concat([tf_ratio_data(rtil,yweo),      yweo[kvar_avai]],axis=1)
    yh     = pd.concat([tf_ratio_data(rh,yweo),        yweo[kvar_avai]],axis=1)
    ycttil = pd.concat([tf_ratio_data(rcttil,ytrue),  ytrue[kvar_avai]],axis=1)
    ycth   = pd.concat([tf_ratio_data(rcth,ytrue),    ytrue[kvar_avai]],axis=1)
    
    rtrue = rtrue
    forecast_rtrue = rtrue.iloc[-h,:].to_frame()
    frtrue = pd.concat([frtrue,forecast_rtrue.T],axis=0)
    
    all_estname = ['weo','til','h','cttil','cth','nai']
    for estname in all_estname:
        rest = globals()['r'+estname]
        forecast_error = (rest-rtrue).iloc[-h,:].to_frame()
        forecast_error.columns = pd.MultiIndex.from_arrays([[v],[estname]],
                                                           names = ['time','estimator'])
        fe = pd.concat([fe,forecast_error.T],axis=0)

# forecast error over time
ruvar = [v+'_con' for v in uvar_avai]
fe.columns.set_names('indicator',inplace=True)
feu = fe[ruvar].unstack('estimator')
frtrue.columns.set_names('indicator',inplace=True)
fru = frtrue[ruvar] + feu

# forecast error plot for selected series
for vs in feu.columns.get_level_values('indicator').unique().sort_values():
    feu.loc[:,idx[vs,['til','weo']]].\
        droplevel(0,axis=1).\
            rename(columns = {'til':r'$\tilde r$: $2^{nd}$ step',
                              'weo':r'$r^{WEO}$: WEO'}).\
                rename_axis('',axis=1).\
                    plot(xlabel='')
    plt.axhline(y=0,color='black')
    plt.title('Forecast Error of ' + vs[3:].replace('_','/'))
    plt.savefig(results+'SC_national_accounts_cond_NG_G_'+vs[3:]+'.png',bbox_inches='tight')

# RMSE bar charts
print(feu.describe().loc[['mean','std'],idx[:,['cttil','til','h','weo']]].stack('estimator'))
print(fe.loc[idx[:,'weo'],:].abs().describe().loc['mean'].sort_values())

rmse = pd.DataFrame((feu**2).mean()**(1/2),columns=['RMSE']).loc[idx[:,['cttil','til','h','weo']],:]
mRMSE = rmse.\
    groupby(level='estimator').\
        mean().\
            sort_values(by = 'RMSE').\
                rename(index = {'weo':r'$r^{WEO}$: WEO',
                                'h':r'$\hat r$: $1^{st}$ step',
                                'til':r'$\tilde r$: $2^{nd}$ step',
                                'cttil':r'$\tilde r^{*}$: $2^{nd}$ step (true)'},
                       columns = {'RMSE':'mean RMSE'})
plt.bar(mRMSE.index, mRMSE['mean RMSE'], color = ['blue','red','blue','blue'])
plt.savefig(results+'SC_national_accounts_cond_NG_G.png',bbox_inches='tight')

rmse_nice = rmse.unstack('estimator').\
    droplevel(0,axis=1)\
        [['til','weo']].\
            rename(columns={'til':r'$\tilde r$: $2^{nd}$ step',
                            'weo':r'$r^{WEO}$: WEO'}).\
                rename_axis('',axis=1).\
                    reset_index().\
                        sort_values('indicator').\
                            reset_index(drop=True)
rmse_nice['indicator'] = [vn[3:vn.find('_')] for vn in rmse_nice['indicator']]
rmse_nice.plot(x='indicator', xlabel='', kind='bar')
plt.savefig(results+'SC_national_accounts_cond_NG_G_all_series.png',bbox_inches='tight')

# absolute error over time
abs_err = pd.DataFrame(feu.stack('indicator').unstack('time').abs().mean(),columns=['abs_err']).unstack('estimator')
abs_err.loc[:,idx['abs_err',['weo','til']]].plot()
