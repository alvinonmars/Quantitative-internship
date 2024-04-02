import sys
sys.path.append('/home/wangs/rs/lib')
import ff
import pandas as pd
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
import extend
from datetime import datetime
import statsmodels.api as sm
import os

def get_data(name,fre):
    data =  pd.DataFrame()
    ini_data = ff.read_binance(name)
    ini_data.index = pd.to_datetime(ini_data.index)
    data.loc[:,'v'] = ini_data.v.resample(f'{fre}T').sum()
    data.loc[:,'c'] = ini_data.c.resample(f'{fre}T').last()
    data.loc[:,'h'] = ini_data.h.resample(f'{fre}T').max()
    data.loc[:,'l'] = ini_data.l.resample(f'{fre}T').min()
    return data
    
# 得到因子Turn、PLUS
# 这里fre是数据频率，n1为换手率周期,n2为因子计算周期
def get_factor1(para):
    name,fre,n1 = para
    data = get_data(name,fre)
    data['Turn'] = data.v/data.v.rolling(n1,min_periods=1).mean()
    data['PLUS'] = (2*data.c-data.h-data.l)/data.c.shift(1)
    return data[['Turn','PLUS']]
        
def get_factor2(para):
    date_Turn,date_PLUS = para
    valid_indices = np.logical_and(~np.isnan(date_Turn), ~np.isnan(date_PLUS))
    date_Turn_valid = date_Turn[valid_indices]
    date_PLUS_valid = date_PLUS[valid_indices]

    PLUS_deTurn = np.full_like(date_Turn, np.nan)
    Turn_dePLUS = np.full_like(date_Turn, np.nan)
    # 防止所有位置均为空值的情况
    if not np.all(~valid_indices):
        date_Turn_valid_c = sm.add_constant(date_Turn_valid)
        date_PLUS_valid_c = sm.add_constant(date_PLUS_valid)
        
        PLUS_deTurn_model = sm.OLS(date_PLUS_valid, date_Turn_valid_c)
        Turn_dePLUS_model = sm.OLS(date_Turn_valid, date_PLUS_valid_c)
        
        PLUS_deTurn_results = PLUS_deTurn_model.fit()
        Turn_dePLUS_results = Turn_dePLUS_model.fit()    
    
        PLUS_deTurn[valid_indices] = PLUS_deTurn_results.resid  
        Turn_dePLUS[valid_indices] = Turn_dePLUS_results.resid  
    else:
        pass    
    return PLUS_deTurn,Turn_dePLUS
'''
换手率计算方式：v/v.rolling(n1).mean() 
参数解释:
name:货币代码
fre:数据频率
n1:换手率计算参数
n2:turn20计算参数
'''
def main():
    n1 = 3
    n2 = 4
    fre = 240
    
    folder_path = '/home/wangs/data/ba/'
    names = []
    for file in os.listdir(folder_path):
        if file.endswith('.h5'):
            file_name = os.path.splitext(file)[0] 
            names.append(file_name)
            
    get_factor1_para_lst = [(name,fre,n1) for name in names]
    turn_lst = []
    plus_lst = []
    with Pool(16) as p:
        get_factor1_res_lst = list(tqdm(p.imap(get_factor1, get_factor1_para_lst), total=len(get_factor1_para_lst)))  
    Turn = pd.DataFrame([res['Turn'] for res in get_factor1_res_lst], index=names).T
    PLUS = pd.DataFrame([res['PLUS'] for res in get_factor1_res_lst], index=names).T
    
    Turn_list = Turn.values.tolist()
    PLUS_list = PLUS.values.tolist()
    get_factor2_para_lst = [(np.array(Turn_list[i]),np.array(PLUS_list[i])) for i in range(len(Turn_list))]
    with Pool(16) as p:
        get_factor2_res_lst = list(tqdm(p.imap(get_factor2, get_factor2_para_lst), total=len(get_factor2_para_lst))) 
    PLUS_deTurn_lst,Turn_dePLUS_lst = zip(*get_factor2_res_lst)
    PLUS_deTurn_df = pd.DataFrame(PLUS_deTurn_lst,columns = Turn.columns,index = Turn.index)
    Turn_dePLUS_df = pd.DataFrame(Turn_dePLUS_lst,columns = Turn.columns,index = Turn.index)
    
    PLUS_deTurn_df_n = PLUS_deTurn_df.rolling(n2,axis=0,min_periods=1).mean()
    Turn_dePLUS_df_n = Turn_dePLUS_df.rolling(n2,axis=0,min_periods=1).mean()
    STR_dePLUS = Turn_dePLUS_df.rolling(n2,axis=0,min_periods=1).std()
    # 非负处理
    PLUS_deTurn_df_n=PLUS_deTurn_df_n.sub(PLUS_deTurn_df_n.min(axis=1), axis=0)
    Turn_dePLUS_df_n=Turn_dePLUS_df_n.sub(Turn_dePLUS_df_n.min(axis=1), axis=0)
    STR_dePLUS=STR_dePLUS.sub(STR_dePLUS.min(axis=1), axis=0)
    
    TPS = Turn_dePLUS_df_n*PLUS_deTurn_df_n
    SPS = STR_dePLUS*PLUS_deTurn_df_n
    
    ff.save('TPS_RC',TPS)
    ff.save('SPS_RC',SPS)
if __name__ == '__main__':
    main()