# %load TPS_SPS.py
import sys
sys.path.append('/home/wangs/rs/lib')
'''sys.path.append('/home/wangs/rs/lwm/lib')'''
import ff
import pandas as pd
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
import extend
# from scipy.stats import pearsonr
import warnings
import logging
import statsmodels.api as sm
warnings.filterwarnings('ignore')
opt.logging.set_verbosity(opt.logging.WARNING)


def get_factor1(para):
    name,fre,n1 = para
    data = ff.read_ba(name,k = f'{fre}H')
    data['Turn'] = data.v/data.v.rolling(n1,min_periods=1).mean()
    # data['Turn_n'] = data.Turn.rolling(n2).mean() # 对应Turn20
    # data['STR'] = data.Turn.rolling(n2).std() # 对应STR
    data['PLUS'] = (2*data.c-data.h-data.l)/data.c.shift(1)
    return data[['Turn','PLUS']]
        
def get_factor2(para):
    i,date_Turn,date_PLUS = para
    valid_indices = np.logical_and(~np.isnan(date_Turn), ~np.isnan(date_PLUS))
    date_Turn_valid = date_Turn[valid_indices]
    date_PLUS_valid = date_PLUS[valid_indices]

    PLUS_deTurn = np.full_like(date_Turn, np.nan)
    Turn_dePLUS = np.full_like(date_Turn, np.nan)
    # 防止所有位置均为空值的情况
    try:
        date_Turn_valid_c = sm.add_constant(date_Turn_valid)
        date_PLUS_valid_c = sm.add_constant(date_PLUS_valid)
        
        PLUS_deTurn_model = sm.OLS(date_PLUS_valid, date_Turn_valid_c)
        Turn_dePLUS_model = sm.OLS(date_Turn_valid, date_PLUS_valid_c)
        
        PLUS_deTurn_results = PLUS_deTurn_model.fit()
        Turn_dePLUS_results = Turn_dePLUS_model.fit()    
    
        PLUS_deTurn[valid_indices] = PLUS_deTurn_results.resid  
        Turn_dePLUS[valid_indices] = Turn_dePLUS_results.resid  
    except:
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
    fre = 4
    n1 = 3
    n2 = 4
    get_factor1_para_lst = [(name,fre,n1) for name in ff.symbols]
    turn_lst = []
    plus_lst = []
    with Pool(16) as p:
        get_factor1_res_lst = list(tqdm(p.imap(get_factor1, get_factor1_para_lst), total=len(get_factor1_para_lst)))  
    Turn = pd.DataFrame([res['Turn'] for res in get_factor1_res_lst], index=ff.symbols).T
    PLUS = pd.DataFrame([res['PLUS'] for res in get_factor1_res_lst], index=ff.symbols).T
    
    Turn_list = Turn.values.tolist()
    PLUS_list = PLUS.values.tolist()
    get_factor2_para_lst = [(i,np.array(Turn_list[i]),np.array(PLUS_list[i])) for i in range(len(Turn_list))]
    with Pool(16) as p:
        get_factor2_res_lst = list(tqdm(p.imap(get_factor2, get_factor2_para_lst), total=len(get_factor2_para_lst))) 
    PLUS_deTurn_lst, Turn_dePLUS_lst = zip(*get_factor2_res_lst)

    PLUS_deTurn_df = pd.DataFrame(PLUS_deTurn_lst,columns = Turn.columns,index = Turn.index)
    Turn_dePLUS_df = pd.DataFrame(Turn_dePLUS_lst,columns = Turn.columns,index = Turn.index)

    PLUS_deTurn_df_n = PLUS_deTurn_df.rolling(n2,axis=0,min_periods=1).mean()
    Turn_dePLUS_df_n = Turn_dePLUS_df.rolling(n2,axis=0,min_periods=1).mean()
    STR_dePLUS = Turn_dePLUS_df.rolling(n2,axis=0,min_periods=1).std()
    # 非负处理
    PLUS_deTurn_df_n=PLUS_deTurn_df_n.apply(lambda row: row - row.min(), axis=1)
    Turn_dePLUS_df_n=Turn_dePLUS_df_n.apply(lambda row: row - row.min(), axis=1)
    STR_dePLUS=STR_dePLUS.apply(lambda row: row - row.min(), axis=1)
    
    TPS = Turn_dePLUS_df_n*PLUS_deTurn_df_n
    SPS = STR_dePLUS*PLUS_deTurn_df_n
    
    ff.save('TPS_RC',TPS)
    ff.save('SPS_RC',SPS)
if __name__ == '__main__':
    main()