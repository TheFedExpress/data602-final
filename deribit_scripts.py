# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 13:17:42 2018

@author: pgood
"""


import requests
import pandas as pd

#scope parameter is there so we can call it to only get active options.  That will be used in flask app
def get_options(scope):
    #get all active and all inactive options.  Keep separate to make sure no
    #duplication occurs
    result = requests.get('https://www.deribit.com/api/v1/public/getinstruments')
    obj = result.json()['result']
    
    df = pd.DataFrame(obj)
    
    df = df.loc[(df.kind == 'option') &  (df.isActive == True)]
    
    if scope == 'all':
    
        result = requests.get('https://www.deribit.com/api/v1/public/getinstruments?expired=true')
        obj = result.json()['result']    
        df2 = pd.DataFrame(obj)
        
        
        return pd.concat([df, df2], axis = 0)
    else:
        return df


#Next, loop through and get every trade for each option above
def get_trades():
    hists = []
    df = get_options('all')
    for item in df['instrumentName']:
        url = 'https://www.deribit.com/api/v1/public/getlasttrades?instrument={}&count=1000'.format(item)
        print(url)
        try:
            result = requests.get(url)
            obj = result.json()['result']
            hists += obj
        except KeyError:
            pass
    trade_df = pd.DataFrame(hists)#put in dataframe for easier viewing and integrity checks before loading to mlab
    return trade_df