# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 00:14:23 2018

@author: pgood
"""

#all functions follow the pattern: 1.get data in json format from api 2. make dataframe 3. do something with
#dataframe 4. return something

def get_current(ticker, trade_type):
    import requests
    
    url = 'https://bittrex.com/api/v1.1/public/getticker?market=USDT-{}'.format(ticker.upper())
    response = requests.get(url)
    json_text = response.json()['result']

    bid = float(json_text['Bid'])
    ask = float(json_text['Ask'])
    last = float(json_text['Last'])
    
    if trade_type in ('buy', 'cover'):
        return ask
    elif trade_type in ('sell', 'short'):
        return bid
    elif trade_type == 'check':
        return(last)
    

def get_sd(date):
    import requests
    import pandas as pd
    import numpy as np
    
    end = date.date()
    
    start = (date.date() - timedelta(days = 200))
    
    url = 'https://api.gdax.com/products/BTC-USD/candles?start={}&end={}&granularity=86400'.format(start,end)
    obj = requests.get(url).json()
    df = pd.DataFrame(obj, columns = ['time', 'low', 'high', 'open', 'close', 'volume'])   
    df['prev'] = df.close.shift(1)
    df['change'] = df.close/df.prev - 1
    return np.std(df.change.values[1:])*np.sqrt(365)
