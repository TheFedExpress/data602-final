# -*- coding: utf-8 -*-
"""
Created on Mon May  7 15:57:29 2018

@author: Peter_goodridge
"""

from deribit_scripts import get_options
from trade_analysis import strike_dist, black_scholes

available = get_options('available')



def last_trades(instrument):
    import requests
    
    url = 'https://www.deribit.com/api/v1/public/getsummary?instrument={}'.format(instrument)
    obj = requests.get(url).json()['result']
    return obj

import pandas as pd
from get_currency_info import get_current
from datetime import datetime, timedelta

from pymongo import MongoClient
import pandas as pd

connection = MongoClient('ds149279.mlab.com', 49279)
db = connection['data602final']
db.authenticate('me', 'mypass')

trades = []
for option in available['instrumentName'].values:
    trades.append(last_trades(option))

df = pd.DataFrame(trades)

df['indexPrice'] = get_current('btc', 'check')
df.rename(columns = {'instrumentName': 'instrument', 'askPrice': 'price', 'created' : 'timeStamp',
                     'volume' : 'quantity'} ,inplace = True)

sds = pd.DataFrame([item for item in db.sds.find({})])


get_date = lambda x: datetime.strptime(x, '%d%b%y')

get_cols = lambda x: pd.Series([item for item in x.split('-')])
cols = df['instrument'].apply(get_cols)
cols.columns = ['uderlying', 'expiration_date', 'strike', 'option_type']
df = pd.concat([df, cols], axis = 1)
df.loc[:, 'expiration_date'] = df['expiration_date'].map(get_date)
df['timeStamp'] = df['timeStamp'].map(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S GMT'))
df = df.loc[df.strike.notna()]
df['strike'] = df['strike'].map(int)
df['time_left'] = (df['expiration_date'] - df['timeStamp']).values.astype('timedelta64[s]').astype(float)/(365*86400)
df['date'] = df['timeStamp'].map(lambda x: x.date())
sds['date'] = sds['date'].map(lambda x: x.date() + timedelta(days = 1))

df = pd.merge(df, sds, on = ['date'])


calls = df.loc[df.option_type == 'C']
puts = df.loc[df.option_type == 'P']

    
call_prices = black_scholes(calls, 'C')
put_prices = black_scholes(puts, 'P')
calls.loc[:, 'bs_price'] = call_prices
puts.loc[:, 'bs_price'] = put_prices

df2 = pd.concat([calls, puts], axis = 0)


    
df2['strike_dist'] = df2.apply(strike_dist, axis = 1)
df2['price'] = df2.apply(lambda x: x.price * x.indexPrice, axis = 1)#option price converted to USD
df2['price_delta'] = df2.apply(lambda x: (x.bs_price - x.price)/x.indexPrice, axis = 1)#difference between option price and
df2.replace('', 0, inplace = True)
X = df2.loc[df2['quantity'] > 0, ['tradeSeq', 'strike_dist', 'price_delta', 'option_type', 'time_left']]
