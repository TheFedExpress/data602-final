# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 18:19:19 2018

@author: pgood
"""

def strike_dist(df):
    if df.option_type == 'C':
        return (df.indexPrice - df.strike)/df.strike
    else:
        return (df.strike - df.indexPrice)/df.strike
    
def black_scholes(df, option_type):
    from scipy.stats import norm
    import numpy as np
    
    current = df.indexPrice.values
    rf = df['rf'].values
    t = (df.expiration_date.values - df.timeStamp.values).astype('timedelta64[s]').astype(float)/(365*86400)
    top_d1 = np.log(current/df.strike.values) + (rf + ((df.sd.values)**2)/2)*t
    bottom_d1 = df.sd.values*np.sqrt(t)
    d1 = top_d1/bottom_d1

    d2 = d1 - df.sd.values*np.sqrt(t)
    if option_type == 'C':
        return current*norm.cdf(d1) - df.strike.values*np.exp(-rf*t)*norm.cdf(d2)
    else:
        return df.strike.values*np.exp(-rf*t)*norm.cdf(-d2) - current*norm.cdf(-d1)



#client = MongoClient('mongodb://finalteam:datascience@ds149279.mlab.com:49279/data602final')

def get_train_data():
    from pymongo import MongoClient
    import pandas as pd

    connection = MongoClient('ds149279.mlab.com', 49279)
    db = connection['data602final']
    db.authenticate('me', 'mypass')
    
    
    #db = client.data602final
    
    df = pd.DataFrame([item for item in db.btc_options.find({})])
    sds = pd.DataFrame([item for item in db.sds.find({})])
    return df, sds
    
def prep_data(sds, df):
    import pandas as pd
    from datetime import datetime, timedelta
    from quandl import get

    
    
    get_date = lambda x: datetime.strptime(x, '%d%b%y')
    
    get_cols = lambda x: pd.Series([item for item in x.split('-')])
    cols = df['instrument'].apply(get_cols)
    cols.columns = ['uderlying', 'expiration_date', 'strike', 'option_type']
    df = pd.concat([df, cols], axis = 1)
    df.loc[:, 'expiration_date'] = df['expiration_date'].map(get_date)
    add_time = lambda x: timedelta(days = 1) - timedelta(seconds = 1) + x #make exp date end of day
    df['expiration_date'] = df['expiration_date'].map(add_time)
    
    df = df.dropna()
    df['strike'] = df['strike'].map(int)
    df['timeStamp'] = df['timeStamp'].map(lambda x: datetime.fromtimestamp(x/1000))
    df['time_left'] = (df['expiration_date'] - 
      df['timeStamp']).values.astype('timedelta64[s]').astype(float)/(365*86400)
    df['date'] = df['timeStamp'].map(lambda x: x.date())
    sds['date'] = sds['date'].map(lambda x: x.date())
    rf = get('FRED/DTB6', start_date = '2016-01-01')/100
    rf.loc[:, 'date'] = rf.index.values
    rf['date'] = rf['date'].map(lambda x: x.date())
    rf.rename(columns = {'Value': 'rf'}, inplace = True)
    
    
    df = pd.merge(df, sds, on = ['date'])
    df = pd.merge(df, rf, on = ['date'])
    
    
    calls = df.loc[df.option_type == 'C']
    puts = df.loc[df.option_type == 'P']
    
        
    call_prices = black_scholes(calls, 'C')
    put_prices = black_scholes(puts, 'P')
    calls.loc[:, 'bs_price'] = call_prices
    puts.loc[:, 'bs_price'] = put_prices
    
    df2 = pd.concat([calls, puts], axis = 0)
    
    
        
    df2['strike_dist'] = df2.apply(strike_dist, axis = 1)
    
    df2['price'] = df2.apply(lambda x: x.price * x.indexPrice, axis = 1)#option price converted to USD
    df2['price_delta'] = df2.apply(lambda x: 
        min(max((x.bs_price - x.price)/x.price, -2), 2), axis = 1)#difference between option price and
    #price predicted by BS formula
    
    df2.sort_values(['instrument', 'tradeSeq'], inplace = True)
    grouped = df2.groupby(['instrument'])['quantity'].cumsum()
    df2.loc[:, 'cum_volume'] = grouped
    df2.sort_values(['instrument', 'tradeSeq'], ascending  = False, inplace = True)
    
    #this operation should be a function for style
    #get the price 2 trades in the future.  We can play around with this number
    df2.loc[:,'fut_option_price'] = df2.groupby(['instrument']).price.shift(2)
    df2.loc[:,'fut_und_price'] = df2.groupby(['instrument']).indexPrice.shift(2)
    df2.loc[:,'fut_bs_price'] = df2.groupby(['instrument']).bs_price.shift(2)
    df2['price_change'] = (df2.fut_option_price - df2.price)/df2.price
    df2['index_change'] = df2.fut_und_price - df2.indexPrice
    df2['net_change'] = df2.apply(
            lambda x: min(max(x.price_change - (x.fut_bs_price - x.bs_price)/x.bs_price, -2), 2), axis = 1)
    
    #I'm still not 100% sure net_change is the value we want to predict.  Here's my thinking:
    #The idea is that we want measure of price change
    #that is independent of price changes in the underlying.  With net change, if the underlying didn't change,
    #future_bs_price - bs_price should be zero, so we're left with the price change.  For a call, if the underlying
    #goes up, the fut_bs_price - bs_price will be positive, and should offset the part of the price change due to
    #a change in price in the underlying.  What you're left with is any change in price due to the option valuation
    #getting closer to a fair value
    
    opt_map = {'C': 1, 'P': 2}
    
    
    df2['option_type'] = df2['option_type'].map(lambda x: opt_map[x])
    X = df2.loc[df2['fut_option_price'].notna(), ['cum_volume', 'strike_dist', 'price_delta', 'time_left', 'option_type']]
    
    Y = df2.loc[df2['fut_option_price'].notna(), ['net_change']]
    return (X, Y, df2)