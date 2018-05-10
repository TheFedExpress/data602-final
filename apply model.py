# -*- coding: utf-8 -*-
"""
Created on Sun May  6 19:51:50 2018

@author: pgood
"""


#train model

def train_model():
    
    from trade_analysis import prep_data, get_train_data
    from get_currency_info import get_current
    from sklearn.ensemble import GradientBoostingClassifier

    df, sds = get_train_data()
    
    X, Y, df2 = prep_data(sds, df)
    
    gb = GradientBoostingClassifier(n_estimators = 150, min_samples_leaf = 5, min_samples_split = 10,
                                    max_depth = 2)
    
    y_train = (Y.values > 0)*1
    gb.fit(X, y_train.ravel())
    return gb

  #get available options
def last_trades(instrument):
    import requests

    

    url = 'https://www.deribit.com/api/v1/public/getsummary?instrument={}'.format(instrument)
    obj = requests.get(url).json()['result']
    return obj

def apply_model():
    from trade_analysis import strike_dist, black_scholes
    import quandl
    from deribit_scripts import get_options    

    import pandas as pd
    from datetime import datetime, timedelta
    
    from pymongo import MongoClient
    
    connection = MongoClient('ds149279.mlab.com', 49279)
    db = connection['data602final']
    db.authenticate('me', 'mypass')
    
    available = get_options('available')    
    trades = []
    for option in available['instrumentName'].values:
        trades.append(last_trades(option))
    
    df = pd.DataFrame(trades)
    model_obj = train_model()
    #transform data
    
    df['indexPrice'] = df['uPx']
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
    df['rf'] = quandl.get('FRED/DTB6', start_date = datetime.today() - timedelta(days = 1)).values[0,0]/100
    df['time_left'] = (df['expiration_date'] - 
      df['timeStamp']).values.astype('timedelta64[s]').astype(float)/(365*86400)
    df['date'] = df['timeStamp'].map(lambda x: x.date())
    sds['date'] = sds['date'].map(lambda x: x.date() + timedelta(days = 2))
    
    df = pd.merge(df, sds, on = ['date'])
    
    
    calls = df.loc[df.option_type == 'C']
    puts = df.loc[df.option_type == 'P']
    
    
    call_prices = black_scholes(calls, 'C')
    put_prices = black_scholes(puts, 'P')
    calls.loc[:, 'bs_price'] = call_prices
    puts.loc[:, 'bs_price'] = put_prices
    
    df2 = pd.concat([calls, puts], axis = 0)
    
    opt_map = {'C': 1, 'P': 2}
    
    df2['option_type'] = df2['option_type'].map(lambda x: opt_map[x])
    df2['strike_dist'] = df2.apply(strike_dist, axis = 1)
    df2['price'] = df2.apply(lambda x: x.price * x.indexPrice, axis = 1)#option price converted to USD
    df2['price_delta'] = df2.apply(lambda x: (x.bs_price - x.price)/x.indexPrice, axis = 1)#difference between option price and
    df2.replace('', 0, inplace = True)
    X2 = df2.loc[df2['quantity'] > 0, ['quantity', 'strike_dist', 'price_delta', 'time_left', 'option_type']]
    
    #use this array to find the best options
    probs = model_obj.predict_proba(X2)
    
    return probs, df2

def make_table():
    import pandas as pd
    
    probs, df = apply_model()
    probs_df = pd.DataFrame(probs, columns = ['Probability of Increase'])
    info_df = df.join(probs_df, how = 'left')
    info_df = info_df.loc[:, ['instrument', 'Probability of Increase']]
    
    return info_df
    
