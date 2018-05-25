# -*- coding: utf-8 -*-
"""
Created on Sun May  6 19:51:50 2018

@author: pgood
"""


#train model

def train_model():
    
    from trade_analysis import prep_data, get_train_data
    from get_currency_info import get_current
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor

    df, sds = get_train_data()
    
    X, Y, df2 = prep_data(sds, df)
    
    gb_reg = GradientBoostingRegressor(n_estimators = 1000, 
                                      min_samples_leaf = 5, 
                                      min_samples_split = 10, 
                                      max_depth = 3
                                      )
    gb_reg.fit(X, Y.values.ravel())    
    
    y_train = (Y.values > 0)*1
    rf = RandomForestClassifier(min_samples_leaf = 10, n_estimators = 100, min_samples_split = 10)
    rf.fit(X, y_train)
     
    return rf, gb_reg

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
    from load_sds import insert_sds
    
    from pymongo import MongoClient
    
    connection = MongoClient('ds149279.mlab.com', 49279)
    db = connection['data602final']
    db.authenticate('me', 'mypass')
    
    available = get_options('available')    
    trades = []
    for option in available['instrumentName'].values:
        trades.append(last_trades(option))
    
    df = pd.DataFrame(trades)
    gb_class, gb_reg = train_model()
    #transform data
    
    df['indexPrice'] = df['uPx']
    df.rename(columns = {'instrumentName': 'instrument', 'askPrice': 'price', 'created' : 'timeStamp',
                         'volume' : 'quantity'} ,inplace = True)
    
    insert_sds()
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
    add_time = lambda x: timedelta(days = 1) - timedelta(seconds = 1) + x #make exp date end of day
    df['expiration_date'] = df['expiration_date'].map(add_time)

    df['rf'] = quandl.get('FRED/DTB6', start_date = datetime.today() - timedelta(days = 10)).values[0,-1]/100
    df['time_left'] = (df['expiration_date'] - 
      df['timeStamp']).values.astype('timedelta64[s]').astype(float)/(365*86400)
    df['date'] = df['timeStamp'].map(lambda x: x.date())
    sds['date'] = sds['date'].map(lambda x: x.date() + timedelta(days = 3))#make sure delays in price
    #history in API doesn't cause app to not work
    
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
    df2['price_delta'] = df2.apply(lambda x: 
        min(max((x.bs_price - x.price)/x.price, -2), 2), axis = 1)#difference between option price and
    df2.replace('', 0, inplace = True)
    df2 = df2.loc[df2['quantity'] > 0]
    X2 = df2.loc[:, ['quantity', 'strike_dist', 'price_delta', 'time_left', 'option_type']]
    
    #use this array to find the best options
    probs = gb_class.predict_proba(X2)
    values = gb_reg.predict(X2)
    
    return probs, values, df2

def make_table():
    import pandas as pd
    
    probs, values, df = apply_model()
    df.reset_index(inplace = True)
    probs_df = pd.DataFrame({'Probability of Increase': probs[:,1].ravel(), 
                             'Expected Increase': values.ravel()})
    info_df = df.join(probs_df, how = 'left')
    info_df = info_df.loc[(info_df['Probability of Increase'] > .65) & (info_df['Expected Increase'] > .05), 
                          ['instrument', 'Probability of Increase', 'Expected Increase']].dropna()
    
    return info_df
