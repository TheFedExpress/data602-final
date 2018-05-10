# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 18:19:19 2018

@author: pgood
"""
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
from sklearn.metrics import auc, roc_curve, f1_score, make_scorer
from sklearn.model_selection import GridSearchCV

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



def test_model(df2, X, Y):
    import matplotlib.pyplot as plt
    
    actuals = df2.loc[df2['fut_option_price'].notna(), ['price_change']].values
    groups = df2.loc[df2['fut_option_price'].notna(), ['instrument']]
    
    #colors = {1: 'red', 2: 'blue'}
    plt.scatter(X['price_delta'], Y['net_change'])
    #ax.scatter(X['price_delta'], Y['net_change'], c = X['option_type'].apply(lambda x: colors[x]))
    plt.show()
    
    #The group shuffle split prevents the same option from being in the train and test sets.  I'm pretty sure that will 
    #help how well the model generalizes
    
    rs = GroupShuffleSplit(n_splits=10, test_size = .25)
    splits = rs.split(X, Y, groups = groups)
    
    #Probably need to clean all these lists up with some DFs or dictionaries
    residuals = []
    probs_log = []
    scores_regr = []
    scores_logistic = []
    all_probs = []
    probs_log_main = []
    scores_log_main = []
    increase_rf = []
    increase_log = []
    increase_gb = []
    scores_gb = []
    probs_gb = []
    for train, test in splits:
        x_train, y_train = X.iloc[train,:], Y.iloc[train,:]
        x_test, y_test, actuals_test = X.iloc[test, :],  Y.iloc[test,:], actuals[test] 
    
        logistic_test = (y_test.values > 0)*1
        logistic_train = (y_train.values > 0)*1
    
        #gradient boost classifier
        param_grid = {'min_samples_leaf': [5],
                  'min_samples_split': [10],
                  'max_depth' : [2],
                  'subsample': [1],
                  'max_features' : [1]
                  }
        
        gb = GridSearchCV(GradientBoostingClassifier(n_estimators = 150), param_grid, make_scorer(f1_score))
        gb.fit(x_train, logistic_train.ravel())
        scores_gb.append(gb.score(x_test, logistic_test))
        probs3 = gb.predict_proba(x_test)
        probs_gb.append(probs3)
        increase_gb.append(actuals_test[probs3[:,1] > .65].mean())
        print("Best estimator found by grid search:{}".format(gb.best_estimator_))
        
        #rf regression (gives continuous predictions)
        regr = RandomForestRegressor(min_samples_leaf = 5, n_estimators = 50)
        regr.fit(x_train, y_train.net_change.values.ravel())
        scores_regr.append(regr.score(x_test, y_test.net_change.values.ravel()))
        pred = regr.predict(x_test)
        residuals.append(mean_squared_error(y_test, pred))
        
        #logistic to classify RF predictions
        logistic = LogisticRegression()
        logistic.fit(pred.reshape(-1,1), logistic_test.ravel())
        probs2 = logistic.predict_proba(pred.reshape(-1,1))
        all_probs.append(probs2)
        increase_rf.append(actuals_test[probs2[:,1] > .6].mean())#find avg price change when classifier prob > .6
        scores_logistic.append(logistic.score(pred.reshape(-1,1), logistic_test))
        
        #logistic classifier on original data
        log_regr = LogisticRegression()
        log_regr.fit(x_train, logistic_train.ravel())
        probs1 = log_regr.predict_proba(x_test)
        probs_log_main.append(probs1)
        scores_log_main.append(log_regr.score(x_test, logistic_test))
        increase_log.append(actuals_test[probs1[:,1] > .6].mean())#find avg price change when classifier prob > .6
        
    fpr_rf, tpr_rf, _ = roc_curve(logistic_test, probs2[:,1].ravel(), 1)
    fpr_log, tpr_log, _ = roc_curve(logistic_test, probs1[:,1].ravel(), 1)
    fpr_gb, tpr_gb, _ = roc_curve(logistic_test, probs3[:,1].ravel(), 1)
    
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_rf, tpr_rf, label='RF')
    plt.plot(fpr_log, tpr_log, label='Logistic')
    plt.plot(fpr_gb, tpr_gb, label='GB')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()
    
    auc_rf = auc(fpr_rf, tpr_rf)
    auc_gb = auc(fpr_gb, tpr_gb)
    
    mse = np.array(residuals).mean()
    return auc_rf, auc_gb

#df, sds = get_train_data()
#X, Y, df2 = prep_data(sds,df)
#auc_rf, aug_gb = test_model(df2, X, Y)