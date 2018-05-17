# -*- coding: utf-8 -*-
"""
Created on Fri May 11 14:39:36 2018

@author: pgood
"""


import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
from sklearn.metrics import auc, roc_curve, f1_score, make_scorer
from sklearn.model_selection import GridSearchCV
from trade_analysis import get_train_data, prep_data

df, sds = get_train_data()
X, Y, df2 = prep_data(sds, df)

def black_scholes(df, option_type):
    from scipy.stats import norm
    import numpy as np
    
    current = df.indexPrice.values
    rf = df['rf'].values
    t = (df.expiration_date.values - df.timeStamp.values).astype('timedelta64[s]').astype(float)/(365*86400)
    top_d1 = np.log(current/df.strike.values) + (rf + ((df.sd.values)**2)/2)*t
    bottom_d1 = df.sd.values*np.sqrt(t)
    d1 = top_d1/bottom_d1

    d2 = d1 - df.sd.values * np.sqrt(t)
    if option_type == 'C':
        return current*norm.cdf(d1) - df.strike.values * np.exp(-rf*t) * norm.cdf(d2)
    else:
        return df.strike.values * np.exp(-rf * t) * norm.cdf(-d2) - current * norm.cdf(-d1)

def impl_vol_func(vol, df):
    from scipy.stats import norm
    
    rf = df['rf']
    t = df['time_left']
    top_d1 = np.log( df.indexPrice/df.strike) + (rf + ((vol)**2)/2)*t
    bottom_d1 = vol * np.sqrt(t)
    d1 = top_d1/bottom_d1

    d2 = d1 - vol * np.sqrt(t)
    if df.option_type == 'C':
        return  df.price - (df.indexPrice*norm.cdf(d1) - df.strike * np.exp(-rf*t) * norm.cdf(d2))
    else:
        return df.price - (df.strike * np.exp(-rf * t) * norm.cdf(-d2) - df.indexPrice * norm.cdf(-d1))
    
def bs_price(vol, df):
    from scipy.stats import norm
    
    rf = df['rf']
    t = df['time_left']
    top_d1 = np.log( df.indexPrice/df.strike) + (rf + ((vol)**2)/2)*t
    bottom_d1 = vol * np.sqrt(t)
    d1 = top_d1/bottom_d1

    d2 = d1 - vol * np.sqrt(t)
    if df.option_type == 'C':
        return  df.indexPrice*norm.cdf(d1) - df.strike * np.exp(-rf*t) * norm.cdf(d2)
    else:
        return df.strike * np.exp(-rf * t) * norm.cdf(-d2) - df.indexPrice * norm.cdf(-d1)
    
def bs_reg(indexPrice,strike,vol,rf,t, option_type):
    from scipy.stats import norm
    
    top_d1 = np.log( indexPrice/strike) + (rf + ((vol)**2)/2)*t
    bottom_d1 = vol * np.sqrt(t)
    d1 = top_d1/bottom_d1

    d2 = d1 - vol * np.sqrt(t)
    if option_type == 'C':
        return  indexPrice*norm.cdf(d1) - strike * np.exp(-rf*t) * norm.cdf(d2)
    else:
        return strike * np.exp(-rf * t) * norm.cdf(-d2) - indexPrice * norm.cdf(-d1)
    
def vega(vol, df):
    from scipy.stats import norm
    
    rf = df['rf']
    t = df['time_left']

    top_d1 = np.log(df.indexPrice/df.strike) + (rf + ((vol)**2)/2)*t
    bottom_d1 = vol * np.sqrt(t)
    d1 = top_d1/bottom_d1
    return df.indexPrice * np.sqrt(t) * norm.cdf(d1, 0.0, 1.0)

def loop_vol (df):
    
    sigma = 1
    for i in range(50):
        diff = (bs_price(sigma, df) - df.price)
        if abs(diff) <= .001:
            break
        sigma -= diff / vega(sigma, df)
    
    return sigma

def get_implied_vol(df):
    from scipy.optimize import fsolve
    
    impl_vol = fsolve(impl_vol_func, df['sd'], args = (df,), fprime = vega)
    
    return impl_vol[0]
    
"""
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
probs_regr = []
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
              'max_depth' : [3],
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
    regr = RandomForestClassifier(min_samples_leaf = 5, n_estimators = 100, min_samples_split = 10)
    regr.fit(x_train, logistic_train.ravel())
    scores_regr.append(regr.score(x_test, logistic_test))
    probs2 = regr.predict_proba(x_test)
    probs_regr.append(probs2)
    increase_rf.append(actuals_test[probs2[:,1] > .65].mean())
    #pred = regr.predict(x_test)
    
    """
    #logistic to classify RF predictions
    logistic = LogisticRegression()
    logistic.fit(pred.reshape(-1,1), logistic_test.ravel())
    probs2 = logistic.predict_proba(pred.reshape(-1,1))
    all_probs.append(probs2)
    increase_rf.append(actuals_test[probs2[:,1] > .6].mean())#find avg price change when classifier prob > .6
    scores_logistic.append(logistic.score(pred.reshape(-1,1), logistic_test))
    """
    
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

#df, sds = get_train_data()
#X, Y, df2 = prep_data(sds,df)
#auc_rf, aug_gb = test_model(df2, X, Y)
"""