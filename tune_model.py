# -*- coding: utf-8 -*-
"""
Created on Fri May 11 14:39:36 2018

@author: pgood
"""


import matplotlib.pyplot as plt
import numpy as np
import sklearn.ensemble as ske
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
from sklearn.metrics import auc, roc_curve, f1_score, make_scorer, r2_score
from sklearn.model_selection import GridSearchCV,  cross_val_score, KFold
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
    if df.option_type == 1:
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
    if df.option_type == 1:
        return  df.indexPrice*norm.cdf(d1) - df.strike * np.exp(-rf*t) * norm.cdf(d2)
    else:
        return df.strike * np.exp(-rf * t) * norm.cdf(-d2) - df.indexPrice * norm.cdf(-d1)
    
def bs_reg(indexPrice,strike,vol,rf,t, option_type):
    from scipy.stats import norm
    
    top_d1 = np.log( indexPrice/strike) + (rf + ((vol)**2)/2)*t
    bottom_d1 = vol * np.sqrt(t)
    d1 = top_d1/bottom_d1

    d2 = d1 - vol * np.sqrt(t)
    if option_type == 1:
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
    

actuals = df2.loc[df2['fut_option_price'].notna(), ['price_change']].values
groups = df2.loc[df2['fut_option_price'].notna(), ['instrument']]

#colors = {1: 'red', 2: 'blue'}
plt.scatter(X['price_delta'], Y['net_change'])
#ax.scatter(X['price_delta'], Y['net_change'], c = X['option_type'].apply(lambda x: colors[x]))
plt.show()

#The group shuffle split prevents the same option from being in the train and test sets.  I'm pretty sure that will 
#help how well the model generalizes

#rs = GroupShuffleSplit(n_splits=10, test_size = .25)
rs  = KFold(n_splits=10, shuffle=True)

#gradient boost classifier
"""
param_grid = {'min_samples_leaf': [2, 5, 10],
          'min_samples_split': [2, 5, 10],
          'max_depth' : [2, 3],
          'subsample': [.8, 1],
          'max_features' : [.8, 1]
          }

y_fit = (Y.values > 0)*1
gb = GridSearchCV(GradientBoostingClassifier(n_estimators = 700), param_grid, make_scorer(f1_score),
                  cv = rs)
gb.fit(X, y_fit.ravel())
print("Best estimator found by grid search:{}".format(gb.best_estimator_))
"""


splits = rs.split(X, Y, groups = groups)
all_mets = []
for train, test in splits:
    x_train, y_train = X.iloc[train,:], Y.iloc[train,:]
    x_test, y_test, actuals_test = X.iloc[test, :],  Y.iloc[test,:], actuals[test] 

    logistic_test = (y_test.values > 0)*1
    logistic_train = (y_train.values > 0)*1

    gb = ske.GradientBoostingClassifier(min_samples_leaf= 5,
          min_samples_split= 10,
          max_depth = 3,
          subsample = .8,
          n_estimators = 1000)    
    gb.fit(x_train, logistic_train.ravel())
    probs3 = gb.predict_proba(x_test)
    
    fpr_gb, tpr_gb, _ = roc_curve(logistic_test, probs3[:,1].ravel(), 1)

    
    #rf regression (gives continuous predictions)
    regr = ske.RandomForestClassifier(min_samples_leaf = 10, n_estimators = 100, min_samples_split = 10)
    regr.fit(x_train, logistic_train.ravel())
    probs2 = regr.predict_proba(x_test)
    fpr_rf, tpr_rf, _ = roc_curve(logistic_test, probs2[:,1].ravel(), 1)
    #pred = regr.predict(x_test)
    
    
    #logistic classifier on original data
    log_regr = LogisticRegression()
    log_regr.fit(x_train, logistic_train.ravel())
    probs1 = log_regr.predict_proba(x_test)
    fpr_log, tpr_log, _ = roc_curve(logistic_test, probs1[:,1].ravel(), 1)    


    #stack
    all_probs = np.array([probs2[:,1].ravel(), probs3[:,1].ravel()]).mean(axis = 0)
    fpr_st, tpr_st, _ = roc_curve(logistic_test, all_probs.ravel(), 1)
    

    #Regressors
    #gb
    gb_reg = ske.GradientBoostingRegressor(n_estimators = 1000, 
                                      min_samples_leaf = 5, 
                                      min_samples_split = 10, 
                                      max_depth = 3
                                      )
    gb_reg.fit(x_train, y_train.values.ravel())
    
    pred1 = gb_reg.predict(x_test) 
    
    rf_reg = ske.GradientBoostingRegressor(n_estimators = 200, 
                                       min_samples_leaf = 5, 
                                       min_samples_split = 10, 
                                       max_depth = 3
                                       )
    rf_reg.fit(x_train, y_train.values.ravel())
    pred2 = rf_reg.predict(x_test) 
    #stack
    pred3 = np.array([pred1, pred2]).mean(axis = 0)

    
    #mets
    mets = { 
                'price_increase_rf': actuals_test[(probs2[:,1] > .65) & (pred1 > .05)].mean(),
                'price_increase_gb': actuals_test[(probs3[:,1] > .65) & (pred2 > .05)].mean(),
                'price_increase_log' : actuals_test[probs1[:,1] > .6].mean(),
                'price_increase_st' : actuals_test[(all_probs > .65) & (pred3 > .05)].mean(),
                'pirce_increase_mean' : actuals_test.mean(),
                'score_gb' : gb.score(x_test, logistic_test),
                'score_rf' : regr.score(x_test, logistic_test),
                'score_log' : log_regr.score(x_test, logistic_test),
                'mse_gb' : mean_squared_error(y_test.values.ravel(), pred1),
                'mse_rf' : mean_squared_error(y_test.values.ravel(), pred2),
                'mse_st' : mean_squared_error(y_test.values.ravel(), pred3),
                'r2_gb' :  r2_score(y_test.values.ravel(), pred1),
                'r2_rf' :  r2_score(y_test.values.ravel(), pred2),
                'r2_st' :  r2_score(y_test.values.ravel(), pred3),
                'auc_rf' : auc(fpr_rf, tpr_rf),
                'auc_gb' : auc(fpr_gb, tpr_gb),
                'auc_st' : auc(fpr_st, tpr_st)
            }
    all_mets.append(mets)


plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_rf, tpr_rf, label='RF')
plt.plot(fpr_log, tpr_log, label='Logistic')
plt.plot(fpr_gb, tpr_gb, label='GB')
plt.plot(fpr_st, tpr_st, label='Stack')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()


#df, sds = get_train_data()
#X, Y, df2 = prep_data(sds,df)
#auc_rf, aug_gb = test_model(df2, X, Y)