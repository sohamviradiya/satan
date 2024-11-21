import pandas as pd
import numpy as np
from scipy.optimize import minimize

def fetch_data(tickers):
    files = [f'./data/processed/{asset}.csv' for asset in tickers]
    spy = pd.read_csv('./data/spy.csv')
    djia = pd.read_csv('./data/djia.csv')
    russell = pd.read_csv('./data/russell.csv')
    return [pd.read_csv(file) for file in files],spy,djia,russell

def equal_weights(train_data,eval_data):
    relative_eval_data = eval_data/eval_data[0]
    return np.mean(relative_eval_data,axis=1)

def min_risk(train_data,eval_data):
    cov = np.cov(train_data.T)
    # objective function for optimization is w^T * cov * w
    def objective(w):
        return w.T @ cov @ w
    
    constraints = [{'type':'eq','fun':lambda w: np.sum(w)-1},]
    
    w0 = np.ones(train_data.shape[1])/train_data.shape[1]
    res = minimize(objective,w0,constraints=constraints,bounds=[(0,1)]*len(w0))
    w = res.x
    print(w)
    relative_eval_data = eval_data/eval_data[0]
    return relative_eval_data @ w
    
def risk_parity(train_data,eval_data):
    vars = np.var(train_data,axis=0)
    inv_vars = 1/vars
    w = inv_vars/np.sum(inv_vars)
    print(w)
    relative_eval_data = eval_data/eval_data[0]
    return relative_eval_data @ w

def std_methods(tickers,train_date_start,train_date_end,eval_date_start,eval_date_end):
    assets,spy,djia,russell = fetch_data(tickers)
    train_data = np.array([prices[(prices['date']>=train_date_start) & (prices['date']<=train_date_end)]['close'].values for prices in assets]).T
    eval_data = np.array([prices[(prices['date']>=eval_date_start) & (prices['date']<=eval_date_end)]['close'].values for prices in assets]).T
    spy_eval = spy[(spy['date']>=eval_date_start) & (spy['date']<=eval_date_end)]['close'].values
    djia_eval = djia[(djia['date']>=eval_date_start) & (djia['date']<=eval_date_end)]['close'].values
    russell_eval = russell[(russell['date']>=eval_date_start) & (russell['date']<=eval_date_end)]['close'].values
    return equal_weights(train_data,eval_data),min_risk(train_data,eval_data),risk_parity(train_data,eval_data),spy_eval/spy_eval[0],djia_eval/djia_eval[0],russell_eval/russell_eval[0]