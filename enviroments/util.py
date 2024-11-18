import numpy as np
import pandas as pd

columns_to_observe = ['close','Return_1d_stock','Return_3d_stock','Return_5d_stock','Return_10d_stock','Return_50d_stock','Return_100d_stock','Return_1d_Correlation','Return_3d_Correlation','Return_5d_Correlation','Return_10d_Correlation','Return_50d_Correlation','Return_100d_Correlation','ADX','Aroon_Up','Aroon_Down','CCI','EMA','KAMA','ROC','RSI','CMF','ADI','FI','Bollinger_high','Bollinger_low','Donchian_low','Donchian_high']

def fetch_observations(tickers,period,start_date,end_date,benchmark_path='./data/djia.csv'):
    files = [f'./data/processed/{asset}.csv' for asset in tickers]
    assets = [pd.read_csv(file) for file in files]
    benchmark = pd.read_csv(benchmark_path)
    assets = [asset[(asset['date'] >= start_date) & (asset['date'] <= end_date)] for asset in assets]
    benchmark = benchmark[(benchmark['date'] >= start_date) & (benchmark['date'] <= end_date)]
    dates = np.array(benchmark['date'].values)
    
    prices = [asset['close'].iloc[::period].values for asset in assets]
    
    prices_array = np.array(prices,dtype=np.float32).T
    benchmark_array = np.array(benchmark['close'].iloc[::period].values,dtype=np.float32)
    
    observations = []
    for asset in assets:
        asset_obs = []
        for column in columns_to_observe:
            asset_obs.append(np.array(asset[column].shift(1).dropna().iloc[::period].values,dtype=np.float32)) # shift by 1 to avoid look-ahead bias
            
        observations.append(asset_obs)
        
    observations_array = np.array(observations).transpose(2, 0, 1)
    return prices_array, benchmark_array, observations_array, dates
