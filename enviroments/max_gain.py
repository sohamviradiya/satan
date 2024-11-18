import pandas as pd
import numpy as np
from gymnasium import spaces,Env
import numpy as np
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv

columns_to_observe = ['close','Return_1d_stock','Return_3d_stock','Return_5d_stock','Return_10d_stock','Return_50d_stock','Return_100d_stock','Return_1d_Correlation','Return_3d_Correlation','Return_5d_Correlation','Return_10d_Correlation','Return_50d_Correlation','Return_100d_Correlation','ADX','Aroon_Up','Aroon_Down','CCI','EMA','KAMA','ROC','RSI','CMF','ADI','FI','Bollinger_high','Bollinger_low','Donchian_low','Donchian_high']

ACTION_SCALE = 3.0
class MaxGainEnv(Env):
    def __init__(self, observation_timeline:np.ndarray, data_timeline:np.ndarray, reward_period=10,risk_aversion=0.5):
        self.num_of_assets = data_timeline.shape[1]
        self.reward_period = reward_period
        self.risk_aversion = risk_aversion
        
        self.observation_timeline = observation_timeline
        self.data_timeline = data_timeline
        
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.num_of_assets+1,),dtype=np.float32) # +1 for cash
        
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_of_assets*len(columns_to_observe),))
        
        self.portfolio_returns:list[float] = []
        self.max_returns:list[float] = []
        
        self.current_step = 0
        self.current_worth = 1.0
        self.reset()
        
    def reset(self,seed=None):
        self.current_step = 0
        self.current_worth = 1.0
        info = {"porfolio_worth":self.current_worth,"weights":np.zeros(self.num_of_assets+1)}
        return self.get_observation(),info
    
    def step(self,action):
        weights = self.evaluate_action(action)
        
        done = False
        reward = self.calculate_reward()
            
        self.current_step += 1
        
        if self.current_step >= len(self.data_timeline)-1:
            done = True
            self.current_step = 0
        info = {"porfolio_worth":self.current_worth,"weights":weights}
        return self.get_observation(), reward, done,False , info
    
    def get_observation(self):
        return self.observation_timeline[self.current_step].flatten()

    def evaluate_action(self,action:np.ndarray):
        weights = self.calculate_weights(action)
        asset_wise_returns = np.zeros(self.num_of_assets+1)
        asset_wise_returns[:-1] = self.data_timeline[self.current_step+1]/self.data_timeline[self.current_step]
        asset_wise_returns[-1] = 1 
        portfolio_return = np.dot(weights,asset_wise_returns)
        self.current_worth *= portfolio_return
        log_return = np.log(portfolio_return)
        log_max_return = np.log(np.max(asset_wise_returns))
        self.portfolio_returns.append(log_return)
        self.max_returns.append(log_max_return)
        return weights,log_return,log_max_return
        
    def calculate_weights(self,action:np.ndarray) -> np.ndarray:
        scaled_action = action*ACTION_SCALE
        exp_action = np.exp(scaled_action - np.max(scaled_action))
        weights = exp_action/np.sum(exp_action)
        return np.round(weights,4)
        
    def calculate_reward(self):
        if len(self.portfolio_returns) < self.reward_period:
            return 0
        diff = np.array(self.portfolio_returns) - np.array(self.max_returns)
        mean_diff = np.mean(diff)
        var_diff = np.var(diff)
        
        self.portfolio_returns.pop(0)
        self.max_returns.pop(0)
        
        return mean_diff - self.risk_aversion*var_diff
    
    def render(self):
        print(f"Step: {self.current_step}, Prices: {self.data_timeline[self.current_step]}")
        
def fetch_observations(tickers,period,start_date,end_date):
    files = [f'./data/processed/{asset}.csv' for asset in tickers]
    assets = [pd.read_csv(file) for file in files]
    assets = [asset[(asset['date'] >= start_date) & (asset['date'] <= end_date)] for asset in assets]
    
    dates = np.array(assets[0]['date'].values)
    
    prices = [asset['close'].iloc[::period].values for asset in assets]
    
    prices_array = np.array(prices,dtype=np.float32).T
    
    observations = []
    for asset in assets:
        asset_obs = []
        for column in columns_to_observe:
            asset_obs.append(np.array(asset[column].shift(1).dropna().iloc[::period].values,dtype=np.float32)) # shift by 1 to avoid look-ahead bias
            
        observations.append(asset_obs)
        
    observations_array = np.array(observations).transpose(2, 0, 1)
    return prices_array, observations_array, dates

def create_env_max_gain(tickers,start_date,end_date,investment_period=1,reward_period=20,risk_aversion=0.5):
    prices, observations, dates = fetch_observations(tickers,investment_period,start_date,end_date)
    env = MaxGainEnv(observations,prices,reward_period,risk_aversion)
    check_env(env)
    return DummyVecEnv([lambda: env]),dates,prices.T



