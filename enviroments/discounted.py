import pandas as pd
import numpy as np
from gymnasium import spaces,Env
import numpy as np
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv

columns_to_observe = ['close','Return_1d_stock','Return_3d_stock','Return_5d_stock','Return_10d_stock','Return_50d_stock','Return_100d_stock','Return_1d_Correlation','Return_3d_Correlation','Return_5d_Correlation','Return_10d_Correlation','Return_50d_Correlation','Return_100d_Correlation','ADX','Aroon_Up','Aroon_Down','CCI','EMA','KAMA','ROC','RSI','CMF','ADI','FI','Bollinger_high','Bollinger_low','Donchian_low','Donchian_high']


ACTION_SCALE = 3.0
class DiscountedEnv(Env):
    def __init__(self, observation_timeline:np.ndarray, data_timeline:np.ndarray, benchmark_timeline:np.ndarray,reward_period=10,risk_aversion=0.5,discount=0.9):
        self.num_of_assets = data_timeline.shape[1]
        self.reward_period = reward_period
        self.risk_aversion = risk_aversion
        
        self.observation_timeline = observation_timeline
        self.data_timeline = data_timeline
        self.benchmark_timeline = benchmark_timeline
        
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.num_of_assets+1,)) # +1 for cash
        
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_of_assets*len(columns_to_observe),))
        
        self.discount = discount
        
        self.reset()
        
    def reset(self,seed=None):
        self.mean_portfolio = 0
        self.mean_benchmark = 0
        self.var_portfolio = 0
        self.var_benchmark = 0
        self.var_diff = 0
        self.covar = 0
        self.current_step = 0
        self.current_worth = 1.0
        info = {"porfolio_worth":self.current_worth,"weights":np.zeros(self.num_of_assets+1)}
        return self.get_observation(),info
    
    def step(self,action):
        reward,weights,_ = self.evaluate_action(action)
        
        done = False
            
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
        log_benchmark_return = np.log(self.benchmark_timeline[self.current_step+1]/self.benchmark_timeline[self.current_step])
        
        reward = self.calculate_reward(log_return,log_benchmark_return)
        
        return reward,weights,log_return,log_benchmark_return
        
    def calculate_weights(self,action:np.ndarray) -> np.ndarray:
        scaled_action = action*ACTION_SCALE
        exp_action = np.exp(scaled_action - np.max(scaled_action))
        weights = exp_action/np.sum(exp_action)
        return np.round(weights,4)
        
    def calculate_reward(self,portfolio_return,benchmark_return):
        self.mean_portfolio = self.mean_portfolio*(self.discount) + (1-self.discount)*portfolio_return
        self.mean_benchmark = self.mean_benchmark*(self.discount) + (1-self.discount)*benchmark_return
        mean_diff = portfolio_return - benchmark_return
        
        self.var_portfolio = self.var_portfolio*(self.discount) + (1-self.discount)*(portfolio_return - self.mean_portfolio)**2
        self.var_benchmark = self.var_benchmark*(self.discount) + (1-self.discount)*(benchmark_return - self.mean_benchmark)**2
        
        self.var_diff = self.var_diff*(self.discount) + (1-self.discount)*(portfolio_return - benchmark_return)**2
        
        self.covar = self.covar*(self.discount) + (1-self.discount)*(portfolio_return - self.mean_portfolio)*(benchmark_return - self.mean_benchmark)
        
        corr = self.covar/(np.sqrt(self.var_portfolio)*np.sqrt(self.var_benchmark))
        
        return mean_diff*(1-corr) - self.risk_aversion*self.var_diff
    
    def render(self):
        print(f"Step: {self.current_step}, Prices: {self.data_timeline[self.current_step]}")
        
def fetch_observations(tickers,period,start_date,end_date):
    files = [f'./data/processed/{asset}.csv' for asset in tickers]
    assets = [pd.read_csv(file) for file in files]
    benchmark = pd.read_csv('./data/djia.csv')
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

def create_env(tickers,start_date,end_date,investment_period=1,reward_period=20,risk_aversion=0.5):
    prices, benchmark, observations, dates = fetch_observations(tickers,investment_period,start_date,end_date)
    env = DiscountedEnv(observations,prices,benchmark,reward_period,risk_aversion)
    check_env(env)
    return DummyVecEnv([lambda: env]),dates,benchmark,prices.T



