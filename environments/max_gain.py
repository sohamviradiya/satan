import pandas as pd
import numpy as np
from gymnasium import spaces,Env
import numpy as np
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv

ACTION_SCALE = 3.0
class MaxGainEnv(Env):
    def __init__(self, observation_timeline:np.ndarray, data_timeline:np.ndarray, reward_period=10,risk_aversion=0.5):
        self.num_of_assets = data_timeline.shape[1]
        self.reward_period = reward_period
        self.risk_aversion = risk_aversion
        
        self.observation_timeline = observation_timeline
        self.data_timeline = data_timeline
        
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.num_of_assets+1,),dtype=np.float32) # +1 for cash
        
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_of_assets*observation_timeline.shape[2],))
        
        self.portfolio_returns:list[float] = []
        self.max_returns:list[float] = []
        
        self.current_step = 0
        self.current_worth = 1.0
        self.reset()
        
    def reset(self,seed=None):
        self.current_step = 0
        self.current_worth = 1.0
        info = {"portfolio_worth":self.current_worth,"weights":np.zeros(self.num_of_assets+1)}
        return self.get_observation(),info
    
    def step(self,action):
        weights = self.evaluate_action(action)
        
        done = False
        reward = self.calculate_reward()
            
        self.current_step += 1
        
        if self.current_step >= len(self.data_timeline)-1:
            done = True
            self.current_step = 0
        info = {"portfolio_worth":self.current_worth,"weights":weights}
        return self.get_observation(), reward, done,False , info
    
    def get_observation(self):
        return self.observation_timeline[self.current_step].flatten()

    def evaluate_action(self,action:np.ndarray):
        weights = self.calculate_weights(action)
        asset_wise_returns = np.zeros(self.num_of_assets+1)
        asset_wise_returns[:-1] = self.data_timeline[self.current_step+1]/(self.data_timeline[self.current_step] + 1e-8)
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
        curr_diff = diff[-1]
        var = np.var(self.portfolio_returns)
        
        self.portfolio_returns.pop(0)
        self.max_returns.pop(0)
        
        return curr_diff - self.risk_aversion*var
    
    def render(self):
        print(f"Step: {self.current_step}, Prices: {self.data_timeline[self.current_step]}")
        
def create_env_max_gain(prices, observations,reward_period=20,risk_aversion=0.5):
    env = MaxGainEnv(observations,prices,reward_period,risk_aversion)
    check_env(env)
    return DummyVecEnv([lambda: env])



