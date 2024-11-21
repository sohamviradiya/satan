import pandas as pd
import numpy as np
from gymnasium import spaces,Env
import numpy as np
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv


ACTION_SCALE = 3.0
class PaperEnv(Env):
    def __init__(self, observation_timeline:np.ndarray, data_timeline:np.ndarray, benchmark_timeline:np.ndarray,reward_period=10,risk_aversion=0.5):
        self.num_of_assets = data_timeline.shape[1]
        self.reward_period = reward_period
        self.risk_aversion = risk_aversion
        
        self.observation_timeline = observation_timeline
        self.data_timeline = data_timeline
        self.benchmark_timeline = benchmark_timeline
        
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.num_of_assets+1,),dtype=np.float32) # +1 for cash
        
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_of_assets*observation_timeline.shape[2],))
        
        self.portfolio_returns:list[float] = []
        self.benchmark_returns:list[float] = []
        
        self.current_step = 0
        self.current_worth = 1.0
        self.reset()
        
    def reset(self,seed=None):
        self.current_step = 0
        self.current_worth = 1.0
        info = {"portfolio_worth":self.current_worth,"weights":np.zeros(self.num_of_assets+1)}
        return self.get_observation(),info
    
    def step(self,action):
        weights,log_return = self.evaluate_action(action)
        
        done = False
        reward = self.calculate_reward()
            
        self.current_step += 1
        
        if self.current_step >= len(self.data_timeline)-1:
            done = True
            self.current_step = 0
        info = {"portfolio_worth":self.current_worth,"weights":weights,"return":log_return}
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
        self.portfolio_returns.append(log_return)
        self.benchmark_returns.append(log_benchmark_return)
        
        return weights,log_return
        
    def calculate_weights(self,action:np.ndarray) -> np.ndarray:
        scaled_action = action*ACTION_SCALE
        exp_action = np.exp(scaled_action - np.max(scaled_action))
        weights = exp_action/np.sum(exp_action)
        return np.round(weights,4)
        
    def calculate_reward(self):
        if len(self.portfolio_returns) < self.reward_period:
            return 0
        diff = np.array(self.portfolio_returns) - np.array(self.benchmark_returns)
        curr_diff = diff[-1]
        var_diff = np.var(diff)
        var_port = np.var(self.portfolio_returns)
        var_bench = np.var(self.benchmark_returns)
        
        if var_port == 0:
            var_port = 1e-8
        if var_bench == 0:
            var_bench = 1e-8
        corr = np.cov(self.portfolio_returns,self.benchmark_returns)[0,1]/(np.sqrt(var_port*var_bench))
        
        self.portfolio_returns.pop(0)
        self.benchmark_returns.pop(0)
        
        return curr_diff*(1-corr)/var_diff
    
    def render(self):
        print(f"Step: {self.current_step}, Prices: {self.data_timeline[self.current_step]}")
        
def create_env_paper(prices, benchmark, observations,reward_period=20):
    env = PaperEnv(observations,prices,benchmark,reward_period)
    check_env(env)
    return DummyVecEnv([lambda: env])



