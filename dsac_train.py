from DSAC import DSAC 

from enviroments.benchmark import create_env_benchmark
from enviroments.max_gain import create_env_max_gain


tickers = ['AAPL','JPM','F','PG', 'BA']

investment_period = 1 # each 1 investment days 
reward_period = 15 # each 15 investment periods
risk_aversions = [0,0.25,0.5,0.75,1]
discount = 0.9

learning_rate=0.001


benchmark_envs = [create_env_benchmark(tickers,'1992-01-01','2022-18-31',investment_period,reward_period,'./data/djia.csv',risk_aversion)[0] for risk_aversion in risk_aversions]
max_gain_envs = [create_env_max_gain(tickers,'1992-01-01','2022-18-31',investment_period,reward_period,risk_aversion)[0] for risk_aversion in risk_aversions]

envs = benchmark_envs + max_gain_envs

for idx,env in enumerate(envs):
    model = DSAC(policy='MlpPolicy', env=env, verbose=1, learning_rate=learning_rate, gamma=discount)
    risk_aversion = risk_aversions[idx%len(risk_aversions)]
    type = 'benchmark' if idx < len(risk_aversions) else 'max_gain'
    save_file_name = model.__class__.__name__ + f'_{type}_risk_{risk_aversion}_inv_{investment_period}_rew_{reward_period}'
    print("Learning model" + save_file_name)
    model.learn(total_timesteps=50000)
    model.save(f'./models/{save_file_name}.zip')
    del model