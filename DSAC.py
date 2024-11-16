from stable_baselines3 import SAC
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from torch import nn,optim


class QuantileCriticNetwork(nn.Module):
        def __init__(self, state_dim, action_dim, n_quantiles, learning_rate=3e-4):
            super(QuantileCriticNetwork, self).__init__()
            self.fc1 = nn.Linear(state_dim + action_dim, 256)
            self.fc2 = nn.Linear(256, 256)
            self.fc3 = nn.Linear(256, n_quantiles)
            self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        def forward(self, state, action):
            x = torch.cat([state, action], dim=-1)
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            quantiles = self.fc3(x)
            return quantiles
        

class DSAC(SAC):
    def __init__(self, policy, env, learning_rate=3e-4, n_quantiles=25, **kwargs):
        super(DSAC, self).__init__(policy, env, learning_rate=learning_rate, **kwargs)
        self.n_quantiles = n_quantiles
        self.critic = QuantileCriticNetwork(self.observation_space.shape[0], self.action_space.shape[0], n_quantiles)
        self.critic_target = QuantileCriticNetwork(self.observation_space.shape[0], self.action_space.shape[0], n_quantiles)

    def actor_loss(self, replay_data):
        actions, log_prob = self.actor.action_log_prob(replay_data.observations)
        quantiles = self.critic(replay_data.observations, actions)
        actor_loss = (log_prob * quantiles.mean(dim=1)).mean()
        return actor_loss
    
    def soft_update(self, source, target, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def train(self, gradient_steps: int, batch_size: int = 256):
        for gradient_step in range(gradient_steps):
            if self.replay_buffer is not None:
                replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            else:
                continue
            with torch.no_grad():                    
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                next_quantiles = self.critic_target(replay_data.next_observations, next_actions)
                target_quantiles = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_quantiles

            current_quantiles = self.critic(replay_data.observations, replay_data.actions)

            td_errors = target_quantiles - current_quantiles
            huber_loss = nn.functional.smooth_l1_loss(current_quantiles, target_quantiles, reduction='none')
            critic_loss = (torch.abs(td_errors) * huber_loss).mean()

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            actor_loss = self.actor_loss(replay_data)
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            self.soft_update(self.critic, self.critic_target, self.tau)
        
            
    
                
        
