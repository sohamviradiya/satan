from stable_baselines3 import SAC
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from torch import nn,optim
from stable_baselines3.common.utils import polyak_update
import numpy as np

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
    def __init__(self, policy, env, learning_rate=3e-4, n_quantiles=5, **kwargs):
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

    def train(self, gradient_steps: int, batch_size: int = 64):
        
        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        self._update_learning_rate(optimizers)


        ent_coef_losses: list[float] = []
        ent_coefs: list[float] = []
        actor_losses: list[float] = []
        critic_losses: list[float] = []

        for gradient_step in range(gradient_steps):
            if self.replay_buffer is not None:
                replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            else:
                continue
            
            if self.use_sde:
                self.actor.reset_noise()
                
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = torch.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor
                
            ent_coefs.append(ent_coef.item())
                
            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()
                
            with torch.no_grad():                    
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                next_quantiles = self.critic_target(replay_data.next_observations, next_actions)
                next_quantiles,_ = torch.min(next_quantiles, dim=0,keepdim=True)
                next_quantiles = next_quantiles - ent_coef * next_log_prob.unsqueeze(-1)
                target_quantiles = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_quantiles

            current_quantiles = self.critic(replay_data.observations, replay_data.actions)

            td_errors = target_quantiles - current_quantiles
            huber_loss = nn.functional.smooth_l1_loss(current_quantiles, target_quantiles, reduction='none')
            
            critic_loss = (torch.abs(td_errors) * huber_loss).mean()
            critic_losses.append(critic_loss.item())
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            actor_loss = self.actor_loss(replay_data)
            actor_losses.append(actor_loss.item())
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)
                
        
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))
        
            
    