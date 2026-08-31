import torch
import numpy as np
from torch.optim import Optimizer
from torch.distributions import Categorical

from agents.connect4.gae import calc_gae
from agents.connect4.ppo_networks import PiTheta, CriticNet

from dataclasses import dataclass

@dataclass
class ActorParams:
    actor: PiTheta
    optimizer: Optimizer
    learning_rate: float


@dataclass
class CriticParams:
    critic: CriticNet
    optimizer: Optimizer
    learning_rate: float


class PPOAgent:
    def __init__(self, actor: ActorParams, critic: CriticParams):
        self.actor = actor.actor
        self.critic = critic.critic
        self.actor_optimizer = actor.optimizer
        self.critic_optimizer = critic.optimizer

    def update(self, 
                states, 
                actions, 
                old_log_probs, 
                advantages, 
                returns,
                epochs,
                minibatch_size,
                eps=0.15,
                c1=0.1, 
                ):
        
        N = len(states)

        states = torch.tensor(np.array(states)).float()
        actions = torch.tensor(np.array(actions))
        old_log_probs = torch.tensor(np.array(old_log_probs))
        advantages = torch.tensor(advantages.flatten()).float()
        returns = torch.tensor(returns.flatten()).float()


        for epoch in range(epochs):
            indices = np.random.permutation(N)
            for start in range(0, N, minibatch_size):
                batch_idx = indices[start : start+minibatch_size]
                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx] 
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]

                new_action_vectors = self.actor.forward(batch_states)
                m = Categorical(logits=new_action_vectors)
                new_batch_log_probs = m.log_prob(batch_actions)
                entropy = m.entropy()

                ratio = torch.exp(new_batch_log_probs - batch_old_log_probs)
                
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1-eps, 1+eps) * batch_advantages

                policy_loss = -torch.min(surr1, surr2).mean() 

                new_values = self.critic.forward(batch_states).squeeze()
                value_loss = ((new_values - batch_returns) ** 2).mean()

                actor_loss = policy_loss - c1 * entropy.mean()
                critic_loss = value_loss

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()
