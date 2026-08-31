import torch
import numpy as np

from torch.distributions import Categorical

from envs.connect4 import Connect4
from utils.device import get_device

def encode_board_states(env:Connect4):
    player = env.current_player
    player_state = (env.board == player).astype(int)
    opponent_state = (env.board == -player).astype(int)
    return player_state, opponent_state




class RolloutCollector:
    def __init__(self, num_envs):
        self.envs = [Connect4() for _ in range(num_envs)]
        self.last_idx_per_env = [None] * num_envs
        self.device = get_device() 


    def collect(self, actor, critic, num_steps=2048):
        actor.to(self.device)
        critic.to(self.device)

        states, actions, log_probs, rewards, dones, values = [], [], [], [], [], []

        for i in range(int(num_steps / len(self.envs))):
            board_states_per_env = [encode_board_states(env) for env in self.envs]
            stacked = np.stack([np.stack(bs, axis=0) for bs in board_states_per_env], axis=0)
            batch_tensor = torch.from_numpy(stacked).float().to(device=self.device)

            action_vectors = actor.forward(batch_tensor)
            m = Categorical(logits=action_vectors)
            batch_actions = m.sample()
            batch_log_probs = m.log_prob(batch_actions)
            batch_values = critic.forward(batch_tensor)

            for env_idx, env in enumerate(self.envs):
                action = batch_actions[env_idx]
                _, reward, done, _ = env.step(action.item())

                idx = len(states)

                states.append(board_states_per_env[env_idx])
                actions.append(action)
                log_probs.append(batch_log_probs[env_idx].detach())
                rewards.append(reward)
                dones.append(done)
                values.append(batch_values[env_idx].detach())

                if done:
                    if self.last_idx_per_env[env_idx] is not None:
                        rewards[self.last_idx_per_env[env_idx]] = -1
                        dones[self.last_idx_per_env[env_idx]] = True

                    self.last_idx_per_env[env_idx] = None
                    env.reset()

                else:
                    self.last_idx_per_env[env_idx] = idx

        return states, actions, log_probs, rewards, dones, values

