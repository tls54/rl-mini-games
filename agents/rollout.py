import torch
import numpy as np

from torch.distributions import Categorical

from envs.connect4 import Connect4

def encode_board_states(env:Connect4):
    player = env.current_player
    player_state = (env.board == player).astype(int)
    opponent_state = (env.board == -player).astype(int)
    return player_state, opponent_state




class RolloutCollector:
    def __init__(self, num_envs):
        self.envs = [Connect4() for _ in range(num_envs)]
        self.last_idx_per_env = [None] * num_envs


    def collect(self, actor, critic, num_steps=2048):
        states = []
        actions = []
        log_probs = []
        rewards = []
        dones = []
        values = []
        for i in range(int(num_steps/len(self.envs))):
            for env_idx, env in enumerate(self.envs):
                board_states = encode_board_states(env)
                state = board_states
                board_states = np.stack(board_states, axis=0)
                board_states_tensor = torch.from_numpy(board_states).float().unsqueeze(0)
                action_vector = actor.forward(board_states_tensor)
                m = Categorical(logits=action_vector)
                action = m.sample()
                _, reward, done, _ = env.step(action.item())
                
                idx = len(states)

                states.append(state)
                actions.append(action)
                log_probs.append(m.log_prob(action)) 
                rewards.append(reward)
                dones.append(done)
                values.append(critic.forward(board_states_tensor))

                if done:
                    if self.last_idx_per_env[env_idx] is not None:
                        rewards[self.last_idx_per_env[env_idx]] = -1
                        dones[self.last_idx_per_env[env_idx]] = True

                    self.last_idx_per_env[env_idx] = None
                    env.reset()

                else: self.last_idx_per_env[env_idx] = idx

        return states, actions, log_probs, rewards, dones, values

                    
                


