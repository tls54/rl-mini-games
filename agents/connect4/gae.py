import numpy as np

def calc_gae(rewards, 
            values, 
            dones, 
            bootstrap_values, 
            num_envs, 
            gamma=0.99, 
            lambda_value=0.95):
    
    rewards = np.array(rewards)
    bootstrap_values = bootstrap_values.reshape(1, num_envs)
    values = np.concat([values, bootstrap_values])

    deltas = rewards + (gamma * values[1:]) - values[:-1]

    advantages = np.zeros_like(deltas)
    running_gae = np.zeros(num_envs)

    for t in reversed(range(deltas.shape[0])):
        running_gae = deltas[t] + gamma * lambda_value * (1 - dones[t]) * running_gae
        advantages[t] = running_gae

    V_target_t = advantages + values[:-1]

    return advantages, V_target_t