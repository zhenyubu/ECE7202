import numpy as np
import matplotlib.pyplot as plt

np.random.seed(7202)

def ucb(true_means, T, c):
    num_arms = len(true_means)
    pulls = np.zeros(num_arms) 
    total_rewards = np.zeros(num_arms)
    
    rewards = np.zeros(T) 

    for t in range(1, T + 1):
        if t <= num_arms:
            arm = t - 1
        else:
            ucb_values = total_rewards / np.maximum(pulls, 1) + c * np.sqrt(np.log(t) / np.maximum(pulls, 1))
            arm = np.argmax(ucb_values) 
        
        reward = np.random.normal(true_means[arm], 1)
        pulls[arm] += 1  
        total_rewards[arm] += reward  

        rewards[t - 1] = reward

    return np.cumsum(rewards)


def run_experiment_ucb(T, num_arms, c_values, num_runs=500):

    avg_rewards = {c: np.zeros(T) for c in c_values}
    
    for c in c_values:
        cumulative_rewards = np.zeros((num_runs, T))
        for run in range(num_runs):

            true_means = np.random.normal(0, 1, num_arms)
            cumulative_rewards[run] = ucb(true_means, T, c)
        
        avg_rewards[c] = np.mean(cumulative_rewards, axis=0) / (np.arange(1, T + 1))
    
    return avg_rewards