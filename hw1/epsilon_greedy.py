import numpy as np
import matplotlib.pyplot as plt

np.random.seed(7202)

def epsilon_greedy(true_means, T, epsilon):
    num_arms = len(true_means)
    pulls = np.zeros(num_arms)  
    total_rewards = np.zeros(num_arms)  
    
    rewards = np.zeros(T) 
    
    for t in range(T):
    
        if np.random.rand() < epsilon:
            arm = np.random.randint(num_arms)
        else:
            estimated_means = total_rewards / np.maximum(pulls, 1)  
            arm = np.argmax(estimated_means)
        
        reward = np.random.normal(true_means[arm], 1)
        pulls[arm] += 1 
        total_rewards[arm] += reward  
        
        rewards[t] = reward
    
    return np.cumsum(rewards)

def run_experiment_epsilon(T, num_arms, epsilon_values, num_runs=500):
    avg_rewards = {epsilon: np.zeros(T) for epsilon in epsilon_values}
    
    for epsilon in epsilon_values:
        cumulative_rewards = np.zeros((num_runs, T))
        for run in range(num_runs):
            true_means = np.random.normal(0, 1, num_arms)
            cumulative_rewards[run] = epsilon_greedy(true_means, T, epsilon)
        
        avg_rewards[epsilon] = np.mean(cumulative_rewards, axis=0) / (np.arange(1, T+1))
    
    return avg_rewards