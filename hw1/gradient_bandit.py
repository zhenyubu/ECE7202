import numpy as np
import matplotlib.pyplot as plt

np.random.seed(7202)

def gradient_bandit(true_means, T, alpha):
    num_arms = len(true_means)
    H = np.zeros(num_arms)  
    avg_reward = 0 
    rewards = np.zeros(T) 
    
    for t in range(T):
        probabilities = np.exp(H) / np.sum(np.exp(H))
        
        arm = np.random.choice(num_arms, p=probabilities)
        
        reward = np.random.normal(true_means[arm], 1)
        rewards[t] = reward 
        
        avg_reward += (reward - avg_reward) / (t + 1)
        
        for i in range(num_arms):
            H[i] += alpha * (reward - avg_reward) * ((1 if i == arm else 0) - probabilities[i])
    
    return np.cumsum(rewards) 


def run_experiment_gradient_bandit(T, num_arms, alpha_values, num_runs=500):
    avg_rewards = {alpha: np.zeros(T) for alpha in alpha_values}
    
    for alpha in alpha_values:
        cumulative_rewards = np.zeros((num_runs, T))
        for run in range(num_runs):
            true_means = np.random.normal(0, 1, num_arms)
            cumulative_rewards[run] = gradient_bandit(true_means, T, alpha)
        
        avg_rewards[alpha] = np.mean(cumulative_rewards, axis=0) / (np.arange(1, T + 1))
    
    return avg_rewards