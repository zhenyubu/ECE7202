import numpy as np
import matplotlib.pyplot as plt

np.random.seed(7202)

def explore_first_greedy(true_means, T, N):

    num_arms = len(true_means)
    pulls = np.zeros(num_arms)
    total_rewards = np.zeros(num_arms)
    
    rewards = np.zeros(T)
    
    for t in range(min(N, T)):
        arm = t % num_arms
        reward = max(0, np.random.normal(true_means[arm], 1))  # v_{i} = N~(mu_{i}, 1), Here we select non-negative rewards
        pulls[arm] += 1
        total_rewards[arm] += reward
        rewards[t] = reward

    # Exploitation
    if N < T:
        estimated_means = total_rewards / np.maximum(pulls, 1)  # Avoid division by zero
        best_arm = np.argmax(estimated_means)  # Select the best arm
        for t in range(N, T):
            reward = max(0, np.random.normal(true_means[best_arm], 1))  # Ensure non-negative reward
            pulls[best_arm] += 1
            total_rewards[best_arm] += reward
            rewards[t] = reward
    
    return np.cumsum(rewards)  # Return cumulative rewards

def run_experiment_N(T, num_arms, N_values, num_runs=500):
    avg_rewards = {N: np.zeros(T) for N in N_values}
    for N in N_values:
        cumulative_rewards = np.zeros((num_runs, T))
        for run in range(num_runs):
            true_means = np.random.normal(0, 1, num_arms)
            cumulative_rewards[run] = explore_first_greedy(true_means, T, N)
        
        # Compute average reward over all runs
        avg_rewards[N] = np.mean(cumulative_rewards, axis=0) / (np.arange(1, T+1))

    return avg_rewards