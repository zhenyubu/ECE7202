import numpy as np
import matplotlib.pyplot as plt

np.random.seed(7202)

def explore_first_greedy(true_means, T, N):

    num_arms = len(true_means)
    pulls = np.zeros(num_arms)
    total_rewards = np.zeros(num_arms)
    
    rewards = np.zeros(T)  # Store rewards at each time step
    
    # Exploration
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
            # Generate true means for arms
            true_means = np.random.normal(0, 1, num_arms)
            # Run explore-first greedy with non-negative rewards
            cumulative_rewards[run] = explore_first_greedy(true_means, T, N)
        
        # Compute average reward over all runs
        avg_rewards[N] = np.mean(cumulative_rewards, axis=0) / (np.arange(1, T+1))
    
    return avg_rewards


# Parameters
T = 1000  # Total time steps
num_arms = 10  # Number of arms
N_values = [10, 50, 100, 200, 500]  # Different exploration steps
num_runs = 5  # Number of runs for averaging

# Run the experiment
avg_rewards_N_nonnegative = run_experiment_N(T, num_arms, N_values, num_runs)

# Plot the results
plt.figure(figsize=(10, 6))
for N, rewards in avg_rewards_N_nonnegative.items():
    plt.plot(np.arange(1, T+1), rewards, label=f'N = {N}')
    
plt.title('Explore-First Greedy (Non-Negative Rewards): Average Reward vs Steps')
plt.xlabel('Steps (T)')
plt.ylabel('Average Reward')
plt.legend()
plt.savefig('./results/reward_explore_first_greedy.png') 
plt.show()


max_avg = max(values[-1] for values in avg_rewards_N_nonnegative.values())
avg_regret = {N: max_avg - avg_rewards_N_nonnegative[N][-1] for N in N_values}

plt.figure()
plt.plot(list(avg_regret.keys()), list(avg_regret.values()), marker='o')
plt.xlabel('Parameters (N)')
plt.ylabel('Average Regret')
plt.title('Average Regret vs Parameters')
plt.savefig('./results/average_regret_explore_first_greedy.png')
plt.show()