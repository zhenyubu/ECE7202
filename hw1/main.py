import numpy as np
import matplotlib.pyplot as plt
from explore_first_greedy import  run_experiment_N
from epsilon_greedy import run_experiment_epsilon
from ucb import run_experiment_ucb
from gradient_bandit import run_experiment_gradient_bandit

np.random.seed(7202)

T = 1000
num_arms = 10
num_runs = 100

# Parameter sets
N_values = [10, 50, 100, 200, 500]
epsilon_values = [0.01, 0.03, 0.05, 0.07, 0.09, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
c_values = [0.1, 0.5, 1.0, 2.0, 2.5, 3.0, 3.5, 4.0]
alpha_values = [0.1, 0.4, 0.7, 1.0, 1.3, 1.6, 1.9, 2.2, 2.5, 3.0]

# Generate random means and find global optimal reward
all_means = np.random.normal(0, 1, (num_runs * len(epsilon_values), num_arms))
global_optimal_reward = np.max(all_means)

# Run experiments for each algorithm
avg_N_greedy = run_experiment_N(T, num_arms, N_values, num_runs)
avg_epsilon = run_experiment_epsilon(T, num_arms, epsilon_values, num_runs)
avg_ucb = run_experiment_ucb(T, num_arms, c_values, num_runs)
avg_grad_bandit = run_experiment_gradient_bandit(T, num_arms, alpha_values, num_runs)

fig, axs = plt.subplots(4, 2, figsize=(12, 16))

# Explore First Greedy
N_regrets = [global_optimal_reward - avg_N_greedy[p][-1] for p in N_values]
axs[0, 0].plot(N_values, N_regrets, marker='*', ms = 10, mec = 'r', mfc = 'r', color='blue')
axs[0, 0].set_xlabel("N_values")
axs[0, 0].set_ylabel("Average Regret")
axs[0, 0].set_title("Explore First Greedy - Regret")

for p in N_values:
    axs[0, 1].plot(range(1, T+1), avg_N_greedy[p], label=f"N={p}")
axs[0, 1].set_title("Explore First Greedy - Reward")
axs[0, 1].set_xlabel("T")
axs[0, 1].set_ylabel("Average Reward")
axs[0, 1].legend()

# Epsilon Greedy
eps_regrets = [global_optimal_reward - avg_epsilon[p][-1] for p in epsilon_values]
axs[1, 0].plot(epsilon_values, eps_regrets, marker='*', ms = 10, mec = 'r', mfc = 'r', color='blue')
axs[1, 0].set_xlabel("epsilon_values")
axs[1, 0].set_ylabel("Average Regret")
axs[1, 0].set_title("Epsilon Greedy - Regret")

for eps in epsilon_values:
    axs[1, 1].plot(range(1, T+1), avg_epsilon[eps], label=f"eps={eps}")
axs[1, 1].set_title("Epsilon Greedy - Reward")
axs[1, 1].set_xlabel("T")
axs[1, 1].set_ylabel("Average Reward")
axs[1, 1].legend()

# UCB
ucb_regrets = [global_optimal_reward - avg_ucb[p][-1] for p in c_values]
axs[2, 0].plot(c_values, ucb_regrets, marker='*', ms = 10, mec = 'r', mfc = 'r', color='blue')
axs[2, 0].set_xlabel("c_values")
axs[2, 0].set_ylabel("Average Regret")
axs[2, 0].set_title("UCB - Regret")

for c in c_values:
    axs[2, 1].plot(range(1, T+1), avg_ucb[c], label=f"c={c}")
axs[2, 1].set_title("UCB - Reward")
axs[2, 1].set_xlabel("T")
axs[2, 1].set_ylabel("Average Reward")
axs[2, 1].legend()

# Gradient Bandit
grad_regrets = [global_optimal_reward - avg_grad_bandit[p][-1] for p in alpha_values]
axs[3, 0].plot(alpha_values, grad_regrets, marker='*', ms = 10, mec = 'r', mfc = 'r', color='blue')
axs[3, 0].set_xlabel("alpha_values")
axs[3, 0].set_ylabel("Average Regret")
axs[3, 0].set_title("Gradient Bandit - Regret")

for alpha in alpha_values:
    axs[3, 1].plot(range(1, T+1), avg_grad_bandit[alpha], label=f"alpha={alpha}")
axs[3, 1].set_title("Gradient Bandit - Reward")
axs[3, 1].set_xlabel("T")
axs[3, 1].set_ylabel("Average Reward")
axs[3, 1].legend()

plt.tight_layout()
plt.savefig("all_results.png")
plt.show()

best_N_min = min(zip(N_values, N_regrets), key=lambda x: x[1])
best_N_max = max(zip(N_values, [avg_N_greedy[v][-1] for v in N_values]), key=lambda x: x[1])
print("Explore First Greedy -> min regret:", best_N_min[1], "param:", best_N_min[0],
    "max reward:", best_N_max[1], "param:", best_N_max[0])

best_eps_min = min(zip(epsilon_values, eps_regrets), key=lambda x: x[1])
best_eps_max = max(zip(epsilon_values, [avg_epsilon[v][-1] for v in epsilon_values]), key=lambda x: x[1])
print("Epsilon Greedy -> min regret:", best_eps_min[1], "param:", best_eps_min[0],
    "max reward:", best_eps_max[1], "param:", best_eps_max[0])

best_c_min = min(zip(c_values, ucb_regrets), key=lambda x: x[1])
best_c_max = max(zip(c_values, [avg_ucb[v][-1] for v in c_values]), key=lambda x: x[1])
print("UCB -> min regret:", best_c_min[1], "param:", best_c_min[0],
    "max reward:", best_c_max[1], "param:", best_c_max[0])

best_alpha_min = min(zip(alpha_values, grad_regrets), key=lambda x: x[1])
best_alpha_max = max(zip(alpha_values, [avg_grad_bandit[v][-1] for v in alpha_values]), key=lambda x: x[1])
print("Gradient Bandit -> min regret:", best_alpha_min[1], "param:", best_alpha_min[0],
    "max reward:", best_alpha_max[1], "param:", best_alpha_max[0])



# Compare UCB and Epsilon-Greedy regret over time
plt.figure(figsize=(10, 6))
plt.plot(range(1, T+1), [global_optimal_reward - r for r in avg_ucb[1.0]], label='UCB (c=1.0)')
plt.plot(range(1, T+1), [global_optimal_reward - r for r in avg_epsilon[0.1]], label='ε-greedy (ε=0.1)')
plt.xlabel('Time Steps (T)')
plt.ylabel('Regret')
plt.title('UCB vs ε-greedy Regret Comparison')
plt.legend()
plt.savefig("regret_comparison_question_c.png")
plt.show()