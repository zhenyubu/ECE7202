import numpy as np
import matplotlib.pyplot as plt

np.random.seed(7202)

def epsilon_greedy(true_means, T, epsilon):
    """
    Run epsilon-greedy algorithm and return cumulative rewards at each step.
    :param true_means: True mean rewards for each arm.
    :param T: Total time steps.
    :param epsilon: Exploration probability.
    :return: Array of cumulative rewards at each step.
    """
    num_arms = len(true_means)
    pulls = np.zeros(num_arms)  # 拉动次数
    total_rewards = np.zeros(num_arms)  # 累计奖励
    
    rewards = np.zeros(T)  # 每一步的即时奖励
    
    for t in range(T):
        # 决定是探索还是利用
        if np.random.rand() < epsilon:
            # 探索：随机选择一个拉杆
            arm = np.random.randint(num_arms)
        else:
            # 利用：选择平均奖励最高的拉杆
            estimated_means = total_rewards / np.maximum(pulls, 1)  # 避免除以零
            arm = np.argmax(estimated_means)
        
        # 获取奖励
        reward = np.random.normal(true_means[arm], 1)  # 从真实分布中采样奖励
        pulls[arm] += 1  # 更新拉动次数
        total_rewards[arm] += reward  # 更新累计奖励
        
        # 保存即时奖励
        rewards[t] = reward
    
    return np.cumsum(rewards)  # 返回累计奖励

def run_experiment_epsilon(T, num_arms, epsilon_values, num_runs=500):
    """
    Run epsilon-greedy experiment for different epsilon values.
    :param T: Total time steps.
    :param num_arms: Number of arms.
    :param epsilon_values: List of epsilon values to test.
    :param num_runs: Number of independent runs to average.
    :return: Dictionary of average rewards for each epsilon.
    """
    avg_rewards = {epsilon: np.zeros(T) for epsilon in epsilon_values}
    
    for epsilon in epsilon_values:
        cumulative_rewards = np.zeros((num_runs, T))
        for run in range(num_runs):
            # 生成每次实验的拉杆真实均值
            true_means = np.random.normal(0, 1, num_arms)
            # 运行 epsilon-greedy 算法
            cumulative_rewards[run] = epsilon_greedy(true_means, T, epsilon)
        
        # 计算每一步的平均奖励
        avg_rewards[epsilon] = np.mean(cumulative_rewards, axis=0) / (np.arange(1, T+1))
    
    return avg_rewards

# 参数设置
T = 1000  # 时间步数
num_arms = 10  # 拉杆数量
epsilon_values = [0.01, 0.03, 0.05, 0.07, 0.09, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]  # 不同的探索概率
# epsilon_values = np.linspace(0.01, 0.2, 20)
num_runs = 5  # 重复实验次数

# 运行实验
avg_rewards_epsilon = run_experiment_epsilon(T, num_arms, epsilon_values, num_runs)

# 绘制结果
plt.figure(figsize=(10, 6))
for epsilon, rewards in avg_rewards_epsilon.items():
    plt.plot(np.arange(1, T+1), rewards, label=f'ε = {epsilon}')
    
plt.title('Epsilon-Greedy: Average Reward vs Steps')
plt.xlabel('Steps (T)')
plt.ylabel('Average Reward')
plt.legend()
plt.savefig('./results/reward_epsilon_greedy.png') 
plt.show()



max_avg = max(values[-1] for values in avg_rewards_epsilon.values())
avg_regret = {N: max_avg - avg_rewards_epsilon[N][-1] for N in epsilon_values}

plt.figure()
plt.plot(list(avg_regret.keys()), list(avg_regret.values()), marker='o')
plt.xlabel('Parameters (epsilon)')
plt.ylabel('Average Regret')
plt.title('Average Regret vs Parameters')
plt.savefig('./results/average_regret_epsilon_greedy.png')
plt.show()