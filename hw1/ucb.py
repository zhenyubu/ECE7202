import numpy as np
import matplotlib.pyplot as plt

np.random.seed(7202)

def ucb(true_means, T, c):
    """
    Run UCB algorithm and return cumulative rewards at each step.
    :param true_means: True mean rewards for each arm.
    :param T: Total time steps.
    :param c: Exploration weight (UCB parameter).
    :return: Array of cumulative rewards at each step.
    """
    num_arms = len(true_means)
    pulls = np.zeros(num_arms)  # 每个拉杆的拉动次数
    total_rewards = np.zeros(num_arms)  # 每个拉杆的累计奖励
    
    rewards = np.zeros(T)  # 每一步的即时奖励

    for t in range(1, T + 1):
        # 初始化阶段：每个拉杆至少拉动一次
        if t <= num_arms:
            arm = t - 1
        else:
            # 计算每个拉杆的UCB值
            ucb_values = total_rewards / np.maximum(pulls, 1) + c * np.sqrt(np.log(t) / np.maximum(pulls, 1))
            arm = np.argmax(ucb_values)  # 选择UCB值最大的拉杆
        
        # 拉动选定的拉杆，采样奖励
        reward = np.random.normal(true_means[arm], 1)
        pulls[arm] += 1  # 更新拉动次数
        total_rewards[arm] += reward  # 更新累计奖励

        # 保存即时奖励
        rewards[t - 1] = reward

    return np.cumsum(rewards)  # 返回累计奖励


def run_experiment_ucb(T, num_arms, c_values, num_runs=500):
    """
    Run UCB experiment for different c values.
    :param T: Total time steps.
    :param num_arms: Number of arms.
    :param c_values: List of c values to test.
    :param num_runs: Number of independent runs to average.
    :return: Dictionary of average rewards for each c.
    """
    avg_rewards = {c: np.zeros(T) for c in c_values}
    
    for c in c_values:
        cumulative_rewards = np.zeros((num_runs, T))
        for run in range(num_runs):
            # 生成拉杆的真实均值
            true_means = np.random.normal(0, 1, num_arms)
            # 运行UCB算法
            cumulative_rewards[run] = ucb(true_means, T, c)
        
        # 计算每一步的平均奖励
        avg_rewards[c] = np.mean(cumulative_rewards, axis=0) / (np.arange(1, T + 1))
    
    return avg_rewards


# 参数设置
T = 1000  # 时间步数
num_arms = 10  # 拉杆数量
c_values = [0.1, 0.5, 1.0, 2.0, 2.5, 3.0, 3.5, 4.0]  # 不同的探索权重
num_runs = 5  # 重复实验次数

# 运行实验
avg_rewards_ucb = run_experiment_ucb(T, num_arms, c_values, num_runs)

# 绘制结果
plt.figure(figsize=(10, 6))
for c, rewards in avg_rewards_ucb.items():
    plt.plot(np.arange(1, T + 1), rewards, label=f'c = {c}')
    
plt.title('UCB: Average Reward vs Steps')
plt.xlabel('Steps (T)')
plt.ylabel('Average Reward')
plt.legend()
plt.savefig('./results/reward_ucb.png')
plt.show()


max_avg = max(values[-1] for values in avg_rewards_ucb.values())
avg_regret = {N: max_avg - avg_rewards_ucb[N][-1] for N in c_values}

plt.figure()
plt.plot(list(avg_regret.keys()), list(avg_regret.values()), marker='o')
plt.xlabel('Parameters (c_values)')
plt.ylabel('Average Regret')
plt.title('Average Regret vs Parameters')
plt.savefig('./results/average_regret_ucb.png')
plt.show()