import numpy as np
import matplotlib.pyplot as plt

np.random.seed(7202)

def gradient_bandit_avg_reward(true_means, T, alpha):
    """
    Run Gradient Bandit algorithm and return cumulative rewards at each step.
    :param true_means: True mean rewards for each arm.
    :param T: Total time steps.
    :param alpha: Learning rate for the gradient update.
    :return: Array of cumulative rewards at each step.
    """
    num_arms = len(true_means)
    H = np.zeros(num_arms)  # 初始化偏好值
    avg_reward = 0  # 平均奖励
    rewards = np.zeros(T)  # 每一步的即时奖励
    
    for t in range(T):
        # 计算每个拉杆的选择概率 (Softmax)
        probabilities = np.exp(H) / np.sum(np.exp(H))
        
        # 根据概率选择拉杆
        arm = np.random.choice(num_arms, p=probabilities)
        
        # 拉动选择的拉杆并获取奖励
        reward = np.random.normal(true_means[arm], 1)
        rewards[t] = reward  # 保存即时奖励
        
        # 更新平均奖励
        avg_reward += (reward - avg_reward) / (t + 1)
        
        # 更新偏好值 (梯度更新)
        for i in range(num_arms):
            H[i] += alpha * (reward - avg_reward) * ((1 if i == arm else 0) - probabilities[i])
    
    return np.cumsum(rewards)  # 返回累计奖励


def run_experiment_gradient_bandit(T, num_arms, alpha_values, num_runs=500):
    """
    Run Gradient Bandit experiment for different alpha values.
    :param T: Total time steps.
    :param num_arms: Number of arms.
    :param alpha_values: List of alpha values to test.
    :param num_runs: Number of independent runs to average.
    :return: Dictionary of average rewards for each alpha.
    """
    avg_rewards = {alpha: np.zeros(T) for alpha in alpha_values}
    
    for alpha in alpha_values:
        cumulative_rewards = np.zeros((num_runs, T))
        for run in range(num_runs):
            # 生成拉杆的真实均值
            true_means = np.random.normal(0, 1, num_arms)
            # 运行Gradient Bandit算法
            cumulative_rewards[run] = gradient_bandit_avg_reward(true_means, T, alpha)
        
        # 计算每一步的平均奖励
        avg_rewards[alpha] = np.mean(cumulative_rewards, axis=0) / (np.arange(1, T + 1))
    
    return avg_rewards


# 参数设置
T = 1000  # 时间步数
num_arms = 10  # 拉杆数量
alpha_values = [0.1, 0.4, 0.8, 1.0]  # 不同的学习率
num_runs = 5  # 重复实验次数

# 运行实验
avg_rewards_gradient_bandit = run_experiment_gradient_bandit(T, num_arms, alpha_values, num_runs)

# 绘制结果
plt.figure(figsize=(10, 6))
for alpha, rewards in avg_rewards_gradient_bandit.items():
    plt.plot(np.arange(1, T + 1), rewards, label=f'alpha = {alpha}')
    
plt.title('Gradient Bandit: Average Reward vs Steps')
plt.xlabel('Steps (T)')
plt.ylabel('Average Reward')
plt.legend()
plt.savefig('./results/reward_gradient_bandit.png')
plt.show()


max_avg = max(values[-1] for values in avg_rewards_gradient_bandit.values())
avg_regret = {N: max_avg - avg_rewards_gradient_bandit[N][-1] for N in alpha_values}

plt.figure()
plt.plot(list(avg_regret.keys()), list(avg_regret.values()), marker='o')
plt.xlabel('Parameters (alpha)')
plt.ylabel('Average Regret')
plt.title('Average Regret vs Parameters')
plt.savefig('./results/average_regret_gradient_bandit.png')
plt.show()