import numpy as np
import matplotlib.pyplot as plt

np.random.seed(307)

def smooth(data, window_size=10):
    cumsum_vec = np.cumsum(np.insert(data, 0, 0))
    smoothed = (cumsum_vec[window_size:] - cumsum_vec[:-window_size]) / window_size
    return smoothed

class CliffWorld:
    def __init__(self):
        self.height = 4
        self.width = 12
        self.start = (3, 0)
        self.goal = (3, 11)
        self.cliff = [(3, i) for i in range(1, 11)]
        self.actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    def step(self, state, action):
        r, c = state
        dr, dc = action
        nr, nc = r + dr, c + dc
        
        # 边界检查
        if nr < 0 or nr >= self.height or nc < 0 or nc >= self.width:
            nr, nc = r, c
        
        reward = -1
        if (nr, nc) in self.cliff:
            reward = -100
            nr, nc = self.start
        return (nr, nc), reward

def epsilon_greedy(Q, state, epsilon, env):
    if np.random.rand() < epsilon:
        return np.random.choice(len(env.actions))
    else:
        return np.argmax(Q[state[0], state[1]])

def q_learning(env, episodes, alpha, gamma, epsilon):

    Q = np.zeros((env.height, env.width, len(env.actions)))
    rewards = []

    for _ in range(episodes):
        state = env.start
        total_reward = 0
        
        while state != env.goal:
            a_idx = epsilon_greedy(Q, state, epsilon, env)
            next_state, reward = env.step(state, env.actions[a_idx])
            best_next_action = np.argmax(Q[next_state[0], next_state[1]])
            
            Q[state[0], state[1], a_idx] += alpha * (
                reward + gamma * Q[next_state[0], next_state[1], best_next_action]
                - Q[state[0], state[1], a_idx]
            )
            state = next_state
            total_reward += reward

        rewards.append(total_reward)

    return rewards, Q

def sarsa(env, episodes, alpha, gamma, epsilon):

    Q = np.zeros((env.height, env.width, len(env.actions)))
    rewards = []

    for _ in range(episodes):
        state = env.start
        a_idx = epsilon_greedy(Q, state, epsilon, env)
        total_reward = 0

        while state != env.goal:
            next_state, reward = env.step(state, env.actions[a_idx])
            next_a_idx = epsilon_greedy(Q, next_state, epsilon, env)
            Q[state[0], state[1], a_idx] += alpha * (
                reward + gamma * Q[next_state[0], next_state[1], next_a_idx]
                - Q[state[0], state[1], a_idx]
            )
            state = next_state
            a_idx = next_a_idx
            total_reward += reward

        rewards.append(total_reward)

    return rewards, Q


# Print the policy
def print_greedy_policy(env, Q):

    arrow_map = {0: '→', 1: '←', 2: '↓', 3: '↑'}
    grid = []

    for r in range(env.height):
        row_symbols = []
        for c in range(env.width):
            if (r, c) == env.goal:
                row_symbols.append('★')
            elif (r, c) in env.cliff:
                row_symbols.append('♦')
            else:
                best_a = np.argmax(Q[r, c])  # find the best action at (r,c)
                row_symbols.append(arrow_map[best_a])
        grid.append(row_symbols)

    print("\nGreedy Policy from Q-table:")
    for row in grid:
        print(" ".join(row))

def run_experiments(num_runs=30, episodes=500, alpha=0.5, gamma=1.0, epsilon=0.1):
    env = CliffWorld()
    q_all_runs = np.zeros((num_runs, episodes))
    sarsa_all_runs = np.zeros((num_runs, episodes))

    for run in range(num_runs):
        q_rewards, _ = q_learning(env, episodes, alpha, gamma, epsilon)
        sarsa_rewards, _ = sarsa(env, episodes, alpha, gamma, epsilon)
        q_all_runs[run, :] = q_rewards
        sarsa_all_runs[run, :] = sarsa_rewards

    # Mean over runs
    q_mean = np.mean(q_all_runs, axis=0)
    sarsa_mean = np.mean(sarsa_all_runs, axis=0)
    return q_mean, sarsa_mean

if __name__ == "__main__":
    episodes = 500
    num_runs = 30
    alpha = 0.5
    gamma = 1.0
    epsilon = 0.1
    
    q_mean, sarsa_mean = run_experiments(num_runs=num_runs, episodes=episodes,
                                         alpha=alpha, gamma=gamma, epsilon=epsilon)

    plt.plot(smooth(q_mean, window_size=30), label="Q-learning")
    plt.plot(smooth(sarsa_mean, window_size=30), label="SARSA")
    plt.xlabel('Episodes')
    plt.ylabel('Sum of rewards during episode')
    plt.title('Cliff Walking: Q-learning vs. SARSA (Average of Multiple Runs)')
    plt.legend()
    plt.grid(True)
    plt.show()

    env = CliffWorld()
    single_run_episodes = 500
    single_run_alpha = 0.5
    single_run_gamma = 1.0
    single_run_epsilon = 0.1

    # Run Q-learning once, then print the final policy
    q_rewards_single, Q_qlearning = q_learning(env, single_run_episodes,
                                               single_run_alpha,
                                               single_run_gamma,
                                               single_run_epsilon)
    print("\n--- Single Run Q-learning ---")
    print("Final Episode Reward:", q_rewards_single[-1])
    print_greedy_policy(env, Q_qlearning)

    # Run SARSA once, then print the final policy
    sarsa_rewards_single, Q_sarsa = sarsa(env, single_run_episodes,
                                          single_run_alpha,
                                          single_run_gamma,
                                          single_run_epsilon)
    print("\n--- Single Run SARSA ---")
    print("Final Episode Reward:", sarsa_rewards_single[-1])
    print_greedy_policy(env, Q_sarsa)