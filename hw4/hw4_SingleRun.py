import numpy as np
import matplotlib.pyplot as plt

def smooth(data, window_size=30):
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
        next_state = (state[0] + action[0], state[1] + action[1])
        
        if (next_state[0] < 0 or next_state[0] >= self.height or
            next_state[1] < 0 or next_state[1] >= self.width):
            next_state = state

        reward = -1 # as the same in the book
        if next_state in self.cliff:
            reward = -100  # as the same in the book
            next_state = self.start

        return next_state, reward

def epsilon_greedy(Q, state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.choice(len(env.actions))
    else:
        return np.argmax(Q[state[0], state[1]])

# Q-learning
def q_learning(env, episodes, alpha, gamma, epsilon):
    Q = np.zeros((env.height, env.width, len(env.actions)))
    rewards = []

    for _ in range(episodes):
        state = env.start
        total_reward = 0
        
        while state != env.goal:
            action_idx = epsilon_greedy(Q, state, epsilon)
            action = env.actions[action_idx]
            next_state, reward = env.step(state, action)
            best_next_action = np.argmax(Q[next_state[0], next_state[1]])
            
            # Q-learning update
            Q[state[0], state[1], action_idx] += alpha * (
                reward + gamma * Q[next_state[0], next_state[1], best_next_action]
                - Q[state[0], state[1], action_idx]
            )

            state = next_state
            total_reward += reward

        rewards.append(total_reward)

    return rewards, Q

# SARSA
def sarsa(env, episodes, alpha, gamma, epsilon):
    Q = np.zeros((env.height, env.width, len(env.actions)))
    rewards = []

    for _ in range(episodes):
        state = env.start
        action_idx = epsilon_greedy(Q, state, epsilon)
        total_reward = 0

        while state != env.goal:
            action = env.actions[action_idx]
            next_state, reward = env.step(state, action)
            next_action_idx = epsilon_greedy(Q, next_state, epsilon)

            # SARSA update
            Q[state[0], state[1], action_idx] += alpha * (
                reward + gamma * Q[next_state[0], next_state[1], next_action_idx]
                - Q[state[0], state[1], action_idx]
            )

            state = next_state
            action_idx = next_action_idx
            total_reward += reward

        rewards.append(total_reward)

    return rewards, Q

env = CliffWorld()

# Hyperparameters
episodes = 500     
alpha = 0.5        
gamma = 1.0         
epsilon = 0.1     

q_rewards, q_Q = q_learning(env, episodes, alpha, gamma, epsilon)
sarsa_rewards, sarsa_Q = sarsa(env, episodes, alpha, gamma, epsilon)

plt.plot(smooth(q_rewards), label='Q-learning')
plt.plot(smooth(sarsa_rewards), label='SARSA')
plt.xlabel('Episodes')
plt.ylabel('Sum of rewards during episode')
plt.title('Comparison of Q-learning and SARSA on Cliffworld')
plt.legend()
plt.grid(True)
plt.show()