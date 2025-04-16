import gym
import numpy as np
import random
import matplotlib.pyplot as plt

# Create the FrozenLake environment
env = gym.make("FrozenLake-v1", is_slippery=True)

# Initialize Q-table of size (number_of_states x number_of_actions)
q_table = np.zeros([env.observation_space.n, env.action_space.n])

# Hyperparameters
num_episodes = 2000       # total episodes for training
max_steps = 100           # max steps per episode
learning_rate = 0.8       # learning rate (alpha)
discount_factor = 0.95    # discount factor (gamma)
epsilon = 1.0             # exploration rate
max_epsilon = 1.0         # exploration probability at start
min_epsilon = 0.01        # minimum exploration probability 
decay_rate = 0.005        # exponential decay rate for exploration prob

# List to contain total rewards per episode
rewards_all_episodes = []

# Q-Learning Algorithm
for episode in range(num_episodes):
    state = env.reset()  # Reset the environment for a new episode
    total_rewards = 0

    for step in range(max_steps):
        # Choose an action (exploration vs exploitation)
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Exploration: random action
        else:
            action = np.argmax(q_table[state, :])  # Exploitation: action with max value from Q-table

        # Take action and observe the outcome state and reward
        new_state, reward, done, _ = env.step(action)
        
        # Update Q(s,a) using the Bellman equation
        q_table[state, action] = q_table[state, action] + learning_rate * (reward + discount_factor * np.max(q_table[new_state, :]) - q_table[state, action])
        
        state = new_state
        total_rewards += reward

        if done:
            break

    # Reduce epsilon (because we need less exploration over time)
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
    rewards_all_episodes.append(total_rewards)

print("Training completed.\n")
print("Final Q-table values:")
print(q_table)

# Plot the rewards per episode
plt.plot(range(num_episodes), rewards_all_episodes)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Rewards per Episode")
plt.show()
