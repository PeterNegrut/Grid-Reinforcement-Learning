from collections import defaultdict
from Agent import *
from world import Environment
from tqdm import tqdm  # Progress bar
from matplotlib import pyplot as plt
import gymnasium as gym
import numpy as np


# Training hyperparameters
learning_rate = 0.01        # How fast to learn (higher = faster but less stable)
n_episodes = 100_000        # Number of hands to practice
start_epsilon = 1.0         # Start with 100% random actions
epsilon_decay = start_epsilon / (n_episodes / 2)  # Reduce exploration over time
final_epsilon = 0.1         # Always keep some exploration
episode_rewards = []
episode_lengths = []

env = Environment()

agent = Agent(
    env=env,
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
)


for episode in tqdm(range(n_episodes)):
    # Start a new hand
    obs = env.reset()
    done = False
    episode_reward = 0
    episode_length = 0

    # Play one complete hand
    while not done:
        # Agent chooses action (initially random, gradually more intelligent)
        action = agent.get_action(obs)

        # Take action and observe result
        next_obs, reward, terminated, truncated, info = env.step(action)

        # Learn from this experience
        agent.update(obs, action, reward, terminated, next_obs)

        # Move to next state
        done = terminated or truncated
        obs = next_obs
        episode_reward += reward
        episode_length += 1

    episode_rewards.append(episode_reward)
    episode_lengths.append(episode_length)

    # Reduce exploration rate (agent becomes less random over time)
    agent.decay_epsilon()

def get_moving_avgs(arr, window, convolution_mode):
    """Compute moving average to smooth noisy data."""
    return np.convolve(
        np.array(arr).flatten(),
        np.ones(window),
        mode=convolution_mode
    ) / window

# Smooth over a 500-episode window
rolling_length = 500
fig, axs = plt.subplots(ncols=3, figsize=(12, 5))

# Episode rewards (win/loss performance)
axs[0].set_title("Episode rewards")
reward_moving_average = get_moving_avgs(
    episode_rewards,
    rolling_length,
    "valid"
)
axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
axs[0].set_ylabel("Average Reward")
axs[0].set_xlabel("Episode")

# Episode lengths (how many actions per hand)
axs[1].set_title("Episode lengths")
length_moving_average = get_moving_avgs(
    episode_lengths,
    rolling_length,
    "valid"
)
axs[1].plot(range(len(length_moving_average)), length_moving_average)
axs[1].set_ylabel("Average Episode Length")
axs[1].set_xlabel("Episode")

# Training error (how much we're still learning)
axs[2].set_title("Training Error")
training_error_moving_average = get_moving_avgs(
    agent.training_error,
    rolling_length,
    "same"
)
axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
axs[2].set_ylabel("Temporal Difference Error")
axs[2].set_xlabel("Step")

plt.tight_layout()
plt.show()


# Add this line to make sure plots finish before rendering starts
input("Press Enter to see the agent in action...")

# After training is complete, test the agent
print("Testing trained agent:")
# ... rest of your testing code

# After training is complete, test the agent
print("Testing trained agent:")
obs = env.reset()
env.render()

for step in range(20):  # Max 20 steps
    action = agent.get_action(obs)
    next_obs, reward, terminated, truncated, info = env.step(action)
    
    print(f"Step {step+1}: Action {action}")
    env.render()
    
    if terminated:
        print(f"Goal reached in {step+1} steps!")
        break
        
    obs = next_obs