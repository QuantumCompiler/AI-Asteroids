import gymnasium as gym
from collections import deque
import torch
import numpy as np
from agent import *

# Device setup
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using CUDA on GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Metal Performance Shaders) backend.")
else:
    device = torch.device("cpu")
    print("No GPU available. Using CPU.")

# Initialize environment and agent
env = gym.make("ALE/Asteroids-v5")
input_shape = env.observation_space.shape
agent = DQNAgent(input_shape=input_shape, action_size=env.action_space.n, seed=0)

# Training function
def dqn(n_episodes=5, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    scores = []
    scores_window = deque(maxlen=100)
    eps = eps_start
    for i_episode in range(1, n_episodes + 1):
        state, info = env.reset(seed=42)
        score = 0
        print(f"\nEpisode {i_episode}")
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                print(f"\nEpisode finished after {t + 1} timesteps")
                break
        scores_window.append(score)
        scores.append(score)
        eps = max(eps_end, eps_decay * eps)
        print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}', end="")
        if i_episode % 100 == 0:
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
    return scores

# Training
scores = dqn()

# Evaluation function
def play_game(env, agent, n_episodes=5):
    for i_episode in range(1, n_episodes + 1):
        state, info = env.reset(seed=42)
        total_reward = 0
        print(f"\nPlaying episode {i_episode}")
        while True:
            action = agent.act(state, eps=0.0)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            state = next_state
            total_reward += reward
            if done:
                print(f"\nEpisode {i_episode} finished with total reward: {total_reward}")
                break

# Load trained model
agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth', weights_only=True))
agent.qnetwork_local.eval()

# Evaluation
play_game(env, agent)
env.close()