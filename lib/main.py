import gymnasium as gym
import torch
from train import DQNAgent

# Evaluation function
def play_game(env, agent, n_episodes=1):
    agent.qnetwork_local.eval()
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
                print(f"Episode {i_episode} finished with total reward: {total_reward}")
                break

env = gym.make("ALE/Asteroids-v5", render_mode="human")

# Initialize agent
input_shape = env.observation_space.shape
agent = DQNAgent(input_shape=input_shape, action_size=env.action_space.n, seed=0)

# Load the trained model
checkpoint = torch.load('Agent.pth', weights_only=True)
agent.qnetwork_local.load_state_dict(checkpoint)

# Evaluation
play_game(env, agent)
env.close()