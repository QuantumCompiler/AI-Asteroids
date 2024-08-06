import gymnasium as gym
from collections import deque
from agent import *

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
def dqn(n_episodes=10, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    scores = []
    scores_window = deque(maxlen=100)
    eps = eps_start
    for i_episode in range(1, n_episodes+1):
        state, info = env.reset(seed=42)
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)
        scores.append(score)
        eps = max(eps_end, eps_decay*eps)
        print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}', end="")
    torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
    return scores

# Train the agent and save the model
scores = dqn()

# Load trained model
agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth', weights_only=True))

env = gym.make("ALE/Asteroids-v5", render_mode="human")

def play_game(env, agent, n_episodes=1):
    for i_episode in range(1, n_episodes+1):
        state, info = env.reset(seed=42)
        total_reward = 0
        while True:
            action = agent.act(state)
            state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        print(f"Episode {i_episode}\tTotal Reward: {total_reward}")

play_game(env, agent)
env.close()