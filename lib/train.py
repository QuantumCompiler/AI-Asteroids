from agent import *

# Initialize environment and agent
env = gym.make("ALE/Asteroids-v5")
input_shape = env.observation_space.shape
agent = DQNAgent(input_shape=input_shape, action_size=env.action_space.n, seed=0)

# Training function
def dqn(n_episodes=50, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
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
                break
        scores_window.append(score)
        scores.append(score)
        eps = max(eps_end, eps_decay * eps)
        print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}', end="")
        torch.save(agent.qnetwork_local.state_dict(), 'Agent.pth')
    return scores

# Training
scores = dqn()