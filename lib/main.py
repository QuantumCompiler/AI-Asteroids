from agent import *

# Evaluation function
def play_game(env, agent, n_episodes=1, agent_name="New_Agent.pth"):
    agent.qnetwork_local.eval()
    total_rewards = []
    if not os.path.exists('../Evaluation Results'):
        os.makedirs('../Evaluation Results')
    results_file = f'../Evaluation Results/{agent_name}_Eval_Results.csv'
    with open(results_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Episode', 'Total Reward'])
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
            total_rewards.append(total_reward)
            running_average = np.mean(total_rewards)
            writer.writerow([i_episode, total_reward])

if __name__ == "__main__":
    agent_name = input("Enter the name for agent you'd like to evaluate: ")
    if agent_name == "":
        agent_name = "New_Agent"
    env = gym.make("ALE/Asteroids-v5", render_mode="human")
    input_shape = env.observation_space.shape
    agent = DQNAgent(input_shape=input_shape, action_size=env.action_space.n, seed=0)
    checkpoint = torch.load(f'../Agents/{agent_name}.pth', weights_only=True)
    agent.qnetwork_local.load_state_dict(checkpoint)
    play_game(env, agent, agent_name=agent_name)
    env.close()