import gymnasium as gym
from dqn_agent import DQN_Agent, process_frame
from gymnasium.wrappers import RecordVideo
import os

#Let the agent play the game with the weights downloaded from the Kaggle notebook and render it to a video
def main():
    env = gym.make("ALE/Carnival-v5", render_mode='rgb_array')
    env = RecordVideo(env, './video', episode_trigger=lambda episode_id: True)
    env.metadata['render_fps'] = 30
    state_size = (96, 80, 1)
    action_size = env.action_space.n

    agent = DQN_Agent(state_size, action_size)

    
    # Load the pre-trained model weights
    agent.load('./agent/Gewichte.weights.h5')

    state, _ = env.reset()
    state = process_frame(state)
    done = False
    total_reward = 0
    
    while not done:
        action = agent.act(state)  # Get action from the agent
        next_state, reward, done, truncated, info = env.step(action)
        next_state = process_frame(next_state)
        state = next_state
        total_reward += reward
        env.render()

    env.close()
    print("Total reward:", total_reward)

    # Ensure that video files are properly closed
    video_directory = './video'
    if os.path.exists(video_directory):
        print(f"Video files saved to: {video_directory}")

if __name__ == "__main__":
    main()
