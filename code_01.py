import gym
import time

env = gym.make('FrozenLake-v1', render_mode="human")
env.reset()

episode_num = 100
for i in range(episode_num):
    print(f"Episode {i}")
    state = env.reset()
    while True:
        action = env.action_space.sample()
        next_state, reward, done, _, info = env.step(action)
        print(f"Current state: {state} -> Action: {action} -> Next state: {next_state}")
        env.render()
        time.sleep(0.2)
        state = next_state
        if done:
            print(f"Episode ended with reward: {reward}\n")
            break

env.close()
