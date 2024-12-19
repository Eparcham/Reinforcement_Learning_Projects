import gym
import numpy as np
from collections import defaultdict

env = gym.make('FrozenLake-v1', is_slippery=True)

eps = 0.3
alpha = 0.1
gamma = 0.9
nb_action = env.action_space.n
max_steps = 100

def policy(state, Q, eps):
    if np.random.random() < eps:
        return env.action_space.sample()
    else:
        return np.argmax([Q.get((state, at), 0.0) for at in range(nb_action)])

def sarsa(eps, alpha, ep=1000):
    Q = defaultdict(lambda: 0.1)
    rewards = []

    for i in range(ep):
        total_reward = 0
        eps = max(0.05, eps * (0.995 ** (i // 100)))

        s = env.reset()[0]
        a = policy(s, Q, eps)
        step_count = 0

        while step_count < max_steps:
            step_count += 1
            s_, r, done, _, _ = env.step(a)
            a_ = policy(s_, Q, eps)

            Q[(s, a)] += alpha * (r + gamma * Q.get((s_, a_), 0.0) - Q[(s, a)])

            s, a = s_, a_
            total_reward += r

            if done:
                break

        rewards.append(total_reward)

    return Q, rewards

def extract_policy(Q, n_states, n_actions):
    policy = np.zeros(n_states, dtype=int)
    for state in range(n_states):
        actions = [Q.get((state, a), 0.0) for a in range(n_actions)]
        policy[state] = np.argmax(actions)
    return policy

Q, rewards = sarsa(eps, alpha, ep=20000)

optimal_policy = extract_policy(Q, env.observation_space.n, nb_action)
print("Optimal Policy:")
for i in range(env.observation_space.n):
    print(f"State {i}: Action {optimal_policy[i]}")

for i in range(0, len(rewards), 100):
    avg_reward = np.mean(rewards[i:i+100])
    print(f"Episode {i}-{i+99}: Average Reward: {avg_reward}")
