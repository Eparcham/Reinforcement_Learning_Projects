import gym
import numpy as np
import time

def value_iteration_step(env, V_last, gamma, policy):
    Ns = env.observation_space.n  # Number of states
    Na = env.action_space.n  # Number of actions
    V = np.zeros(Ns)

    for s in range(Ns):
        for a in range(Na):
            for prob, next_state, reward, _ in env.P[s][a]:
                V[s] += policy[s][a] * prob * (reward + gamma * V_last[next_state])

    return V


def derive_policy_step(env, V, gamma):
    Ns = env.observation_space.n  # Number of states
    Na = env.action_space.n  # Number of actions
    policy = np.zeros([Ns, Na])

    for s in range(Ns):
        q_values = np.zeros(Na)
        for a in range(Na):
            for prob, next_state, reward, _ in env.P[s][a]:
                q_values[a] += prob * (reward + gamma * V[next_state])
        best_action = np.argmax(q_values)
        policy[s] = np.eye(Na)[best_action]\

    return policy


def value_iteration(env, gamma=0.99, theta=1e-20, max_iterations=1000):
    Ns = env.observation_space.n  # Number of states
    Na = env.action_space.n  # Number of actions
    V = np.zeros(Ns)  # Initialize value function
    policy = np.ones([Ns, Na]) / Na  # Initialize policy with equal probabilities

    for iteration in range(max_iterations):
        V_last = np.copy(V)
        V = value_iteration_step(env, V_last, gamma, policy)
        policy = derive_policy_step(env, V, gamma)
        delta = np.max(np.abs(V - V_last))

        # Check for convergence
        if delta < theta:
            print(f"Converged in {iteration + 1} iterations.")
            break

    return V, policy


# Usage
env = gym.make('Taxi-v3')
V, policy = value_iteration(env)

# Display results
print("Optimal Policy:")
print(policy)
env.close()


## infrence
env = gym.make('Taxi-v3', render_mode="human")
env.reset()


state, info = env.reset()
episode_num = 100
for i in range(episode_num):
    print(f"Episode {i+1}")
    done = False
    while not done:
        action = np.argmax(policy[state])
        next_state, reward, done, truncated, info = env.step(action)
        print(f"Current state: {state} -> Action: {action} -> Next state: {next_state}")
        env.render()
        time.sleep(0.1)
        state = next_state
        if done:
            print(f"Episode ended with reward: {reward}\n")
            state, info = env.reset()
            break

env.close()