import gym
import numpy as np
import time
from collections import defaultdict
import pandas as pd

# Initialize variables
N = defaultdict(int)
total_return = defaultdict(float)

env = gym.make('Blackjack-v1')
NS = 10 * 10 * 2  # 10 for sum between (12, 21), 10 for dealer (2-11), 2 for usable ace
Na = 2  # hit is 1, stand is 0
ep = 50000

def argmax_rand(arr):
    """
    Returns the index of the maximum value in the array, breaking ties randomly.
    """
    return np.random.choice(np.flatnonzero(arr == np.amax(arr)))

def generate_episode(env, Pi, epsilon):
    """
    Generates an episode by following the epsilon-greedy policy.

    Args:
        env: OpenAI Gym environment.
        Pi: Action-value function (policy).
        epsilon: Exploration rate.

    Returns:
        episode: List of tuples (state, action, reward).
    """
    episode = []
    state, _ = env.reset()  # Get initial state, discard info
    done = False

    while not done:
        action = epsilon_greedy_policy(state, Pi, epsilon)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        episode.append((state, action, reward))
        state = next_state

    return episode

def epsilon_greedy_policy(state, Pi, epsilon):
    """
    Epsilon-greedy policy for action selection.
    """
    if state not in Pi:
        Pi[state] = np.ones(Na) / Na  # Initialize equal probabilities for all actions

    if np.random.rand() < epsilon:
        return np.random.choice(Na)  # Exploration: random action
    else:
        return argmax_rand(Pi[state])  # Exploitation: action with the highest value

def first_visit_mc_prediction(env, num_episodes, gamma=1.0, epsilon=0.1):
    """
    Performs first-visit Monte Carlo prediction to estimate the action-value function Q(s, a).
    """
    Q = defaultdict(float)  # Action-value function
    returns_sum = defaultdict(float)  # Sum of returns for each state-action pair
    returns_count = defaultdict(int)  # Count of returns for each state-action pair
    Pi = defaultdict(lambda: np.ones(Na) / Na)  # Initialize policy with equal probabilities

    for episode_num in range(num_episodes):
        episode = generate_episode(env, Pi, epsilon)
        visited_state_actions = set()
        for i, (state, action, reward) in enumerate(episode):
            if (state, action) not in visited_state_actions:
                visited_state_actions.add((state, action))
                G = 0.0
                gamma_power = 1.0
                for j in range(i, len(episode)):
                    _, _, r = episode[j]
                    G += gamma_power * r
                    gamma_power *= gamma
                returns_sum[(state, action)] += G
                returns_count[(state, action)] += 1
                Q[(state, action)] = returns_sum[(state, action)] / returns_count[(state, action)]

                # Update policy based on the current action-value function
                max_action = argmax_rand([Q[(state, a)] for a in range(Na)])
                for action in range(Na):
                    if action == max_action:
                        Pi[state][action] = 1 - epsilon + (epsilon / Na)
                    else:
                        Pi[state][action] = epsilon / Na

    return Q, Pi

# Run the Monte Carlo prediction
Q, Pi = first_visit_mc_prediction(env, ep, gamma=0.9, epsilon=0.07)
print(Pi)
