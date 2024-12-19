import gym
import numpy as np
import time
from collections import defaultdict
import pandas as pd
import random

env = gym.make('FrozenLake-v1')
ep = 50000


def policy(state):
    """
    Optimized policy for FrozenLake-v1.

    Args:
        state: Current state (agent's position on the grid).

    Returns:
        action: Chosen action based on the policy.
    """
    action_probabilities = [0.1, 0.4, 0.4, 0.1]  # [Left, Down, Right, Up]
    action = np.random.choice([0, 1, 2, 3], p=action_probabilities)
    return action


def generate_episode(env, policy):
    """
    Generates an episode by following the given policy in the environment.

    Args:
        env: OpenAI Gym environment.
        policy: Function that maps state to action.

    Returns:
        episode: List of tuples (state, action, reward).
    """
    episode = []
    state, info = env.reset()  # Reset environment to get initial state

    while True:
        # Choose action according to policy
        action = policy(state)
        # Take action in the environment
        next_state, reward, terminated, truncated, info = env.step(action)
        # Append state, action, reward to the episode list
        episode.append((state, action, reward))
        if terminated or truncated:
            break
        state = next_state  # Move to next state

    return episode


def first_visit_mc_prediction(policy, env, num_episodes, gamma=1.0):
    """
    Performs first-visit Monte Carlo prediction to estimate the value function V(s).

    Args:
        policy: Function that maps state to action.
        env: OpenAI Gym environment.
        num_episodes: Number of episodes to sample.
        gamma: Discount factor.

    Returns:
        V: A dictionary mapping states to their estimated values.
    """
    # Initialize the returns and value function dictionaries
    returns_sum = defaultdict(float)  # Sum of returns for each state
    returns_count = defaultdict(int)  # Count of returns for each state
    V = defaultdict(float)            # Value function

    for episode_num in range(num_episodes):
        # Generate an episode
        episode = generate_episode(env, policy)
        # Set to keep track of states we've already seen in this episode
        visited_states = set()
        for i, (state, action, reward) in enumerate(episode):
            # Check if it's the first visit to the state in this episode
            if state not in visited_states:
                visited_states.add(state)
                # Calculate the return G from this state onwards
                G = 0.0
                gamma_power = 1.0
                for j in range(i, len(episode)):
                    _, _, r = episode[j]
                    G += gamma_power * r
                    gamma_power *= gamma
                # Update the returns_sum and returns_count
                returns_sum[state] += G
                returns_count[state] += 1
                # Update the value function estimate
                V[state] = returns_sum[state] / returns_count[state]

    return V

# Run the Monte Carlo prediction
V = first_visit_mc_prediction(policy, env, ep)

# Optional: Print the estimated value function
print("Estimated Value Function:")
for state in range(env.observation_space.n):
    print(f"State {state}: {V.get(state, 0):.4f}")




