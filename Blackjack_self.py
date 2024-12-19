import gym
import numpy as np
import time
from collections import defaultdict
import pandas as pd

N = defaultdict(int)
total_return = defaultdict(float)



env = gym.make('Blackjack-v1')
NS = 10 * 10 * 2 ## 10 FOR CUM NUMBER BETWEN (12, 21) 10 FOR DELEVER (2-11) 2 FOR USE 11 UMBER OR NO
Na = 2 # hit is 1 stand is 0
iteration = 10000
episode = 100

def policy(sum_state):
    return 0 if sum_state>19 else 1

def generate_policy(policy):
    episode_data = []
    state = env.reset()[0]
    # print(state)
    while True:
        action = policy(state[0])
        next_state, reward, done, truncated, info = env.step(action)
        episode_data.append([state, action, reward])
        if done:
            break
        state = next_state

    return episode_data

def every_visit():
    for i in range(iteration):
        gen = generate_policy(policy)
        states, actions, rewards = zip(*gen)
        for t, state in enumerate(states):
            N[state] += 1
            if state not in states[0:t]:
                R = sum(rewards[t:])
                total_return[state] += R

    return total_return, N

def first_visit():
    for i in range(iteration):
        gen = generate_policy(policy)
        states, actions, rewards = zip(*gen)
        for t, state in enumerate(states):
            if state not in states[0:t]:
                N[state] += 1
                R = sum(rewards[t:])
                total_return[state] += R

    return total_return, N

total_return, N = every_visit()
total_return, N = first_visit()
total_return = pd.DataFrame(total_return.items(), columns=['state', 'total_return'])
N = pd.DataFrame(N.items(), columns=['state', 'N'])
df = pd.merge(total_return, N, on='state')
df['value'] = df['total_return']/df['N']

print(df.head())
env.close()