import gym
import numpy as np
import time
from collections import defaultdict
import pandas as pd


# Usage
env = gym.make('FrozenLake-v1')


def policy(state):
    if state in [1, 4]:
        return 0
    elif state in [2, 9, 14]:
        return 1
    elif state in [0, 6, 13]:
        return 2
    else:
        return 3

def td_learning(env, alpha=0.2, gamma=1.0, ep=1000):
    V = defaultdict(int)
    for _ in range(ep):
        state = env.reset()[0]
        while True:
            a = policy(state)
            s_, r, done, _, _ = env.step(a)
            V[state] = V[state] + alpha * (r + gamma * V[s_] - V[state])
            state = s_
            if done:
                break
    return V


V = td_learning(env)
V = pd.DataFrame(V.items(), columns=['state', 'value'])

print(V)