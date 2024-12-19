import gym
import numpy as np
from collections import defaultdict
import pandas as pd

env = gym.make('FrozenLake-v1')

eps = 0.2
alpha = 0.2
gamma = 1.0
nb_action = 4

def argmax_rand(arr):
    return np.random.choice(np.flatnonzero(arr == np.amax(arr)))

def sarsa(eps, alpha, ep=1000):
    Q = defaultdict(float)

    def policy(state, pi):
        return np.random.choice([0, 1, 2, 3], p=[pi[(state, a)] for a in range(nb_action)])

    pi = defaultdict(lambda: 1 / nb_action)

    for _ in range(ep):
        s = env.reset()[0]
        a = policy(s, pi)

        while True:
            s_, r, done, _, _ = env.step(a)
            a_ = policy(s_, pi)

            Q[(s, a)] = Q[(s, a)] + alpha * (r + gamma * Q[(s_, a_)] - Q[(s, a)])

            A_star = argmax_rand([Q[(s, a)] for a in range(nb_action)])

            for a in range(nb_action):
                if a == A_star:
                    pi[(s, a)] = 1 - eps + eps / nb_action
                else:
                    pi[(s, a)] = eps / nb_action

            s = s_
            a = a_
            if done:
                break

    return Q, pi

Q, pi = sarsa(eps, alpha, ep=1000)
pi_df = pd.DataFrame(list(pi.items()), columns=['state_action', 'probability'])
print(pi_df)
