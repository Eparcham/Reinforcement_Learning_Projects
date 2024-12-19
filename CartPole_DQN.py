import gym
import time
from collections import deque
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

env = gym.make('CartPole-v1', render_mode="human")
env.reset()

M = 500
T = 210
batch_size = 64


class DQN(nn.Module):
    def __init__(self, state_size, action_size, alpha, gamma, epsilon, epsilon_min, epsilon_decay):
        super(DQN, self).__init__()
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.alpha = alpha
        self.nS = state_size
        self.nA = action_size
        self.memory = deque([], maxlen=6400)

        # Define the model
        self.fc1 = nn.Linear(self.nS, 64)  # First hidden layer
        self.fc2 = nn.Linear(64, 64)  # Second hidden layer
        self.fc3 = nn.Linear(64, self.nA)  # Output layer

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.loss = []

    def forward(self, x):
        x = F.relu(self.fc1(x))  # Apply ReLU activation to the first layer
        x = F.relu(self.fc2(x))  # Apply ReLU activation to the second layer
        x = self.fc3(x)  # Linear output for the Q-values
        return x

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.nA)  # Random action (exploration)
        state = torch.FloatTensor(state).unsqueeze(0)  # Add batch dimension
        q_values = self.forward(state)
        return torch.argmax(q_values).item()  # Best action (exploitation)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state).unsqueeze(0)
            next_state = torch.FloatTensor(next_state).unsqueeze(0)
            target = reward
            if not done:
                target += self.gamma * torch.max(self.forward(next_state)).item()

            q_values = self.forward(state)
            target_f = q_values.clone()
            target_f[0][action] = target

            self.optimizer.zero_grad()
            loss = F.mse_loss(q_values, target_f)
            loss.backward()
            self.optimizer.step()
            self.loss.append(loss.item())

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# Parameters
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
alpha = 0.0001
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995

# Create the agent
agent = DQN(state_size, action_size, alpha, gamma, epsilon, epsilon_min, epsilon_decay)

# Training loop
episode_num = 100
for i in range(episode_num):
    print(f"Episode {i}")
    state = env.reset()[0]  # For Gym version >= 0.22, reset returns a tuple
    total_reward = 0
    while True:
        action = agent.act(state)
        next_state, reward, done, _, info = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        if done:
            print(f"Episode ended with total reward: {total_reward}\n")
            break
        agent.replay(batch_size)

env.close()
