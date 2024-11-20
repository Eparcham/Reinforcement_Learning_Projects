import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean, cityblock, chebyshev
import random


# Define the environment
class EmbeddingEnvironment:
    def __init__(self, threshold=0.8):
        self.threshold = threshold

    def reset(self, embedding_1, embedding_2, label):
        self.embedding_1 = embedding_1
        self.embedding_2 = embedding_2
        self.label = label
        return self._get_state()

    def _get_state(self):
        cosine_sim = cosine_similarity(self.embedding_1, self.embedding_2).mean()
        euclidean_dist = np.mean([euclidean(e1, e2) for e1, e2 in zip(self.embedding_1, self.embedding_2)])
        manhattan_dist = np.mean([cityblock(e1, e2) for e1, e2 in zip(self.embedding_1, self.embedding_2)])
        chebyshev_dist = np.mean([chebyshev(e1, e2) for e1, e2 in zip(self.embedding_1, self.embedding_2)])
        return cosine_sim, euclidean_dist, manhattan_dist, chebyshev_dist

    def step(self, action):
        similarity_score, euclidean_dist, manhattan_dist, chebyshev_dist = self._get_state()
        done = True

        # Reward function based on input label
        if (self.label == 1 and action == 1) or (self.label == 0 and action == 0):
            reward = 1  # Correct decision
        else:
            reward = -1  # Incorrect decision

        return (similarity_score, euclidean_dist, manhattan_dist, chebyshev_dist), reward, done


# Define the agent
class RLAgent:
    def __init__(self, actions, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0, exploration_decay=0.995):
        self.q_table = {}
        self.actions = actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay

    def get_q_value(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, state):
        if random.uniform(0, 1) < self.exploration_rate:
            return random.choice(self.actions)
        q_values = [self.get_q_value(state, a) for a in self.actions]
        return self.actions[np.argmax(q_values)]

    def learn(self, state, action, reward):
        current_q = self.get_q_value(state, action)
        updated_q = current_q + self.learning_rate * (reward - current_q)
        self.q_table[(state, action)] = updated_q
        # Decay exploration rate
        self.exploration_rate *= self.exploration_decay


# Training the agent
def train_agent(embedding_pairs, labels, episodes=1000):
    env = EmbeddingEnvironment()
    agent = RLAgent(actions=[0, 1])

    for episode in range(episodes):
        idx = random.choice(range(len(embedding_pairs)))
        embedding_1, embedding_2 = embedding_pairs[idx]
        label = labels[idx]
        state = env.reset(embedding_1, embedding_2, label)

        action = agent.choose_action(state)
        _, reward, done = env.step(action)

        agent.learn(state, action, reward)

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{episodes}, Exploration Rate: {agent.exploration_rate:.4f}")

    return agent


# Example usage
if __name__ == "__main__":
    # Example embedding matrices
    embedding_1 = np.random.rand(10, 5)
    embedding_2 = np.random.rand(10, 5)
    embedding_3 = embedding_1 + np.random.normal(0, 0.01, (10, 5))  # Slightly different from embedding_1

    # Training data (pairs of embeddings) and labels
    embedding_pairs = [
        (embedding_1, embedding_2),  # Not similar
        (embedding_1, embedding_3),  # Similar
    ]
    labels = [0, 1]  # Labels indicating similarity (0: Not similar, 1: Similar)

    # Train the agent
    trained_agent = train_agent(embedding_pairs, labels)

    # Test the agent
    test_embedding_1 = np.random.rand(10, 5)
    test_embedding_2 = test_embedding_1 + np.random.normal(0, 0.01, (10, 5))
    cosine_sim, euclidean_dist, manhattan_dist, chebyshev_dist = cosine_similarity(test_embedding_1,
                                                                                   test_embedding_2).mean(), np.mean(
        [euclidean(e1, e2) for e1, e2 in zip(test_embedding_1, test_embedding_2)]), np.mean(
        [cityblock(e1, e2) for e1, e2 in zip(test_embedding_1, test_embedding_2)]), np.mean(
        [chebyshev(e1, e2) for e1, e2 in zip(test_embedding_1, test_embedding_2)])
    state = (cosine_sim, euclidean_dist, manhattan_dist, chebyshev_dist)
    action = trained_agent.choose_action(state)
    print("Action (1 means similar, 0 means not similar):", action)
