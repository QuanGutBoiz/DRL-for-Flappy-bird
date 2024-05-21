import random
import numpy as np

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0

    def add(self, state, action, reward, next_state, done):
        max_priority = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.priorities[self.pos] = max_priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.pos]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        
        # states = np.array([s.cpu().numpy() for s in batch[0]])  # Chuyển đổi các tensor từ CUDA về CPU trước khi chuyển đổi thành numpy array
        # actions = np.array(batch[1])
        # rewards = np.array(batch[2])
        # next_states = np.array([s.cpu().numpy() for s in batch[3]])  # Chuyển đổi các tensor từ CUDA về CPU trước khi chuyển đổi thành numpy array
        # dones = np.array(batch[4])

        return samples, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        
        for idx, priority in zip(batch_indices, batch_priorities):
            self.priorities[idx] = priority
# [15 24 22 22 23 12 16 24  3 25 17 18  9  1 21 32 16 28 12 12  4 13 25  3
#  27  7 24 22 24 15  1  5]