from collections import namedtuple
import random
from collections import deque


class ReplayMemory():
    def __init__(self, memory_size = 500, batch_size = 50):
        self.Content = namedtuple('memory', 'state reward next_q action')
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.memory = deque()

    def clear(self):
        self.memory.clear()

    def add(self, observation, reward, next_q, action):
        self.memory.append(self.Content(observation, reward, next_q, action))
        if len(self.memory) > self.memory_size:
            self.memory.popleft()

    def get(self, size = None):
        if size == None:
            batch_size = self.batch_size
        batch_size = min(batch_size, len(self.memory))
        samples = random.sample(list(self.memory), batch_size)
        states=[]
        rewards=[]
        next_q=[]
        actions=[]
        for sample in samples:
            states.append(sample.state)
            rewards.append(sample.reward)
            next_q.append(sample.next_q)
            actions.append(sample.action)
        return (states, rewards, next_q, actions)