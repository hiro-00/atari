import gym
import numpy as np
import tensorflow as tf
import time
import dateutil
import functools


class PollNn():
    def __init__(self, state_dim, action_num, session):
        self.LEARNING_RATE = 0.01
        self.DISCOUNT_RATE = 0.99
        self.MIN_GRAD = 0.01
        self.variable_list = []

        self.input = tf.placeholder('float32', shape=[None, state_dim])
        hidden1 = self._add_hidden_layer(self.input, 100)
        #hidden2 = self._add_hidden_layer(hidden1, 100)
        self.q_values = self._add_output_layer(hidden1, action_num)

        self.next_q = tf.placeholder('float32', shape=[None])
        self.reward = tf.placeholder("float32", shape=[None])
        q_new = tf.add(self.reward,  tf.mul(self.DISCOUNT_RATE, self.next_q))

        self.actions = tf.placeholder('int32', shape=[None])
        q_current = tf.reduce_sum(tf.mul(self.q_values, tf.one_hot(self.actions, 2)), reduction_indices=1)
        error = tf.sub(q_new, q_current)
        #error = tf.clip_by_value(error, -1.0, 1.0)
        loss = tf.reduce_mean(tf.pow(error, 2))

        self.optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE).minimize(loss)

        self.session = session

    def get_variables(self):
        return self.variable_list


    def train(self, state, reward, next_q, action):
        self.session.run(self.optimizer, feed_dict=
                           {
                            self.input : state,
                            self.reward : reward,
                            self.next_q : next_q,
                            self.actions : action
                           })

    def eval(self, state):
        state = np.reshape(state, (1, 4))
        return self.session.run(self.q_values, feed_dict = {self.input : state})


    def _add_hidden_layer(self, input, node_num):
        dim = int(input.get_shape()[1])
        flat_input = input
        w = tf.Variable(tf.random_normal(shape=[dim, node_num], stddev=0.01))
        b = tf.Variable(tf.constant(0.1, shape=[node_num]))
        self.variable_list.append(w)
        self.variable_list.append(b)
        return tf.nn.relu(tf.add(tf.matmul(flat_input, w), b))

    def _add_output_layer(self, input, node_num):
        dim = int(input.get_shape()[1])
        w = tf.Variable(tf.random_normal(shape=[dim, node_num], stddev=0.01))
        b = tf.Variable(tf.constant(0.1, shape=[node_num]))
        self.variable_list.append(w)
        self.variable_list.append(b)
        return tf.add(tf.matmul(input, w), b)


class PollAgent():
    def __init__(self, observation):
        self.session = tf.Session()
        self.model = PollNn(len(observation), 2, self.session)
        self.target_newtork = PollNn(len(observation), 2, self.session)
        self.session.run(tf.initialize_all_variables())

        model_variables = self.model.get_variables()
        target_network_variables = self.target_newtork.get_variables()
        self.target_network_update_funcs = []
        for i, _ in enumerate(model_variables):
            self.target_network_update_funcs.append(target_network_variables[i].assign(model_variables[i]))
        self.update_target_network()

        self.replay_memory = ReplayMemory()
        self.EXPLORE_RATE = 1.0
        self.EXPLORE_DECAY = 0.9995
        self.action = 0
        self.tick = 1

    def update_target_network(self):
        self.session.run(self.target_network_update_funcs)

    def begin_episode(self, observation):
        self.state = observation

    def update(self, step, new_state, reward, done):
        next_q = 0
        if done == False:
            next_q = np.max(self.target_newtork.eval(new_state))
        self.replay_memory.add(self.state, reward, next_q, self.action)
        (batch_state, batch_reward, batch_next_q, batch_action) = self.replay_memory.get()
        self.model.train(batch_state, batch_reward, batch_next_q, batch_action)
        if self.tick % 50 == 0:
            self.update_target_network()
        #self.replay_memory.clear()
        self.state = new_state
        self.action = self.__decide_action(new_state)

        self.tick += 1
        self.EXPLORE_RATE = self.EXPLORE_RATE * self.EXPLORE_DECAY

    def get_action(self):
        return self.action


    def __decide_action(self, state):
        # epsilon_greedy
        if np.random.uniform(0, 1) < self.EXPLORE_RATE:
            return np.random.randint(0, 2)
        else:
            return self.__get_best_action(state)


    def __get_best_action(self, state):
        return np.argmax(self.model.eval(state))

from collections import namedtuple
import random
from collections import deque
class ReplayMemory():
    def __init__(self, size = 1000):
        self.Content = namedtuple('memory', 'state reward next_q action')
        self.states = []
        self.rewards = []
        self.next_q = []
        self.actions = []
        self.size = size
        self.memory = deque()

    def clear(self):
        self.memory.clear()

    def add(self, observation, reward, next_q, action):
        self.memory.append(self.Content(observation, reward, next_q, action))
        if len(self.memory) > self.size:
            self.memory.popleft()

    def get(self):
        size = min(300, len(self.memory))
        samples = random.sample(list(self.memory), size)
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

env = gym.make('CartPole-v0')
env.reset()
total_episode = 3000


observation = env.reset()


agent = PollAgent(observation)
step = 0
for ep in range(total_episode):
    observation = env.reset()
    agent.begin_episode(observation)
    for t in range(210):
        step += 1
        action = agent.get_action()
        observation, reward, done, info = env.step(action)
        if done:
            if t < 195:
                agent.update(step, observation, 0, done)
            print(str(ep)+": Eposode done "+str(t)+":  "+str(agent.EXPLORE_RATE))
            break

        agent.update(step, observation, reward, done)
