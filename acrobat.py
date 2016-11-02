import gym
import numpy as np
import tensorflow as tf
import time
from dqn_agent import DqnAgent
from dqn_network import DqnNetwork
from replay_memory import ReplayMemory


class AcrobatNn(DqnNetwork):
    def __init__(self, session, state_space, action_space, learning_rate, discount_rate):
        super(AcrobatNn, self).__init__(session, action_space, learning_rate, discount_rate)

        H1 = 50
        H2 = 50

        self.input = tf.placeholder('float32', shape=[None, state_space])
        hidden1 = self._add_hidden_layer(self.input, H1)
        hidden2 = self._add_hidden_layer(hidden1, H2)
        self.q_values = self._add_output_layer(hidden2, action_space)

        self.next_q = tf.placeholder('float32', shape=[None])
        self.reward = tf.placeholder("float32", shape=[None])
        q_new = tf.add(self.reward,  tf.mul(self.DISCOUNT_RATE, self.next_q))

        self.actions = tf.placeholder('int32', shape=[None])
        q_current = tf.reduce_sum(tf.mul(self.q_values, tf.one_hot(self.actions, action_space)), reduction_indices=1)
        error = tf.sub(q_new, q_current)
        loss = tf.reduce_mean(tf.pow(error, 2))

        self.optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE).minimize(loss)

LEARNING_RATE = 0.001
DISCOUNT_RATE = 0.99

env = gym.make('Acrobot-v1')
env.reset()
total_episode = 3000


observation = env.reset()
session = tf.Session()
state_space = len(observation)
action_space = env.action_space.n
model = AcrobatNn(session, state_space, action_space, LEARNING_RATE, DISCOUNT_RATE)
target = AcrobatNn(session, state_space, action_space, LEARNING_RATE, DISCOUNT_RATE)
agent = DqnAgent(session, action_space, model, target, ReplayMemory())
step = 0
saver = tf.train.Saver(agent.get_tf_variables())
saver.restore(session, save_path='acrobat_ckpt/150')
for ep in range(total_episode):
    if ep % 50 == 0:
        saver.save(session, save_path="./acrobat_ckpt/"+str(ep))

    observation = env.reset()
    agent.begin_episode(observation)
    for t in range(21000):
        step += 1
        action = agent.get_action()
        observation, reward, done, info = env.step(action)
        if ep %50==0 and t < 300:
            env.render()
            time.sleep(0.05)
        if done:
            agent.update(observation, 1000, done)
            print(str(ep)+": Eposode done "+str(t)+":  "+str(agent.explore_rate))
            break

        agent.update(observation, reward, done)
