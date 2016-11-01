import gym
import time
import random
import numpy as np
import tensorflow as tf
from dqn_agent import DqnAgent
from dqn_network import DqnNetwork
from replay_memory import ReplayMemory

import functools
class AtariNetwork(DqnNetwork):
    def __init__(self, session, input_width, input_height, frame_num, action_space, learning_rate = 0.00025 , discount_rate = 0.99):
        super(AtariNetwork, self).__init__(session, action_space, learning_rate, discount_rate)
        self.MIN_GRAD = 0.01
        self.MOMENTUM = 0.95

        self.input = tf.placeholder('float32', [None, input_width, input_height, frame_num], name='input')
        conv1 = self._add_conv_layer(self.input, filter_size=8, color_channel=frame_num, filter_num=32, stride=4)
        conv2 = self._add_conv_layer(conv1, filter_size=8, color_channel=32, filter_num=64, stride=2)
        conv3 = self._add_conv_layer(conv2, filter_size=3, color_channel=64, filter_num=64, stride=1)

        conv3_shape = conv3.get_shape().as_list()
        conv3_flat = tf.reshape(conv3, [-1, conv3_shape[1]*conv3_shape[2]*conv3_shape[3]])
        hidden1 = self._add_hidden_layer(conv3_flat, 512)
        self.q_values = self._add_output_layer(hidden1, action_space)

        self.next_q = tf.placeholder('float32', shape=[None])
        self.reward = tf.placeholder("float32", shape=[None])
        q_new = tf.add(self.reward,  tf.mul(self.DISCOUNT_RATE, self.next_q))

        self.actions = tf.placeholder('int32', shape=[None])
        q_current = tf.reduce_sum(tf.mul(self.q_values, tf.one_hot(self.actions, action_space)), reduction_indices=1)
        error = tf.abs(q_new - q_current)
        quadratic_part = tf.clip_by_value(error, 0.0, 1.0)
        linear_part = error - quadratic_part
        loss = tf.reduce_mean(0.5 * tf.square(quadratic_part) + linear_part)

        self.optimizer = tf.train.RMSPropOptimizer(self.LEARNING_RATE, momentum = self.MOMENTUM, epsilon=self.MIN_GRAD).minimize(loss)


from skimage.color import rgb2gray
from skimage.transform import resize
class FramePreProcessor():
    def __init__(self, resize_width, resize_height, state_length):
        self.RESIZE_WIDTH = resize_width
        self.RESIZE_HEIGHT = resize_height
        self.STATE_LENGTH = state_length

    def process(self, frame, prev_frame):
        processed_frame = np.maximum(frame, prev_frame)
        processed_frame = np.uint8(resize(rgb2gray(processed_frame), (self.RESIZE_WIDTH, self.RESIZE_HEIGHT)))
        processed_frame = [processed_frame for _ in range(self.STATE_LENGTH)]
        processed_frame = np.concatenate([processed_frame[i][...,np.newaxis] for i in range(self.STATE_LENGTH)] ,axis=2)
        return processed_frame

RESIZE_WIDTH = 84
RESIZE_HEIGHT = 84
FRAME_STACK_NUM = 4
REPLAY_MEMORY_SIZE = 400000
BATCH_SIZE = 32
TRAIN_INTERVAL = 4
TARGET_UPDATE_INTERVAL = 10000
ACTION_INTERVAL = 4

env = gym.make('Breakout-v0')
observation = env.reset()
frame_preprocessor = FramePreProcessor(RESIZE_WIDTH, RESIZE_HEIGHT, FRAME_STACK_NUM)

session = tf.Session()
state_space = len(observation)
action_space = env.action_space.n
model = AtariNetwork(session, RESIZE_WIDTH, RESIZE_HEIGHT, FRAME_STACK_NUM, action_space)
target = AtariNetwork(session, RESIZE_WIDTH, RESIZE_HEIGHT, FRAME_STACK_NUM, action_space)
agent = DqnAgent(session, action_space, model, target, ReplayMemory(REPLAY_MEMORY_SIZE, BATCH_SIZE),
                 TRAIN_INTERVAL, TARGET_UPDATE_INTERVAL, ACTION_INTERVAL, 1000000)

step = 0
TOTAL_EPISODES = 1000
INITIAL_ACTION_SKIPS = 30
for ep in range(TOTAL_EPISODES):
    observation = env.reset()
    prev_observation = observation
    for _ in range(random.randint(1, INITIAL_ACTION_SKIPS)):
        prev_observation = observation
        observation, _, _, _ = env.step(0)

    agent.begin_episode(frame_preprocessor.process(observation, prev_observation))

    done = False
    while not done:
        env.render()

        step += 1
        action = agent.get_action()
        observation, reward, done, info = env.step(action)

        images = frame_preprocessor.process(observation, prev_observation)
        prev_observation = observation

        if ep % 50 == 0:
            env.render()
            time.sleep(0.05)
        if done:
            agent.update(images, 0, done)
            print(str(ep) + ": Eposode done " + str(t) + ":  " + str(agent.explore_rate))

        agent.update(images, reward, done)
