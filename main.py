import gym
import time
import numpy as np
import tensorflow as tf
from dqn_agent import DqnAgent
from dqn_network import DqnNetwork

import functools
class AtariNetwork(DqnNetwork):
    def __init__(self, session, input_width, input_height, action_space, learning_rate = 0.00025 , discount_rate = 0.99):
        super(AtariNetwork, self).__init__(session, action_space, learning_rate, discount_rate)
        self.MIN_GRAD = 0.01
        self.MOMENTUM = 0.95

        self.input = tf.placeholder('float32', [None, input_width, input_height, 1], name='input')
        conv1 = self._add_conv_layer(self.input, 8, 1, 32, 4)
        conv2 = self._add_conv_layer(conv1, 8, 32, 64, 2)
        conv3 = self._add_conv_layer(conv2, 3, 64, 64, 1)

        conv3_shape = conv3.get_shape().as_list()
        conv3_flat = tf.reshape(conv3, [-1, conv3_shape[1]*conv3_shape[2]*conv3_shape[3]])
        hidden1 = self._add_hidden_layer(conv3_flat, 512)
        self.q_values = self._add_output_layer(hidden1, action_space)

        self.next_q = tf.placeholder('float32', shape=[None])
        self.reward = tf.placeholder("float32", shape=[None])
        q_new = tf.add(self.reward,  tf.mul(self.DISCOUNT_RATE, self.next_q))

        self.actions = tf.placeholder('int32', shape=[None])
        q_current = tf.reduce_sum(tf.mul(self.q_values, tf.one_hot(self.actions, action_space)), reduction_indices=1)
        #q_current = tf.mul(self.q_values, self.actions)
        error = tf.sub(q_new, q_current)
        loss = tf.reduce_sum(tf.mul(tf.constant(0.5),tf.pow(error, 2)))

        self.optimizer = tf.train.RMSPropOptimizer(self.LEARNING_RATE, momentum = self.MOMENTUM, epsilon=self.MIN_GRAD).minimize(loss)


from skimage.color import rgb2gray
from skimage.transform import resize
class FramePreProcessor():
    def __init__(self, resize_width, resize_height):
        self.RESIZE_WIDTH = resize_width
        self.RESIZE_HEIGHT = resize_height

    def process(self, frames):
        processed = np.uint8(resize(rgb2gray(frames), (self.RESIZE_WIDTH, self.RESIZE_HEIGHT, 1)))
        print("--")
        print(np.array(processed).shape)
        return np.array(processed)

RESIZE_WIDTH = 84
RESIZE_HEIGHT = 84
NN_INPUT_FRAMENUM = 4

env = gym.make('Breakout-v0')
observation = env.reset()
total_episode = 100
frame_preprocessor = FramePreProcessor(RESIZE_WIDTH, RESIZE_HEIGHT)

session = tf.Session()
state_space = len(observation)
action_space = env.action_space.n
model = AtariNetwork(session, RESIZE_WIDTH, RESIZE_HEIGHT, action_space)
target = AtariNetwork(session, RESIZE_WIDTH, RESIZE_HEIGHT, action_space)
agent = DqnAgent(session, action_space, model, target)

#state = frame_preprocessor.process(frame_queue)
step = 0
for ep in range(total_episode):
    observation = env.reset()
    agent.begin_episode(frame_preprocessor.process(observation))
    for t in range(10000000):
        env.render()

        step += 1
        action = agent.get_action()
        observation, reward, done, info = env.step(action)
        images = frame_preprocessor.process(observation)
        print(images.shape)
        if ep % 50 == 0 and t < 300:
            env.render()
            time.sleep(0.05)
        if done:
            agent.update(step, images, 0, done)
            print(str(ep) + ": Eposode done " + str(t) + ":  " + str(agent.explore_rate))
            break

        agent.update(step, images, reward, done)
