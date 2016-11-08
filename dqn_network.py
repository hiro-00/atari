import tensorflow as tf
import numpy as np

class DqnNetwork():
    def __init__(self, session, action_space, learning_rate, discount_rate):
        self.LEARNING_RATE = learning_rate
        self.DISCOUNT_RATE = discount_rate

        self.session = session
        self.action_space = action_space
        self.variable_list = []


    def get_variables(self):
        return self.variable_list

    def train(self, state, reward, next_q, action):
        self.session.run(self.optimizer, feed_dict=
                           {
                            self.input : np.float32(np.array(state)/255.0),
                            self.reward : reward,
                            self.next_q : next_q,
                            self.actions : action
                           })

    def eval(self, state):
        state = np.array([state])
        return self.session.run(self.q_values, feed_dict = {self.input : np.float32(state/255.0)})


    def _add_hidden_layer(self, input, node_num):
        dim = int(input.get_shape()[1])
        w = tf.Variable(tf.random_normal(shape=[dim, node_num], stddev=0.01))
        b = tf.Variable(tf.constant(0.1, shape=[node_num]))
        self.variable_list.append(w)
        self.variable_list.append(b)
        return tf.nn.relu(tf.add(tf.matmul(input, w), b))

    def _add_output_layer(self, input, node_num):
        dim = int(input.get_shape()[1])
        w = tf.Variable(tf.random_normal(shape=[dim, node_num], stddev=0.01))
        b = tf.Variable(tf.constant(0.1, shape=[node_num]))
        self.variable_list.append(w)
        self.variable_list.append(b)
        return tf.add(tf.matmul(input, w), b)


    def _add_conv_layer(self, input, filter_size, color_channel, filter_num, stride):
        w = tf.Variable(tf.truncated_normal([filter_size, filter_size, color_channel, filter_num], stddev=0.1))
        b = tf.Variable(tf.constant(0.1, shape=[filter_num]))
        self.variable_list.append(w)
        self.variable_list.append(b)
        conv1 = tf.nn.conv2d(input, w, strides=[1, stride, stride, 1], padding='VALID')
        return tf.nn.relu(tf.add(conv1, b))
