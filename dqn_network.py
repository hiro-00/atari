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

    def copy_variables(self, model):
        new_variables = model.get_variables()
        copy_func = []
        for i, _ in enumerate(self.variable_list):
            copy_func.append(self.variable_list[i].assign(new_variables[i]))
        self.session.run(copy_func)

    def train(self, state, reward, next_q, action):
        self.session.run(self.optimizer, feed_dict=
                           {
                            self.input : state,
                            self.reward : reward,
                            self.next_q : next_q,
                            self.actions : action
                           })

    def eval(self, state):
        state = np.array([state])
        return self.session.run(self.q_values, feed_dict = {self.input : state})


    def _add_hidden_layer(self, input, node_num):
        print(input.get_shape())
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


    def _add_conv_layer(self, input, filter_size, color_channel, filter_num, stride):
        w = tf.Variable(tf.truncated_normal([filter_size, filter_size, color_channel, filter_num], stddev=0.1))
        b = tf.Variable(tf.constant(0.1, shape=[filter_num]))
        self.variable_list.append(w)
        self.variable_list.append(b)
        print(input.get_shape())
        print(w.get_shape())
        conv1 = tf.nn.conv2d(input, w, strides=[1, stride, stride, 1], padding='VALID')
        return tf.nn.relu(tf.add(conv1, b))
