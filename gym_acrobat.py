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
        #error = tf.clip_by_value(error, -1.0, 1.0)
        loss = tf.reduce_mean(tf.pow(error, 2))

        self.optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE).minimize(loss)



'''

class AcrobatAgent():
    def __init__(self, session, observation, action_space):
        self.explore_rate = 1.0
        self.explore_decay = 0.9999

        self.space_state = len(observation)
        self.action_space = action_space

        self.action = 0
        self.tick = 1

        self.session = session
        self.model = AcrobatNn(len(observation), action_space, self.session)
        self.target_network = AcrobatNn(len(observation), action_space, self.session)
        self.session.run(tf.initialize_all_variables())
        self.__initialize_target_network()

        self.replay_memory = ReplayMemory()

    def get_tf_variables(self):
        return self.model.get_variables() + self.target_network.get_variables()

    def __initialize_target_network(self):
        model_variables = self.model.get_variables()
        target_variables = self.target_network.get_variables()
        self._target_network_updates = [target_variables[i].assign(model_variables[i]) for i in range(len(model_variables))]
        self.update_target_network()

    def update_target_network(self):
        self.session.run(self._target_network_updates)

    def begin_episode(self, observation):
        self.state = observation

    def update(self, step, new_state, reward, done):
        next_q = 0
        if done == False:
            next_q = np.max(self.target_network.eval(new_state))
        self.replay_memory.add(self.state, reward, next_q, self.action)
        (batch_state, batch_reward, batch_next_q, batch_action) = self.replay_memory.get()
        self.model.train(batch_state, batch_reward, batch_next_q, batch_action)
        if self.tick % 50 == 0:
            self.update_target_network()

        self.state = new_state
        self.action = self.__decide_action(new_state)

        self.tick += 1
        self.explore_rate = self.explore_rate * self.explore_decay

    def get_action(self):
        return self.action


    def __decide_action(self, state):
        # epsilon_greedy
        if np.random.uniform(0, 1) < self.explore_rate:
            return np.random.randint(0,self.action_space)
        else:
            return self.__get_best_action(state)


    def __get_best_action(self, state):
        return np.argmax(self.model.eval(state))
'''

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
saver.restore(session, save_path='save/result')
for ep in range(total_episode):
    if ep % 50 == 0:
        saver.save(session, save_path="./save/result")

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
