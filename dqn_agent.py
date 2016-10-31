from replay_memory import ReplayMemory
import tensorflow as tf
import numpy as np

class DqnAgent():
    def __init__(self, session, action_space, model, target):
        self.explore_rate = 1.0
        self.explore_decay = 0.9999

        self.action_space = action_space

        self.action = 0
        self.tick = 1

        self.session = session
        self.model = model
        self.target_network = target
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