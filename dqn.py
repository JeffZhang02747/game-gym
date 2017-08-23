import gym
import tensorflow as tf
import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adagrad
from keras.models import load_model

# with both target net and experience replay
class deepQNet:
    # feature_size = 128
    # action_size = 7

    discount = 0.99
    epsilon = 0.05
    maxSteps = 500
    episodes = 5000
    bufferSize = 1000
    batchSize = 50
    hiddenSize = 10
    learning_rate = 0.1
    experbuffer = deque(maxlen=bufferSize)
    model_name = "atari_model_shape"

    def __init__(self, env):
        self.env = env

        self.action_size = self.env.action_space.n
        self.feature_size = self.env.observation_space.shape
        # self.feature_size = 128
        # self.env.observation_space.shape


    def init_net(self):
        try:
            self.qmodel = load_model(self.model_name)

        except:
            self.qmodel = Sequential()
            self.qmodel.add(Dense(self.hiddenSize, input_shape=self.feature_size, activation='relu'))
            self.qmodel.add(Dense(self.hiddenSize, activation='relu'))
            # identity 
            self.qmodel.add(Dense(self.action_size, activation='linear'))
            self.qmodel.compile(loss='mse',optimizer=Adagrad(lr=self.learning_rate))

        # never train this though, only use to store target network
        qmodel_config = self.qmodel.get_config()
        self.targetmodel = Sequential.from_config(qmodel_config)  
        self.updateTargetNet()


    def updateTargetNet(self):
        self.targetmodel.set_weights(self.qmodel.get_weights())

    # give action suggestion
    def predict(self, obs):
        dice = np.random.rand()
        if dice <= self.epsilon:
            return random.randrange(0, self.action_size)
        output = self.qmodel.predict(obs)
        return np.argmax(output)

    def remember(self, state, action, reward, next_state, done):
        self.experbuffer.append((state, action, reward, next_state, done))

    def learnByReplay(self):
        samples = random.sample(self.experbuffer, self.batchSize)
        total_shape = (self.batchSize,) + (self.feature_size)
        all_next_states = np.zeros(total_shape)
        all_cur_states = np.zeros(total_shape)

        # all_next_states = np.zeros((self.batchSize, self.feature_size))
        # all_cur_states = np.zeros((self.batchSize, self.feature_size))

        for index, (state, action, reward, next_state, done) in enumerate(samples):
            all_next_states[index] = next_state
            all_cur_states[index] = state

        target_output = self.targetmodel.predict(all_next_states)
        learning_output = self.qmodel.predict(all_cur_states)

        for index, (state, action, reward, next_state, done) in enumerate(samples):
            replace_val = reward
            if not done:
                replace_val += self.discount * np.amax(target_output[index])
            else:
                replace_val = 0

            learning_output[index, action] = replace_val

        self.qmodel.fit(all_cur_states, learning_output, batch_size=self.batchSize, epochs=1, verbose=0)


    def simulate(self):
        perf = []
        convert_shape = (1,) + self.feature_size

        for i_episode in xrange(0, self.episodes):
            observation = self.env.reset()
            observation = np.reshape(observation, convert_shape)

            score = 0
            while True:
                self.env.render()
                action = self.predict(observation)
                new_observation, reward, done, info = self.env.step(action)
                new_observation = np.reshape(new_observation, convert_shape)

                score += reward
                self.remember(observation, action, reward, new_observation, done)
                if len(self.experbuffer) >= self.batchSize:
                    self.learnByReplay()

                if done:
                    print("Episode finished with {} score".format(score))
                    perf.append(score)
                    break
                observation = new_observation

            if i_episode % 2 == 1:
                self.updateTargetNet()
                self.qmodel.save(self.model_name)


        return perf

env = gym.make('Assault-ram-v0')
net = deepQNet(env)
net.init_net()
scores = net.simulate()
print scores