import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import numpy as np
import time
import random
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import matplotlib.pyplot as plt
# from tensorflow.keras.callbacks import TensorBoard

from PathEnv_v4_general import MazeEnv

tf.keras.utils.disable_interactive_logging()

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=20000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.update_target_frequency = 10
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()  # 新增：初始化時就更新目標網路

        self.loss_history = []

        # self.tensorboard = TensorBoard(log_dir="logs/{}".format(time.time()))

    def _build_model(self):
        model = Sequential()
        model.add(Conv2D(256, (3, 3), input_shape=(self.state_size, self.state_size, 1), activation='relu'))
        model.add(Conv2D(256, (3, 3), activation='relu'))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate), metrics=['accuracy'])
        return model
    
    def preprocess_state(self, state):
        # Preprocess the state if necessary
        return state.reshape((1, self.state_size, self.state_size, 1))
    
    def remember(self, state, action, reward, next_state, done):
        # state = self.preprocess_state(state)
        # next_state = self.preprocess_state(next_state)
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = self.preprocess_state(state)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])
    
    def replay(self, batch_size):
        for _ in range(batch_size):
            state, action, reward, next_state, done = random.choice(self.memory)
            state = self.preprocess_state(state)
            next_state = self.preprocess_state(next_state)
            # target = self.model.predict(state)
            target = self.target_model.predict(state)

            if done:
                target[0][action] = reward
            else:
                # a = np.argmax(self.model.predict(next_state)[0])
                # target[0][action] = reward + self.gamma * (self.model.predict(next_state)[0][a])

                # a = np.argmax(self.target_model.predict(next_state)[0])
                a = np.argmax(self.model.predict(next_state)[0])
                target[0][action] = reward + self.gamma * (self.target_model.predict(next_state)[0][a])
            self.model.fit(state, target, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def plot_loss(self):
        plt.figure(2)
        plt.plot(self.loss_history, label='loss')
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

            

def save_model(agent, episode, model_dir='models'):
    try:
        print('Model saving')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_path = os.path.join(model_dir, f'model_{episode}.keras')  # 加入情节编号
        agent.model.save(model_path)
        print(f'Model saved to {model_path}')
    except Exception as e:
        print(f"Error saving model: {e}")


if __name__ == '__main__':
    state_size = 10
    num_obstacles = 20
    action_size = 4
    env = MazeEnv(state_size, num_obstacles)

    agent = DQNAgent(state_size, action_size)
    batch_size = 100
    EPISODES = 1000
    episodes_max_step = env.get_max_steps()

    for episode in range(EPISODES):
        state = env.hit_reset()
        total_reward = 0
        step = 0

        for time in range(episodes_max_step):
            step+=1

            action = agent.act(state)
            next_state, reward, done = env.step(action)
            # reward = reward if not done else -10
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward+=reward

            env.render(time = 0.0001)

            if done:
                print("Episode: {}/{}, Total Reward: {}, Epsilon: {:.2}".format(episode, EPISODES, total_reward, agent.epsilon))
                break

        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

            # 新增：儲存每次訓練的 loss
            loss = agent.model.history.history['loss']
            agent.loss_history.append(loss)

        if episode%50 ==0 and episode!=0:
            save_model(agent, episode)

        if episode % agent.update_target_frequency == 0:
            agent.update_target_model()  # 新增：定期更新目標網路

        agent.plot_loss()

    
