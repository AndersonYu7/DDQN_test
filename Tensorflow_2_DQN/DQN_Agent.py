import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from collections import deque
import matplotlib.pyplot as plt
import os

from DQN_env import GridWorldEnvironment

import random

tf.keras.utils.disable_interactive_logging()

# # 自定義的GridWorldEnvironment
# class GridWorldEnvironment:
#     # ... (您的GridWorldEnvironment的定义，与之前一样)

# DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self._state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        self.model = self._build_model()

        self._losses = []

    def _build_model(self):
        model = models.Sequential()

        model.add(layers.Conv2D(32, (3, 3), input_shape=(self._state_size[0], self._state_size[1], 1), activation='relu'))
        # model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        # model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())

        model.add(layers.Dense(24, activation='relu'))
        model.add(layers.Dense(24, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=tf.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(np.reshape(state, (1, *self._state_size, 1)))
        return np.argmax(act_values[0])

    # def replay(self, batch_size):
    #     minibatch = random.sample(self.memory, batch_size)
    #     for state, action, reward, next_state, done in minibatch:
    #         target = reward
    #         if not done:
    #             target = reward + self.gamma * np.amax(self.model.predict(np.reshape(next_state, (1, *self._state_size, 1)))[0])
    #         target_f = self.model.predict(np.reshape(state, (1, *self._state_size, 1)))[0]
    #         target_f[action] = target

    #         print("state shape:", state.shape)
    #         print("target_f shape:", target_f.shape)    
    #         breakpoint()

    #         history = self.model.fit(state, target_f, epochs=1, verbose=0)
    #         loss = history.history['loss'][0]
    #         self._losses.append(loss)

    #     if self.epsilon > self.epsilon_min:
    #         self.epsilon *= self.epsilon_decay

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        loss = 0
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state.reshape(1, *self._state_size, 1))[0])
            target_f = self.model.predict(np.reshape(state, (1, *self._state_size, 1)))
            target_f[0][action] = target

            history = self.model.fit(np.reshape(state, (1, *self._state_size)), np.reshape(target_f, (1, 4)), epochs=1, verbose=0)

            loss += history.history['loss'][0]

        loss/=len(minibatch)
        self._losses.append(loss)
            
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

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

# 初始化 GridWorld 環境
env = GridWorldEnvironment(grid_size=10, num_obstacles=10)
# state_size = np.prod(np.array(env.current_observation()).shape)  # Flatten observation
state_size = env.current_observation().shape  # 这里使用.shape获取形状


# 初始化 DQN Agent
action_size = 4  # 因為有上、下、左、右四種動作
agent = DQNAgent(state_size, action_size)

# 訓練 DQN Agent
num_episodes = 1000
batch_size = 64
episodes_step = env.get_max_steps()

scores = []  # 初始化得分列表
steps = []
average_rewards = []

for episode in range(num_episodes):
    s = env.reset()
    # state = np.reshape(s, [1, state_size])
    state=s
    total_reward = 0
    step = 0

    for time in range(episodes_step):  # 限制每個episode的步數
        step+=1

        env.render(time = 0.0001)

        action = agent.act(state)
        next_state, reward, done = env.step(action)
        reward = reward if not done else -10  # 惩罚结束时的reward
        # next_state = np.reshape(next_state, [1, state_size])
        print(action)
        print(next_state)

        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        if done:
            print("Episode: {}/{}, Total Reward: {}, Epsilon: {:.2}".format(episode, num_episodes, total_reward, agent.epsilon))
            break

    if len(agent.memory) > batch_size:
        agent.replay(batch_size)

    scores.append(total_reward)
    steps.append(step)
    average_rewards.append(np.mean(scores[-100:]))  # 更新平均奖励

    if episode%100 ==0 and episode!=0:
        save_model(agent, episode)

    if episode % 10 == 0 and episode != 0:
        plt.figure(1)
        plt.subplot(241)
        plt.plot(scores, color='deepskyblue', linewidth=1)
        plt.title("Total Reward")

        plt.subplot(242)
        plt.plot(agent._losses, color='orange', linewidth=1)
        plt.title("Loss")

# 結束後關閉視窗
env.close()
