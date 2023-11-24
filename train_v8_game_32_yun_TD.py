import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #因為Tensorflow 須從源代碼安裝 可能安裝方法不對 導致使用CPU 用這行可以消除警告

import numpy as np
import time
import random
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import math
# from tensorflow.keras.callbacks import TensorBoard

from PathEnv_v8_game_32_yun import MazeEnv

tf.keras.utils.disable_interactive_logging()

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# breakpoint()
# tf.debugging.set_log_device_placement(True)


class DQNAgent:
    def __init__(self, state_size, action_size, alpha=0.6, beta=0.4, beta_increment_per_sampling=0.001, epsilon_alpha=1e-6):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.05
        self.epilon_max = 1.0
        self.epsilon_decay = 0.998
        self.epilon_max_decay = 0.95
        self.learning_rate = 0.001
        self.update_target_frequency = 10
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()  # 新增：初始化時就更新目標網路

        self.loss_history = []
        self.total_rewards = []
        self.rewards_stable = deque(maxlen=5)

        # 優先順序回放參數
        self.alpha = alpha
        self.beta = beta
        self.beta_increment_per_sampling = beta_increment_per_sampling
        self.epsilon_alpha = epsilon_alpha

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
        # 計算 TD 誤差，並使用優先順序回放的重要性值（priority）初始化記憶體中的樣本
        td_error = self.compute_td_error(state, action, reward, next_state, done)
        priority = (np.abs(td_error) + self.epsilon_alpha) ** self.alpha    #防止優先順序值為零

        self.memory.append((state, action, reward, next_state, done, priority))

    def sample_batch(self, batch_size):
        # 依據樣本的優先順序進行抽樣
        priorities = [sample[-1] for sample in self.memory]
        probs = priorities / (np.sum(priorities) + self.epsilon_alpha)  #將優先順序值轉換為抽樣的概率分布 -> 樣本的概率分布就與它們的優先順序成比例
        indices = np.random.choice(len(self.memory), size=batch_size, p=probs, replace=True)    #抽取的樣本的索引
        batch = [self.memory[i] for i in indices]

        # 計算抽樣的最大優先順序，用於更新權重
        max_priority = np.max(priorities)
        # 更新 beta 參數，使其漸進地增加
        self.beta = np.min([1.0, self.beta + self.beta_increment_per_sampling])

        # 返回抽樣的樣本、抽樣的樣本索引、最大優先順序和 beta 參數
        return batch, indices, max_priority, self.beta

    def compute_td_error(self, state, action, reward, next_state, done):
        state = np.reshape(state, (1, self.state_size, self.state_size, 1))
        next_state = np.reshape(next_state, (1, self.state_size, self.state_size, 1))
        target = self.target_model.predict(state)[0]

        if done:
            target[action] = reward
        else:
            a = np.argmax(self.target_model.predict(next_state)[0])
            target[action] = reward + self.gamma * (self.target_model.predict(next_state)[0][a])

        return target[action] - self.model.predict(state)[0][action]

    def update_memory_priority(self, indices, errors):
        for i, error in zip(indices, errors):
            priority = (np.abs(error) + self.epsilon_alpha) ** self.alpha
            weighted_priority = priority ** self.beta
            self.memory[i] = self.memory[i][:5] + (weighted_priority,)  #priority

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = self.preprocess_state(state)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])
    
    def replay(self, batch_size, episode):
        batch, indices, max_priority, beta = self.sample_batch(batch_size)

        states, actions, rewards, next_states, dones, priorities = zip(*batch)

        td_errors = []
        for i in range(batch_size):
            state = self.preprocess_state(states[i])
            next_state = self.preprocess_state(next_states[i])
            target = self.target_model.predict(state)[0]

            if dones[i]:
                target[actions[i]] = rewards[i]
            else:
                a = np.argmax(self.target_model.predict(next_state)[0])
                target[actions[i]] = rewards[i] + self.gamma * (self.target_model.predict(next_state)[0][a])

            td_errors.append(target[actions[i]] - self.model.predict(state)[0][actions[i]])

            self.model.fit(state, target.reshape(-1, self.action_size), epochs=1, verbose=0)

        # 更新樣本的優先順序
        self.update_memory_priority(indices, td_errors)

        # 更新回放網絡
        if episode % self.update_target_frequency == 0:
            self.update_target_model()

        # 調整探索率
        self.epsilon = self.epsilon + 0.001 - 0.99 * math.exp(-episode / 1000)


    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def plot(self, graph, title_, ylabel, num = 2, area = 1):
        plt.figure(num)
        plt.subplot(area+120)
        plt.plot(graph, label='loss')
        plt.title(title_)
        plt.xlabel('Epoch')
        plt.ylabel(ylabel)
        # plt.clf()

            
def save_model(agent, episode, model_dir='models'):
    try:
        print('Model saving')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_path = os.path.join(model_dir, f'v8_game_model_{episode}.keras')  # 加入情节编号
        agent.model.save(model_path)
        print(f'Model saved to {model_path}')
    except Exception as e:
        print(f"Error saving model: {e}")


def are_all_elements_equal(my_deque):
    # 如果deque是空的，直接返回True
    # print('len: ', len(my_deque))
    if(len(my_deque)!=5) or not my_deque:
        return False

    # if not my_deque:
    #     return False
    
    # 将deque的第一个元素作为参考值
    reference_value = my_deque[0]

    # 遍历deque中的每个元素，检查是否与参考值相同
    for value in my_deque:
        if value != reference_value:
            return False
    
    # 如果遍历完成，所有元素都相同
    return True

if __name__ == '__main__':
    state_size = 32
    num_obstacles = 20
    action_size = 4
    env = MazeEnv(state_size, num_obstacles)

    agent = DQNAgent(state_size, action_size)
    batch_size = 64
    EPISODES = 1000
    episodes_max_step = env.get_max_steps()

    done = False
    # cnt = 0
    cnt2 = 0

    for episode in range(EPISODES):
        state = env.game_reset()
        # print('cnt: ', cnt)
        print('cnt2: ', cnt2)
        are_all_deqeue = are_all_elements_equal(agent.rewards_stable)

        # if(done):
        #     state = env.game_reset()

        total_reward = 0
        step = 0

        for time in trange(episodes_max_step):
            step+=1

            action = agent.act(state)
            next_state, reward, done = env.step(action)
            # reward = reward if not done else -10
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward+=reward

            env.render(time = 0.000000001)

            if done:
                break
        print("Episode: {}/{}, Total Reward: {}, Epsilon: {:.2}, step: {}".format(episode, EPISODES, total_reward, agent.epsilon, step))

        agent.total_rewards.append(total_reward)
        agent.rewards_stable.append(total_reward)

        agent.plot(agent.loss_history, 'Training Loss', 'loss')
        agent.plot(agent.total_rewards,  'Total reward', 'reward', area = 2)

        if len(agent.memory) > batch_size:
            agent.replay(batch_size, episode)

            # 新增：儲存每次訓練的 loss
            loss = agent.model.history.history['loss']
            agent.loss_history.append(loss)

        if episode%50 ==0 and episode!=0:
            save_model(agent, episode)
            plt.savefig(f'v8_game_loss_reward_plot_episode_{episode}.png')  # Save the reward plot

        if episode % agent.update_target_frequency == 0:
            agent.update_target_model()  # 新增：定期更新目標網路

        tf.keras.backend.clear_session()
    
