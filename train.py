import numpy as np
import random
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras.optimizers import Adam

from PathEnv import MazeEnv

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Conv2D(128, (3, 3), input_shape=(self.state_size, self.state_size, 1), activation='relu'))
        model.add(Conv2D(256, (3, 3), activation='relu'))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def preprocess_state(self, state):
        # Preprocess the state if necessary
        return state.reshape((1, self.state_size, self.state_size, 1))

    def remember(self, state, action, reward, next_state, done):
        state = self.preprocess_state(state)
        next_state = self.preprocess_state(next_state)
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = self.preprocess_state(state)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                a = np.argmax(self.model.predict(next_state)[0])
                target[0][action] = reward + self.gamma * (self.model.predict(next_state)[0][a])
            self.model.fit(state, target, epochs=1, verbose=0)

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    def get_state_size(self):
        return self.state_size

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


# Example of using the agent with your MazeEnv environment
env = MazeEnv(10, 20)
state_size = env.size  # Assumes it's a simple numeric state
action_size = env.action_size

agent = DQNAgent(state_size, action_size)
batch_size = 32
EPISODES = 1000
episodes_step = agent.get_state_size()*agent.get_state_size()*2

for episode in range(EPISODES):
    state = env.hit_reset()
    total_reward = 0
    step = 0

    for time in range(episodes_step):
        step+=1

        env.render(time = 0.0001)

        action = agent.act(state)
        next_state, reward, done = env.step(action)

        reward = reward if not done else -10

        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward+=reward

        if done:
            print("Episode: {}/{}, Total Reward: {}, Epsilon: {:.2}".format(episode, EPISODES, total_reward, agent.epsilon))
            break

    if len(agent.memory) > batch_size:
        agent.replay(batch_size)

    if episode%50 ==0 and episode!=0:
        save_model(agent, episode)

        print(action)
        breakpoint()


