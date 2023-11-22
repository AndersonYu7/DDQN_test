import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import numpy as np
from tensorflow.keras.models import load_model
from PathEnv_v5 import MazeEnv

# from DQN_env import GridWorldEnvironment


def test_model(model_path, num_episodes=100):
    maze_size = 10
    num_obstacles = 20
    env = MazeEnv(size=maze_size, num_obstacles=num_obstacles)
    model = load_model(model_path)
    
    for episode in range(num_episodes):
        state = env.hit_reset()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            action = np.argmax(model.predict(np.array([state])))
            # print(model.predict(np.array([state])))
            # breakpoint()
            next_state, reward, done = env.step(action)
            total_reward += reward
            steps += 1
            state = next_state

            # 可选：如果要显示环境，取消注释下面的行
            env.render()

        print(f"Episode: {episode + 1}, Total reward: {total_reward}, Steps: {steps}")

if __name__ == "__main__":
    model_path = './models/model_300.keras'  # 更改为您的模型路径
    test_model(model_path)
