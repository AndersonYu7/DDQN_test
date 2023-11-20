import numpy as np
import random
import matplotlib.pyplot as plt

BOARD_SIZE = 5

#action direction
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

#Rewards 
HIT_OBSTACLE = -100
FINISH = 100
OUTOFRANGE = -100
REVISIT_PENALTY = -20
NEW_PATH_REWARD = 2

#Env
OBSTACLE = -1
SPACE = 0
GOAL = 1
START = 2

# 簡單的自定義環境：GridWorld
class GridWorldEnvironment():
    def __init__(self, grid_size=5, num_obstacles=5):
        self.grid_size = grid_size
        self.num_obstacles = num_obstacles
        self.start = self._generate_random_position()
        self.goal = self._generate_random_position()


        self.visited_positions = []
        self._distance_to_goal = np.abs(self.start[0] - self.goal[0]) + np.abs(self.start[1] - self.goal[1])

        # Generate random obstacle coordinates
        self.obstacles = set()
        while len(self.obstacles) < num_obstacles:
            obstacle = (
                random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)
            )
            if obstacle != self.start and obstacle != self.goal:
                self.obstacles.add(obstacle)

        self.current_position = self.start
        self.episode_ended = False

    def _generate_random_position(self):
        position = random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)
        return position

    def reset(self):
        self.start = self._generate_random_position()
        self.goal = self._generate_random_position()

        self.visited_positions = []

        # Generate random obstacle coordinates
        self.obstacles = set()
        while len(self.obstacles) < self.num_obstacles:
            obstacle = (
                random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)
            )
            if obstacle != self.start and obstacle != self.goal:
                self.obstacles.add(obstacle)

        while self.start in self.obstacles or self.goal in self.obstacles:
            self.start = self._generate_random_position()
            self.goal = self._generate_random_position()

        self.current_position = self.start
        self.episode_ended = False
        return self.current_observation()

    def current_observation(self):
        observation = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        observation[tuple(self.current_position)] = 1  # Assuming 1 represents the current position
        for obstacle_pos in self.obstacles:
            observation[obstacle_pos] = -1  # Assuming -1 represents obstacles
        observation[self.goal] = 2  # Assuming 2 represents the goal
        return observation

    def render(self, time):
        plt.figure(2)
        plt.title('GridWorld Environment')
        plt.imshow(self.current_observation(), cmap='gray')
        # plt.show(block=False)
        plt.pause(time)
        plt.clf()

    def step(self, action):
        if self.episode_ended:
            return self.reset(), 0, False  # If the episode has ended, reset the environment

        new_position = np.copy(self.current_position)

        if action == 0:  # Move Up
            new_position[0] = max(0, new_position[0] - 1)
        elif action == 1:  # Move Down
            new_position[0] = min(self.grid_size - 1, new_position[0] + 1)
        elif action == 2:  # Move Left
            new_position[1] = max(0, new_position[1] - 1)
        elif action == 3:  # Move Right
            new_position[1] = min(self.grid_size - 1, new_position[1] + 1)
        else:
            raise ValueError("Invalid action")

        # self.current_position = new_position


        if(tuple(new_position)[0] == self.current_position[0] and tuple(new_position)[1] == self.current_position[1]):
            self._episode_ended = True
            return self.current_observation(), OUTOFRANGE, True

        elif tuple(new_position) in map(tuple, self.obstacles):
            self._episode_ended = True
            return self.current_observation(), HIT_OBSTACLE, True

        elif np.array_equal(new_position, self.goal):
            self._episode_ended = True
            return self.current_observation(), FINISH, True
        else:
            
            distance_togoal = np.sqrt((new_position[0] - self.goal[0])**2 + (new_position[1] - self.goal[1])**2)
            reward = 0

            if distance_togoal < int(self._distance_to_goal/2):
                reward = (self.grid_size - distance_togoal)*2+5
            elif tuple(new_position) in self.visited_positions:
                # 如果新位置已经被访问过，扣分
                reward += REVISIT_PENALTY
            else:
                # 如果新位置是新的，将其添加到visited_positions列表中
                self.visited_positions.append(tuple(new_position))

            self.current_position = new_position
            self._distance_to_goal = distance_togoal


            return self.current_observation(), reward, False

    def get_max_steps(self):
        return self.grid_size * self.grid_size * 2


# # 測試環境
# env = GridWorldEnvironment(grid_size=5, num_obstacles=5)

# # 輸出初始狀態
# print("Initial State:")
# env.render()

# # 走幾步
# R = 0
# for _ in range(5):
#     action = random.randint(0, 3)
#     observation, reward, done = env.step(action)
#     R+=reward
#     print(f"Action: {action}, Reward: {reward}")
#     env.render()

# print(R/5)
# # 重置環境
# print("Resetting Environment:")
# env.reset()
# env.render()
