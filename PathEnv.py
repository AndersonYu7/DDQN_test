import numpy as np
import random
import matplotlib.pyplot as plt
from checkpath import Checking

class MazeEnv():
    def __init__(self, size, num_obstacles):
        self.size = size
        self.num_obstacles = num_obstacles
        self.episode_ended = False
        self.obstacles = []

        #生成起點與終點
        self.state_space = np.zeros((size, size))
        self.start = self.generate_random_position()
        self.goal = self.generate_random_position()
        while(self.goal == self.start):
            self.goal = self.generate_random_position()

        self.current_position = self.start
        self.action_size = 4
        self.state_space[self.start] = 1  # mark the start
        self.state_space[self.goal] = 2  # mark the goal
        self.generate_random_obstacles()
        

    def generate_random_position(self):
        position = np.random.randint(0, self.size), np.random.randint(0, self.size)
        return position    

    def generate_random_obstacles(self):
        start = self.start[0]*self.size+self.start[1]
        end = self.goal[0]*self.size+self.goal[1]
        graph = np.zeros((self.size, self.size))
        graph[self.start] = 1
        graph[self.goal] = 2
        
        obstacles = np.random.choice(range(self.size*self.size), size=self.num_obstacles, replace=False).tolist()
        for pos in obstacles:
            graph[(int(pos/self.size), int(pos%self.size))] = -1

        while(start in obstacles or end in obstacles or not Checking(graph, self.start, self.goal)):
            obstacles = np.random.choice(range(self.size*self.size), size=self.num_obstacles, replace=False).tolist()
            graph = np.zeros((self.size, self.size))
            graph[self.start] = 1
            graph[self.goal] = 2
            for pos in obstacles:
                graph[(int(pos/self.size), int(pos%self.size))] = -1

        self.obstacles = obstacles
        for pos in obstacles:
            self.state_space[(int(pos/self.size), int(pos%self.size))] = -1

    def get_next_position(self, current_position, action):
        if action == 0: #up
            return max(0, current_position[0] - 1), current_position[1]
        elif action == 1: #down
            return min(self.size - 1, current_position[0]+1), current_position[1]
        elif action == 2: #left
            return current_position[0], max(0, current_position[1] - 1)
        elif action == 3:   #right
            return current_position[0], min(self.size - 1, current_position[1] + 1)
        else:
            return ValueError("Invalid action")

    def hit_reset(self):    #僅僅初始化起終點
        self.state_space = np.zeros((self.size, self.size))
        self.current_position = self.start
        self.state_space[self.start] = 1
        self.state_space[self.goal] = 2
        for pos in self.obstacles:
            self.state_space[(int(pos/self.size), int(pos%self.size))] = -1

        return self.state_space

    def reset(self):       #新的起終點
        # self.state_space = np.zeros((self.size, self.size))
        graph = np.zeros((self.size, self.size))

        #重建障礙物
        for pos in self.obstacles:
            graph[(int(pos/self.size), int(pos%self.size))] = -1

        #新的初始點
        start = self.generate_random_position()
        while(start in self.obstacles):
            start = self.generate_random_position()

        #新的結束點
        goal = self.generate_random_position()
        while(goal in self.obstacles or goal == start):
            goal = self.generate_random_position()

        graph[start] = 1
        graph[goal] = 2

        cnt = 0 #防止搜尋不到 假如搜尋不到 反回原本的圖
        while(not Checking(graph, start, goal)):
            cnt+=1
            if(cnt>20):
                break
            graph[start] = 0
            graph[goal] = 0

            #新的初始點
            start = self.generate_random_position()
            while(start in self.obstacles):
                start = self.generate_random_position()

            #新的結束點
            goal = self.generate_random_position()
            while(goal in self.obstacles or goal == start):
                goal = self.generate_random_position()

            graph[start] = 1
            graph[goal] = 2

        if(cnt>20):
            self.current_position = self.start
            return self.state_space
        else:
            self.current_position = start
            self.state_space = graph
            return graph

    def step(self, action):
        #return state, reward, done, hit

        # Implement step logic
        if self.episode_ended:
            return self.reset(), 0, False

        new_position = self.get_next_position(self.current_position, action)
        # print(new_position)
        # breakpoint()

        if(new_position == self.current_position):
            self._episode_ended = True
            return np.copy(self.state_space), -1, True
        
        elif(new_position in self.obstacles):
            self._episode_ended = True
            return np.copy(self.state_space), -1, True

        elif(new_position == self.goal):
            self._episode_ended = True
            return np.copy(self.state_space), 1, True

        else:
            self._episode_ended = False
            self.state_space[self.current_position] = 0
            self.current_position = new_position
            print(new_position)
            self.state_space[new_position] = 1
            return np.copy(self.state_space), 0, False
        
        

    def render(self, time):
        plt.imshow(self.state_space, cmap='gray')
        plt.title('Maze')
        plt.pause(time)
        plt.clf()

# # Example usage
# maze_size = 10
# num_obstacles = 20
# env = MazeEnv(size=maze_size, num_obstacles=num_obstacles)
# # env.reset()
# # print(env.state_space)
# # print(Checking(env.state_space, env.start, env.goal))
# # env.render()
# # print(env.current_position)
# # env.step('right')
# # env.render()

# env.hit_reset()
# env.render()
# env.hit_reset()
# env.render()

# env.reset()
# env.render()

# Now you have a maze with a path from start to goal, and you can add logic to randomly generate other positions.
