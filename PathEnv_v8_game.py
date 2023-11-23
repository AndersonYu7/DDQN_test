import numpy as np
import random
import matplotlib.pyplot as plt
import math
from checkpath import Checking

class MazeEnv():
    def __init__(self, size, num_obstacles):    #起終點及障礙物位置
        self.size = size
        self.num_obstacles = num_obstacles
        self.episode_ended = False
        self.obstacles = [] #為一個值

        #生成起點與終點
        # self.start = self.generate_random_position()
        # self.current_position = self.start
        # self.goal = self.generate_random_position()
        # while(self.goal == self.start):
        #     self.goal = self.generate_random_position()

        #生成障礙物 DFS 一定有一條路徑連通起終點
        # self.generate_random_obstacles()
        self.generate_box_enviroment()
        self.generate_start_end_pos()
        self.current_position = self.start

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
        
        for pos in obstacles:    
            self.obstacles.append((int(pos/self.size), int(pos%self.size)))

    
    def generate_start_end_pos(self):
        self.start = self.generate_random_position()
        while(self.start in self.obstacles):
            self.start = self.generate_random_position()

        self.goal = self.generate_random_position()
        while(self.goal in self.obstacles or self.start == self.goal):
            self.goal = self.generate_random_position()

        

    def generate_box_enviroment(self):
        for x in range(14):
            for y in range(64):
                self.obstacles.append((x, y))

        for x in range(15):
            for y in range(64):
                self.obstacles.append((x+49, y))
        
        
        box_pos = [19, 7]     #(20, 8)
        for box_row in range(2):
            for num in range(4):
                for x in range(9):
                    for y in range(7):
                        self.obstacles.append(tuple(box_pos))
                        box_pos[1]+=1
                    
                    box_pos[0]+=1
                    box_pos[1]-=7

                box_pos[0]-=9
                box_pos[1]+=14

            box_pos[0] = 34
            box_pos[1] = 7

        # print(self.obstacles)

        

    
    def current_observation(self):
        observation = np.zeros((self.size, self.size))

        for obstacle_pos in self.obstacles:
            # observation[(int(obstacle_pos/self.size), int(obstacle_pos%self.size))] = -1  # Assuming -1 represents obstacles
            observation[obstacle_pos] = -1
        observation[self.current_position] = 1  # Assuming 1 represents the current position
        observation[self.goal] = 2  # Assuming 2 represents the goal
        return observation

    def render(self, time = 0.0001, mode = 'train'):
        graph = self.current_observation()

        plt.figure(1)
        plt.imshow(graph, cmap='gray')
        plt.title('Maze')
        if(mode == "human"):
            plt.show()
        elif(mode == "train"):
            plt.pause(time)
            plt.clf()
        else:
            ValueError("Invalid action")

    def hit_reset(self):    #僅初始化起終點
        self.current_position = self.start
        self.episode_ended = False
        return self.current_observation()
    
    def new_start_reset(self):  #same obstacles new start and goal, and must have a path 
        self.episode_ended = False
        original_start = self.start
        original_goal = self.goal
        
        #新的初始點
        start = self.generate_random_position()
        while(start in self.obstacles):
            start = self.generate_random_position()

        #新的結束點
        goal = self.generate_random_position()
        while(goal in self.obstacles or goal == start):
            goal = self.generate_random_position()

        self.current_position = start
        self.goal = goal
        graph = self.current_observation()

        cnt = 0
        while(not Checking(graph, start, goal)):
            cnt+=1
            if(cnt>20):
                break

            #新的初始點
            start = self.generate_random_position()
            while(start in self.obstacles):
                start = self.generate_random_position()

            #新的結束點
            goal = self.generate_random_position()
            while(goal in self.obstacles or goal == start):
                goal = self.generate_random_position()

            self.current_position = start
            self.goal = goal
            graph = self.current_observation()

        if(cnt>20): #start <-> goal
            self.current_position = original_goal
            self.goal = original_start
            return self.current_observation()
        else:
            self.current_position = start
            self.goal = goal
            graph = self.current_observation()
            return graph

    def all_reset(self):    #全部重新生成
        #生成起點與終點
        self.start = self.generate_random_position()
        self.current_position = self.start
        self.goal = self.generate_random_position()
        while(self.goal == self.start):
            self.goal = self.generate_random_position()

        #生成障礙物 DFS 一定有一條路徑連通起終點
        self.obstacles = []
        self.generate_random_obstacles()

        return self.current_observation()

    def game_reset(self):
        self.generate_start_end_pos()


    def step(self, action):
        if self.episode_ended:
            return self.hit_reset(), 0, False  # If the episode has ended, reset the environment
        
        new_position = np.copy(self.current_position)

        if action == 0:  # Move Up
            new_position[0] = max(20, new_position[0] - 1)
        elif action == 1:  # Move Down
            new_position[0] = min(49, new_position[0] + 1)
        elif action == 2:  # Move Left
            new_position[1] = max(0, new_position[1] - 1)
        elif action == 3:  # Move Right
            new_position[1] = min(49-1, new_position[1] + 1)
        else:
            return ValueError("Invalid action")
        
        new_position = tuple(new_position)
        
        if(self.current_position == new_position):
            # self._episode_ended = True
            return self.current_observation(), -1, False
        
        elif(new_position in self.obstacles):
            # self._episode_ended = True
            self.current_position = new_position
            return self.current_observation(), -1, False
        
        elif(new_position == self.goal):
            self._episode_ended = True
            self.current_position = new_position
            distance = np.linalg.norm(np.array(self.start) - np.array(self.goal))
            # return self.current_observation(), 5, True
            return self.current_observation(), distance*0.5, True
        else:
            self.current_position = new_position

            distance_reward = 1 / (1 + np.linalg.norm(np.array(new_position) - np.array(self.goal)))

            return self.current_observation(), -0.5 + distance_reward, False

    def get_max_steps(self):
        return (self.size * self.size - 64*28)*2


# Example usage
# maze_size = 64
# num_obstacles = 20
# env = MazeEnv(size=maze_size, num_obstacles=num_obstacles)
# env.render(mode = "human")
# env.step(2)
# env.hit_reset()
# env.render(mode = "human")
# env.game_reset()
# env.render(mode = "human")

# env.new_start_reset()
# # print(env.current_position, env.goal)
# # print(env.current_observation())
# env.render(mode = "human")

# env.all_reset()
# env.render(mode = "human")

# for action in range(4):
#     env.step(1)
#     env.render(show=True)

# env.current_position = env.goal
# env.step(1)
# env.render(show=True)
# env.step(1)