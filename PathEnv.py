import numpy as np
import random
import matplotlib.pyplot as plt

class MazeEnv:
    def __init__(self, size, num_obstacles):
        self.size = size
        self.num_obstacles = num_obstacles

        #生成起點與終點
        self.state_space = np.zeros((size, size))
        self.start = self.generate_random_position()
        self.goal = self.generate_random_position()
        while(self.goal == self.start):
            self.goal = self.generate_random_position()

        self.current_position = self.start
        self.actions = ['up', 'down', 'left', 'right']
        self.state_space[self.start] = 1  # mark the start
        self.state_space[self.goal] = 2  # mark the goal

        #生成障礙物
        self.obstacles = []
        while(len(self.obstacles)<num_obstacles):
            obstacle = self.generate_random_position()
            if(obstacle!=self.start and obstacle!=self.goal and obstacle not in self.obstacles):
                self.obstacles.append(obstacle)

        print(len(self.obstacles))
        for obstacles in self.obstacles:
            self.state_space[obstacles] = -1

    def generate_random_position(self):
        position = np.random.randint(0, self.size), np.random.randint(0, self.size)
        return position    

    # def generate_path(self):
    #     visited = np.zeros((self.size, self.size), dtype=bool)
    #     stack = [self.start]
    #     visited[self.start] = True

    #     while stack:
    #         current = stack[-1]
    #         neighbors = self.get_neighbors(current)
    #         unvisited_neighbors = [neighbor for neighbor in neighbors if not visited[neighbor]]
    #         if unvisited_neighbors:
    #             next_cell = random.choice(unvisited_neighbors)
    #             self.state_space[next_cell] = -1  # mark the path
    #             visited[next_cell] = True
    #             stack.append(next_cell)
    #         else:
    #             stack.pop()

    # def get_neighbors(self, cell):
    #     neighbors = []
    #     for action in self.actions:
    #         neighbor = self.get_next_position(cell, action)
    #         if 0 <= neighbor[0] < self.size and 0 <= neighbor[1] < self.size:
    #             neighbors.append(neighbor)
    #     return neighbors

    def get_next_position(self, current_position, action):
        if action == 'up':
            # return (current_position[0] - 1, current_position[1])
            return max(0, current_position[0] - 1), current_position[1]
        elif action == 'down':
            # return (current_position[0] + 1, current_position[1])
            return min(self.size - 1, current_position+[0]+1), current_position[1]
        elif action == 'left':
            # return (current_position[0], current_position[1] - 1)
            return current_position[0], max(0, current_position[1] - 1)
        elif action == 'right':
            # return (current_position[0], current_position[1] + 1)
            return current_position[0], min(self.size - 1, current_position[1] + 1)
        else:
            return ValueError("Invalid action")

    def reset(self):
        self.current_position = self.start
        return np.copy(self.state_space)

    def step(self, action):
        # Implement step logic
        pass

    def render(self):
        plt.imshow(self.state_space, cmap='gray')
        plt.title('Maze')
        plt.show()

# Example usage
maze_size = 10
num_obstacles = 10
env = MazeEnv(size=maze_size, num_obstacles=num_obstacles)
env.reset()
print(env.state_space)
env.render()

# Now you have a maze with a path from start to goal, and you can add logic to randomly generate other positions.
