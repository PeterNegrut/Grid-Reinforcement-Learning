import numpy as np

class Environment: 
    def __init__(self, grid_size=(5, 5), start=(0, 0), goal=(4, 4)):
        self.action_space = type('', (), {'n': 4, 'sample': lambda: np.random.randint(0, 4)})()
        self.grid_size = grid_size
        self.start = start
        self.goal = goal
        self.state = start
    
    def reset(self):
        self.state = self.start
        return self.state
    
    def step(self, action):
        self.action = action
        self.x = self.state[0]
        self.y = self.state[1]
        if action == 0: # up
            if self.y > 0:
                next_state = (self.x, self.y-1)
            else:
                next_state = (self.x, self.y)
        
        elif action == 2: # down
            if self.y < self.grid_size[1] - 1:
                next_state = (self.x, self.y+1)
            else:
                next_state = (self.x, self.y)
        
        elif action == 3: # left
            if self.x > 0:
                next_state = (self.x-1, self.y)
            else:
                next_state = (self.x, self.y)

        elif action == 1: # right
            if self.x < self.grid_size[0] - 1:
                next_state = (self.x+1, self.y)
            else:
                next_state = (self.x, self.y)

        self.state = next_state

        if self.state == (4, 4):
            reward = 10
            Terminated = True
            Truncated = False
        else:
            reward = -0.01
            Terminated = False
            Truncated = False

        info = {}


        return self.state, reward, Terminated, Truncated, info

    def render(self):
        # Create a simple grid visualization
        grid = [['.' for _ in range(self.grid_size[1])] for _ in range(self.grid_size[0])]
        
        # Mark goal
        grid[self.goal[0]][self.goal[1]] = 'G'
        
        # Mark agent
        grid[self.state[0]][self.state[1]] = 'A'
        
        # Print the grid
        for row in grid:
            print(' '.join(row))
        print()