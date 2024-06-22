import torch 
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

class Grid:
    """
    Base class for grids
    """   
    def __init__(self, grid = None):
        self.grid = None
    
    def __getitem__(self, index):
        return self.grid[index]

    @property
    def shape(self):
        return self.grid.shape

class Grid1D(Grid): 
    """
    Generates a one-dimensional grid of values
    """
    def __init__(self, start, end, step):
        super().__init__()
        self.start = start
        self.end = end
        self.step = step
        self.grid = torch.arange(start=self.start,end=self.end,step=self.step)
    
    def __len__(self):
        return len(self.x_grid)    
    
class Grid2D(Grid):
    """
    Generates a two-dimensional grid of values
    """
    def __init__(self, x_start, x_end, x_step, t_start, t_end, t_step):
        x_length = int((x_end - x_start)/x_step) + 1
        t_length = int((t_end - t_start)/t_step) + 1
        self.x_length = x_length
        self.t_length = t_length

        grid = torch.zeros((x_length, t_length, 2))
        
        for i in range(x_length):
            for j in range(t_length):
                x = x_start + i*x_step
                t = t_start + j*t_step 
                grid[i,j] = torch.tensor([x,t])

        self.grid = grid

    def get_boundary(self):
        """
        Extracts the boundary points of a two-dimensional grid
        """
        left_boundary = self[0,:,:]
        right_boundary = self[-1, :, :]
        return [left_boundary, right_boundary]
    
    def get_initial_state(self):
        return self[:,0,:]

    def get_interior(self):
        """
        Returns the grid less the boundary and initial state tensors  
        """
        return self[1:-2,1:-1,:]

    def plot(self, skip = 100):
        """
        Plots a selected number of points of the grid
        """
        x = self[0:-1:skip,0:-1:skip,0]
        t = self[0:-1:skip,0:-1:skip,1]
        x = x.reshape(-1)
        t = t.reshape(-1)

        plt.scatter(x,t)
        plt.xlabel("x")
        plt.ylabel("t")
        plt.show()

##################
#  Grid sampling #
##################
def sample(grid: Grid, method = 'uniform'):
    pass

g = Grid2D(0,10,0.01,0,10,0.01)
g.plot()


x = torch.zeros((1001,1001,2))

x[:,:,:]

x = torch.zeros((2,2))
x[-1,:] = 1
x


x = torch.tensor([[[1,1],[1,2]],[[2,1],[2,2]]])
y = x[:,:,0]
y.reshape(-1)

z = x[:,:,1]
z.reshape(-1)

x = torch.ones(100)
x[0:-1:100]