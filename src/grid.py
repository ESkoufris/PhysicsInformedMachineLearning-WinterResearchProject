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

    def first_partial_derivative(self, func, var_index):
        """
        Evaluates the first partial derivative of a function at a grid of two-vectors
        Return a grid of partial derivative 
        """
        x = self[...,0]
        t = self[...,1]
        z = func(x,t)
        if var_index == 0:
            z = torch.autograd.grad(outputs=z, 
                                    inputs=x, 
                                    grad_outputs=torch.ones_like(z), 
                                    create_graph=True)[0]
        else:
            z= torch.autograd.grad(outputs=z, 
                                    inputs=t, 
                                    grad_outputs=torch.ones_like(z), 
                                    create_graph=True)[0]
        return [x,t,z]

    def second_partial_derivative(self, func, var_index1, var_index2):
        """
        Evaluates the second partial derivative of a function 
        """
        if var_index1 == 0:
            [x,t,z] = self.first_partial_derivative(func, 0)
        elif var_index1 == 1:
            [x,t,z] = self.first_partial_derivative(func, 1)
        
        if var_index2 == 0:
            return torch.autograd.grad(outputs=z, 
                                    inputs=x, 
                                    grad_outputs=torch.ones_like(z))[0]
        elif var_index2 == 1:
            return torch.autograd.grad(outputs=z, 
                                    inputs=t, 
                                    grad_outputs=torch.ones_like(z))[0]
##################
#  Grid sampling #
##################
def sample(grid: Grid, method = 'uniform'):
    pass

