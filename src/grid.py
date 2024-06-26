import numpy as np
import torch 
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

class Grid:
    """
    Base class for grids
    """   
    def __init__(self, grid = None):
        self.grid = None
    
    def __getitem__(self, index):
        return self.grid[index]
    
    def reshape(self,i):
        return self.grid.reshpape(i)  

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
        self.x_end = x_end
        self.t_end = t_end

        nx_steps = int((x_end - x_start)/x_step)
        nt_steps = int((t_end - t_start)/t_step)

        x = np.linspace(x_start, x_end, nx_steps)
        t = np.linspace(t_start, t_end, nt_steps)
 
        
        grid = torch.tensor(np.meshgrid(x,t), dtype=torch.float32).permute(2,1,0)

        self.grid = grid

    def get_boundary(self):
        """
        Extracts the boundary points of a two-dimensional grid
        """
        return self[[0,-1],...]
    
    def get_initial_state(self):
        return self[:,0,:]
    
    def get_interior(self):
        """
        Returns the grid less the boundary and initial state tensors  
        """
        return self[1:-1,1:,:]

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

# g = Grid2D(0,2*np.pi, 0.01, 0, 10, 0.01)
# gr = g.grid

# gr[[0,-1],...]

# gr[:,0,:]

# gr[1:-1, 1:,:]