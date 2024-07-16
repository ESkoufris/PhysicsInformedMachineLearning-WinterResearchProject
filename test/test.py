from src.grid import *
from src.loss import *
from src.pinn import *
from src.pde import *
from src.train import *

grid = Grid2D(0,1,0.01,0,1,0.01)

bdry_pts = grid[torch.logical_or(grid[...,0] == 0, grid[...,0] == 1)]
init_pts = grid[grid[...,1] == 0]
int_pts = grid[torch.logical_not(torch.logical_or(torch.logical_or(grid[...,0] == 0, grid[...,0] == 1), grid[...,1] == 0))]

layer = nn.Linear(3,100)

x = int_pts[...,0].flatten()
t = bdry_pts[...,1].flatten()

x = x[:5].repeat_interleave(10)
t = t[:5].repeat_interleave(10)
ratios = torch.rand(10*5)

heat_equation1D(x,t,ratios)


grid = Grid2D(0,1,0.01,0,1,0.01)

bdry_pts = grid[[0,-1],...]
init_pts = grid[:,0,:].unsqueeze(0)
int_pts = grid[1:-1, 1:,:]

# initialise linear coefficients 
ratios = 3*torch.rand(100) + 1

bdry_pts = bdry_pts.reshape(-1,bdry_pts.shape[2])
bdry_pts = bdry_pts[torch.randperm(bdry_pts.shape[0])]
init_pts = init_pts.reshape(-1,init_pts.shape[2])
int_pts = int_pts.reshape(-1,int_pts.shape[2])   

pts = concat_ratios(bdry_pts,ratios)
x = pts[...,0].flatten()
t = pts[...,0].flatten()
ratios = pts[...,0].flatten()

pinn = pinn1D()
pinn


pinn(x,t,ratios)