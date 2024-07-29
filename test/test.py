from src.grid import *
from src.loss import *
from src.pinn import *
from src.pde import *
from src.train import *


grid = Grid2D(0,1,0.01,0,1,0.01)

bdry_pts = grid[[0,-1],...]
init_pts = grid[:,0,:].unsqueeze(0)
int_pts = grid[1:-1, 1:,:]

# initialise linear coefficients 
ratios = 3*torch.rand(2,100) + 1

bdry_pts = bdry_pts.reshape(-1,bdry_pts.shape[2])
# randomly arrange boundary points
bdry_pts = bdry_pts[torch.randperm(bdry_pts.shape[0])]
init_pts = init_pts.reshape(-1,init_pts.shape[2])
int_pts = int_pts.reshape(-1,int_pts.shape[2])   

# concatenate with ratios 
bdry_pts = concat_ratios(bdry_pts, ratios)
init_pts = concat_ratios(init_pts,ratios)
int_pts = concat_ratios(int_pts,ratios)

pinn = pinn1D()
pinn


pinn(x,t,ratios)