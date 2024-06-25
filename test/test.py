from src.grid import *
from src.loss import *
from src.pinn import *
from src.pde import *
from src.train import *

grid = Grid2D(0,1,0.01,0,1,0.01)

bdry_pts = grid[torch.logical_or(grid[...,0] == 0, grid[...,0] == 1)]
init_pts = grid[grid[...,1] == 0]
int_pts = grid[torch.logical_not(torch.logical_or(torch.logical_or(grid[...,0] == 0, grid[...,0] == 1), grid[...,1] == 0))]

dataset = MultiTensorDataset(bdry_pts, init_pts, int_pts)
dataloader = DataLoader(dataset, batch_size=4, collate_fn=custom_collate_fn, shuffle=True)
pinn = pinn1D()

for bdry_batch, init_batch, int_batch  in dataloader:
    print(int_batch)