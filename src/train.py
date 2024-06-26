from src.pinn import *
from src.pde import *
from src.loss import *
from src.grid import *
import torch
from torch.utils.data import Dataset, DataLoader
from torch import optim
import tqdm 
from time import time
from tqdm import tqdm

class MultiTensorDataset(Dataset):
    def __init__(self, *tensors):
        self.tensors = tensors

    def __len__(self):
        # Return the length of the smallest tensor to avoid index out of range issues
        return min(len(tensor) for tensor in self.tensors)

    def __getitem__(self, index):
        # Return a tuple containing elements from each tensor at the given index
        return tuple(tensor[index] for tensor in self.tensors)

def custom_collate_fn(batch):
    # Assuming each element in the batch is a tuple of tensors
    batch_tensors = list(zip(*batch))
    
    # Pad or process tensors as needed
    # For simplicity, we'll just convert the list of tuples to a list of tensors
    return [torch.stack(tensors) for tensors in batch_tensors]

#####################
# Training function #
#####################
def train(pinn: PINN, pde: PDE, grid, lr=0.001, nepochs=100, batch_size=4):
    """
    Training function for a PINN
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    losses = []
    bdry_losses = []
    init_losses = []
    int_losses = []
    epoch_losses = []
    pinn.to(device)

    # set optimizer
    optimizer = optim.Adam(pinn.parameters(), lr=lr)

    # seperate boundary and initial points from interior points 
    bdry_pts = grid[[0,-1],...]
    init_pts = grid[:,0,:].unsqueeze(0)
    int_pts = grid[1:-1, 1:,:]

    bdry_pts = bdry_pts.reshape(-1,bdry_pts.shape[2])
    bdry_pts = bdry_pts[torch.randperm(bdry_pts.shape[0])]
    init_pts = init_pts.reshape(-1,init_pts.shape[2])
    int_pts = int_pts.reshape(-1,int_pts.shape[2])
    
    dataset = MultiTensorDataset(bdry_pts, init_pts, int_pts)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=custom_collate_fn, shuffle=True)
    
    loop = tqdm(range(nepochs), ncols=110)

    ics = pde.ics
    bcs = pde.bcs 
    pde = pde.pde 

    for i in loop:
        t0 = time()
        epoch_loss = 0
        n_batches = 0 
        for bdry_batch, init_batch, int_batch  in dataloader:
            n_batches += 1
            
            # move batch to device    
            bdry_batch = bdry_batch.to(device)
            init_batch = init_batch.to(device)
            int_batch = int_batch.to(device)

            # reset gradient 
            optimizer.zero_grad()

            # compute losses 
            Lb = boundary_loss(pinn, bdry_batch, bcs)
            Lo = initial_loss(pinn, init_batch, ics)
            Lp = physics_loss(pinn, int_batch, pde)
            loss = Lb + Lo + Lp
            
            # saving loss
            epoch_loss += loss.item()
            bdry_losses.append(Lb.item())
            init_losses.append(Lo.item())
            int_losses.append(Lp.item())
            losses.append(loss.item())

            # gradient descent
            loss.backward()
            optimizer.step()
        
        epoch_losses.append(epoch_loss)  
        epoch_loss /= n_batches
        loop.set_postfix(loss="%5.5f" % (epoch_loss)) 

    return epoch_losses, losses 

####################
# Testing function #
####################
def test(pinn: PINN, pde: PDE, grid, batch_size=4):
    bdry_pts = grid[[0,-1],...]
    init_pts = grid[:,0,:].unsqueeze(0)
    int_pts = grid[1:-1, 1:,:]

    ics = pde.ics
    bcs = pde.bcs 
    pde = pde.pde 

    bdry_pts = bdry_pts.reshape(-1,bdry_pts.shape[2])
    init_pts = init_pts.reshape(-1,init_pts.shape[2])
    int_pts = int_pts.reshape(-1,int_pts.shape[2])

    Lb = boundary_loss(pinn, bdry_pts, bcs)
    Lo = initial_loss(pinn, init_pts, ics)
    Lp = physics_loss(pinn, int_pts, pde)

    return [Lb, Lo, Lp]

    