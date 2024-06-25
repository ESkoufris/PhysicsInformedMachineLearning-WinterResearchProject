import torch
from torch.nn.functional import mse_loss

def physics_loss(u, points, pde):
    """
    Computes the PDE residual of a PINN

    Args: 

    Returns: 
    """
    # needs to compute difference from 0
    return mse_loss(pde(u, points), torch.zeros(len(points)))

def boundary_loss(u, points, bcs = None):
    """
    Computes the loss of the PINN at the boundary of a domain
    """
    x = points[...,0]
    t = points[...,1]
    return mse_loss(u(x,t), bcs(points))

def initial_loss(u, points, ics):
    """
    Compute the loss of the PINN at the initial state  
    """
    x = points[...,0]
    t = points[...,1]
    return mse_loss(u(x,t), ics(points))

