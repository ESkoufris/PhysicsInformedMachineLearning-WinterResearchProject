import torch
from torch.nn.functional import mse_loss

def physics_loss(u, points, pde):
    """
    Computes the PDE residual of a PINN

    Args: 
        u (function): A function that acts upon points of the form (x,t), where x is a vector 
        points (torch.tensor): Points at which the PDE is evaluated 
        pde (function): PDE residual

    Returns: 
        torch.float: Average squared mean PDE residual of the function u 
    """
    # needs to compute difference from 0
    return mse_loss(pde(u, points), torch.zeros(points.shape[0]))

def boundary_loss(u, points, bcs = None):
    """
    Computes the loss of the PINN at the boundary of a domain
    """
    x = points[...,0].flatten()
    t = points[...,1].flatten()
    ratios = points[...,2].flatten()
    return mse_loss(u(x, t, ratios), bcs(points))

def initial_loss(u, points, ics):
    """
    Compute the loss of the PINN at the initial state  
    """
    x = points[...,0].flatten()
    t = points[...,1].flatten()
    ratios = points[...,2].flatten()
    return mse_loss(u(x, t, ratios), ics(x))

