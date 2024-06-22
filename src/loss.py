def total_loss(pde, grid):
    pass

def physics_loss(pde, data):
    """
    Computes the PDE residual of a PINN
    """
    pass

def boundary_loss(pde, data, bcs):
    """
    Computes the loss of the PINN at the boundary of a domain
    """

    return MSE(pde(data) - bcs(data))

def initial_loss(pde,data):
    """
    Compute the loss of the PINN at the initial state  
    """
    pass

