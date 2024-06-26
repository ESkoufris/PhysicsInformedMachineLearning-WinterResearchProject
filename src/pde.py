import torch
from src.pinn import *

########################
# Derivative functions #
########################

def first_partial_derivative(func, points, var_index, retain_graph = True):
    """
    Evaluates the first partial derivative of a real function with domain R^2 at a set of points
    
    Args: 
        func: The function whose derivative is evaluated
        points: The points at which the derivative is evaluated
        var_index: The index of the variable with respect to which the derivative is evaluated

    Returns: 
        partial_derivative: The derivatives 
    """
    points.requires_grad_(True)
    x = points[...,0].flatten()
    t = points[...,1].flatten()

    z = func(x,t)

    if var_index == 0:
        grad_var = x
    else:
        grad_var = t
    
    # Compute the gradient
    partial_derivative = torch.autograd.grad(outputs=z, 
                                             inputs=grad_var, 
                                             grad_outputs=torch.ones_like(z), 
                                             create_graph=retain_graph,
                                             retain_graph=retain_graph,
                                             allow_unused=True)[0]

    return [x, t, partial_derivative]



def second_partial_derivative(func, points, var_index1, var_index2):
    """
    Evaluates the second partial derivative of a function 

    Args: 
        func: The function whose derivative is evaluated
        points: The points at which the derivative is evaluated
        var_index: The index of the variable with respect to which the derivative is evaluated

    Returns: 
        partial_derivative: The derivatives 
    """
    if var_index1 == 0:
       [x,t,z] = first_partial_derivative(func, points, 0)
    elif var_index1 == 1:
       [x,t,z] = first_partial_derivative(func, points, 1)
    
    if var_index2 == 0:
        return torch.autograd.grad(outputs=z, 
                                   inputs=x, 
                                   grad_outputs=torch.ones_like(z))[0]
    elif var_index2 == 1:
        return torch.autograd.grad(outputs=z, 
                                   inputs=t, 
                                   grad_outputs=torch.ones_like(z))[0]

#############
# PDE class #
#############
class PDE:
    """
    Base PDE class that stores PDE, boundary conditions and initial conditions 

    Attributes:
        pde (func): the partial differential equation
        ics (func):
        bcs (func): 
    """
    def __init__(self, pde, ics, bcs):
        self.pde = pde
        self.ics = ics
        self.bcs = bcs

def boundary_conditions_to_function(bcs):
    """
    Converts the boundary conditions to a function that acts on points

    Args: 
        bcs (torch.tensor): boundary conditions of a 
    """
    pass

def initial_conditions_to_function(ics):
    """
    Converts the initial conditions to a function that acts on points
    """
    pass
    
def basic_ics(points):
    pass
################
# Example PDEs #
################

def heat_equation1D(u, points, c=1):
    """
    One-dimensional heat equation PDE residual
    """
    return first_partial_derivative(u,points,1)[-1] - (c**2)*second_partial_derivative(u,points,0,0)

def wave_equation1D(u, t, x, c=1):
    """
    One-dimensional wave equation PDE residual
    """
    return second_partial_derivative(u,t,x,1,1) - (c**2)*second_partial_derivative(u,t,x,0,0)


####################
# Useful functions #
####################

def initial_sine(x):
    """
    Takes in points where t = 0 
    """
    return torch.sin(x)

# boundary condition functions 
def xero(points):
    return torch.zeros(points.shape[0])

dirichlet_heat_equation_pde = PDE(heat_equation1D, initial_sine, xero)