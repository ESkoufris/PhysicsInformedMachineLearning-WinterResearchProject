import torch
from scipy.integrate import quad
import numpy as np
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
        partial_derivative: A one-dimensional tensor containing the partial derivatives
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
        points (torch.tesnor): The points at which the derivative is evaluated
        var_index1 (int): The index of the variable with respect to which the first derivative is evaluated
        var_index2 (int): The index of the variable with respect to which the second derivative is evaluated

    Returns: 
        partial_derivative: A one-dimensional tensor containing the partial derivatives
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
    Base PDE class that stores a PDE in standard form, boundary conditions and initial conditions 

    Attributes:
        pde (func): The partial differential equation, written in standard form 
        ics (func): A function which, when acted upon points at t = 0, returns the initial state of the system 
        bcs (func): A function which, when acted upon points at the boundary of the spatial  domain, 
        returns the initial state of the system 
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
# initial condition functions 
def initial_sine(x):
    """
    Takes in points where t = 0 
    """
    return torch.sin(x)

# boundary condition functions 
def xero(points):
    return torch.zeros(points.shape[0])

# fixed PDE instances 
dirichlet_heat_equation_pde = PDE(heat_equation1D, initial_sine, xero)

# Fourier's solution 
def fourier_heat_eq_solution(x, t, f, L, n_max, c=1):
    """
    Computes the solution of the heat equation using Fourier series.

    Args:
        x (float): The spatial coordinate.
        t (float): The time coordinate.
        f (callable): The initial temperature distribution function.
        L (float): The length of the domain.
        n_max (int): The number of terms in the Fourier series.

    Returns:
        float: The solution of the heat equation at (x, t).
    """
    
    def a_n(n):
        """
        Computes the Fourier coefficient a_n.
        
        Args:
            n (int): The index of the Fourier coefficient.
        
        Returns:
            float: The value of the Fourier coefficient a_n.
        """
        integrand = lambda s: f(s) * np.sin(n * np.pi * s / L)
        result, _ = quad(integrand, 0, L)
        return (2 / L) * result

    solution = torch.zeros_like(x)
    for n in range(1, n_max + 1):
        a_n_val = a_n(n)
        terms = a_n_val * torch.sin(n * np.pi * x / L) * torch.exp(- (n * np.pi * c / L)**2 * t)
        solution += terms
    
    return solution

