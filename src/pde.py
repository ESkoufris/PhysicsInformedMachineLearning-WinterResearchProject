import torch

########################
# Derivative functions #
########################
def first_partial_derivative(func, x_val, y_val, var_index):
    """
    Evaluates the first partial derivative of a function 
    """
    x = torch.tensor(x_val, dtype=torch.float32, requires_grad=True)
    y = torch.tensor(y_val, dtype=torch.float32, requires_grad=True)
    output = func(x,y)
    first_derivative = torch.autograd.grad(outputs=output, inputs=[x, y])[var_index]
    return first_derivative.item()

def second_partial_derivative(func, x_val, y_val, var_index1, var_index2):
    """
    Evaluates the second partial derivative of a function 
    """
    x = torch.tensor(x_val, dtype=torch.float32, requires_grad=True)
    y = torch.tensor(y_val, dtype=torch.float32, requires_grad=True)
    

    output = func(x, y)
    first_derivative = torch.autograd.grad(outputs=output, inputs=[x, y], create_graph=True)[var_index1]
    

    second_derivative = torch.autograd.grad(outputs=first_derivative, inputs=[x, y])[var_index2]
    
    return second_derivative.item()


################
# Example PDEs #
################
def heat_equation1D(u, t, x, c=1):
    """
    One-dimensional heat equation PDE residual
    """
    return first_partial_derivative(u,t,x,0) + (c**2)*second_partial_derivative(u,t,x,1,1)

def wave_equation1D(u, t, x, c=1):
    """
    One-dimensional heat equation PDE residual
    """
    return second_partial_derivative(u,t,x,0,0) + (c**2)*second_partial_derivative(u,t,x,1,1)

def u(t,x):
    return torch.sin(x)*torch.exp(-t^2)
