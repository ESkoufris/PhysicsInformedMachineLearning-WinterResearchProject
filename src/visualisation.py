from src.grid import Grid
from src.pinn import PINN
import numpy as np
import plotly.graph_objects as go


def visualise_solution(pinn, grid: Grid, title = "PINN-approximated solution"):
    x = grid[...,0].flatten()
    t = grid[...,1].flatten()
    z = pinn(x,t)

    L = grid.x_end 
    T = grid.t_end

    x = x.detach().numpy()
    t = t.detach().numpy()
    z = z.detach().numpy()

    m,M = min(z),max(z)

    fig = go.Figure(data=[go.Surface(z=z.reshape(grid[...,0].shape), x=grid[...,0], y=grid[...,1])])
    
    fig.update_layout(
        title='PINN heat equation solution',
        scene=dict(
            xaxis=dict(
                title='x',
                range=[0, L],
                tickmode='linear',
                tick0=0,
                dtick=0.5
            ),
            yaxis=dict(
                title='t',
                range=[0, T],
                tickmode='linear',
                tick0=0,
                dtick=0.5
            ),
            zaxis=dict(
                title=title,
                range=[1.1*m, 1.1*M],
                tickmode='linear',
                tick0=0,
                dtick=0.1
            ),
        )
    )

    # Show the plot
    fig.show()