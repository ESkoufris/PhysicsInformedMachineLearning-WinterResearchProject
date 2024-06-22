from src.pinn import PINN
import torch
from torch import optim
import tqdm 

def train(pinn: PINN, lr=0.001, nepochs=100, batch_size=30):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pinn.to(device)

    optimizer = optim.Adam(pinn.parameters(), lr=lr)

    dataloader = DataLoader(DatasetWrapper(x, y), batch_size=batch_size, 
                            shuffle=True)
    
    loop = tqdm(range(nepochs), ncols=110)
    for i in loop:
        pass
    
    


