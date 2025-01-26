import torch
from torch.autograd import backward

def train_model(model, data, criterion, optimizer, device):
    model.train()
    total_loss = 0.0 

    for batch in data:
        batch = batch.to(device)
        optimizer.zero_grad()

        target_predicted = model(batch)
        target = batch.target.to(device).squeeze()

        loss = criterion(target_predicted.squeeze(), target)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss/len(data)
