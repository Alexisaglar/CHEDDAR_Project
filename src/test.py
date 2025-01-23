import torch

def train_model(model, data, criterion, device):
    total_loss = 0.0 
    for batch in data:
        batch = batch.to(device)

        target_predicted = model(batch)
        target = data.y.to(device).squeeze()

        loss = criterion(target_predicted.squeeze(), target)
        total_loss += loss.item()

    return total_loss/len(data)
