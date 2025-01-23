import torch
import torch.nn
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from src.train import train_model
from src.test import evaluate_model
import matplotlib.pyplot as plt

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_list = create_dataset("data/cheddar_priority.h5") 
    train_data, temp_data = train_test_split(data_list, test_size = 0.3, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size = 0.5, random_state = 42)

    model = STGAT(input_channels, hidden_channels, output_channels, heads).to(device)
    model = model.to.device()

    optimizer = torch.optim.adam(model.parameters(), lr=0.0001)
    criterion = torch.nn.MSELoss()
    
    num_epochs = 100
    best_val_loss = float(inf)

    train_losses = []
    test_losses = []
    for epoch in range(num_epochs):
        train_loss = train_model(model, train_data, criterion, optimizer, device)
        val_loss = test_model(model, val_data, criterion, device)

        # train_losses.append(train_loss)
        # val_losses.append(val_loss)
        print(f"Epoch[{epoch+1}/{num_epochs}] Train_loss = {train_loss:.6f}, Val_loss = {val_loss:.6f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "data/best_priority_model.pt")

    # Option 2: Plot directly
    plt.figure()
    plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs+1), val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training/Validation Loss over Epochs')
    plt.legend()
    plt.savefig("results/loss_plot_GCN.png")
    plt.show()

if __name__ = "__main__":
    main()

