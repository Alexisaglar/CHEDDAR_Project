import torch
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import mean_absolute_error  # <-- NEW
from torch_geometric.loader import DataLoader
from sklearn.exceptions import UndefinedMetricWarning
from src.data_loader import create_dataset
from model.GAT import GATModule 
import warnings
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

def test_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Test Data
    data_list = create_dataset("data/priority_dataset.h5")
    _, data_test = train_test_split(data_list, test_size=0.3, random_state=42)
    test_loader = DataLoader(data_test, batch_size=1, shuffle=False)

    # Load the trained model
    input_channels = 4
    hidden_channels = 128
    output_channels = 4
    heads = 4
    model = GATModule(input_channels, hidden_channels, output_channels, heads).to(device)
    model.load_state_dict(torch.load("data/best_priority_model.pt"))
    model.eval()

    # Define loss functions
    criterion = torch.nn.CrossEntropyLoss()

    # Initialize metrics
    all_preds_bus = []
    all_targets_bus = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            bus_out = model(batch)

            # --- Bus classification task ---
            target_bus = batch.target.to(device)
            loss_bus = criterion(bus_out, target_bus)

            bus_preds = torch.argmax(bus_out, dim=1).cpu().numpy()
            bus_targets = target_bus.cpu().numpy()

            all_preds_bus.extend(bus_preds)
            all_targets_bus.extend(bus_targets)

    # --- Final metrics across test set ---

    # 1) Bus Classification
    accuracy_bus = accuracy_score(all_targets_bus, all_preds_bus)

    print(f"Bus Classification Accuracy: {accuracy_bus:.6f}")
    # --- Visualization ---

    # Confusion Matrix for Bus Classification
    cm = confusion_matrix(all_targets_bus, all_preds_bus)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(4))
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix - Bus Classification")
    plt.show()

if __name__ == "__main__":
    test_model()
