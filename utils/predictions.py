import time
import torch
import json
import numpy as np
from model.GAT import GATModule
import pandapower as pp
import pandapower.networks as nw
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import pandas as pd

from utils.constants import CLASS_MAPPING, NODE_TYPE, PEAK_LOAD

network = nw.case33bw()
line_data = np.array(network.line[['from_bus', 'to_bus', 'length_km', 'r_ohm_per_km', 'x_ohm_per_km']])
# Example model loading (adapt as needed)
classification_model = torch.load("data/STGAT_ATT.pt")
priority_model = torch.load("data/best_priority_model.pt")
# Classification is only needed once
# class_input = torch.randn(24, 33, 2)
# For the moment we are given a base class
# class_output = classification_model(class_input)  # Suppose [1, 33] or [33]

# class_map = ["Residential", "Commercial", "Industrial"]
# class_labels = [class_map[idx.item()] for idx in class_indices]
priority_map = {
    0: 'N/A',
    1: 'Low',
    2: 'Medium',
    3: 'High'
}

def get_live_voltage_and_load():
    # Replace this with real USRP reading / calculation
    # Return two tensors: shape [33] for voltage, load
    voltage = torch.rand(33)  # placeholder
    load = torch.rand(33)     # placeholder
    return voltage, load

def get_mode(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
        mode = 0
        mode = data.get("comm_mode", mode)
    return mode

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # Load the trained model
    model = GATModule(in_channels=4, hidden_channels=128, last_layer_output=4, heads=4).to(device)
    model.load_state_dict(torch.load("data/best_priority_model.pt"))
    model.eval()
    while True:
        vectorized_mapping = np.vectorize(CLASS_MAPPING.get)
        node_type = vectorized_mapping(NODE_TYPE)

        # Acquire live voltage and load
        voltage, load = get_live_voltage_and_load()

        # Suppose you also have P, Q from somewhere, shape [33]
        load_data = network.load[['p_mw','q_mvar']]
        slack_bus_load = pd.DataFrame([[0, 0]], columns=['p_mw', 'q_mvar'])
        load_data = pd.concat([slack_bus_load, load_data], ignore_index=True)
        # print(load_data)
        random_values = np.random.uniform(0.5, 1.5, size=(33,1))
        random_loads = load_data * np.hstack((random_values, random_values))
        random_loads = torch.tensor(random_loads.to_numpy(), dtype=torch.float)

        # Convert class_indices to float so we can stack
        mode = get_mode("node_values.json")
        mode = torch.tensor(np.full((33,), mode))  # e.g. 0,1,2
        mode = mode.reshape(-1,1)

        # Create edge index (remains constant for all time steps)
        edge_index = np.vstack((line_data[:, 0], line_data[:, 1])).astype(int)
        edge_features = line_data[:, 2:5]  # Edge attributes (x, r, length)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_features = torch.tensor(edge_features, dtype=torch.float)

        # print(random_loads.shape, node_type.shape, mode.shape)
        priority_input = torch.hstack([random_loads, torch.tensor(node_type.reshape(-1,1)), mode])

        data = Data(
            x=priority_input,
            edge_index=edge_index,
            edge_attr=edge_features,
        ).to(device)

        # Get priority predictions
        priority_output = model(data)  # shape [33, #priority_classes]
        priority_indices = priority_output.argmax(dim=1)
        _, predicted_indices = torch.max(priority_output, dim=1)
        print(priority_indices)
        priority_labels = [priority_map[idx.item()] for idx in priority_indices]


        # Prepare JSON data
        json_data = {}
        for i in range(33):
            node_id = str(i + 1)
            json_data[node_id] = {
                "class": NODE_TYPE[i],           # from one-time classification
                "priority": priority_labels[i],     # from priority model
                "voltage": float(voltage[i]),
                "load": float(load[i])
            }

        # Overall comm_mode, or anything else you want at top-level
        json_data["comm_mode"] = int(mode[0].item())

        # Write to file
        with open("node_values.json", "w") as f:
            json.dump(json_data, f, indent=2)

        # Small delay (e.g., 2 seconds) before next loop
        time.sleep(1)

if __name__ == "__main__":
    main()
