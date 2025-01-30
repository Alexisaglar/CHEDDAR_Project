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
from filelock import FileLock

from utils.constants import CLASS_MAPPING, NODE_TYPE, PEAK_LOAD
lock = FileLock("voltages_loads.json.lock")


network = nw.case33bw()
line_data = np.array(network.line[['from_bus', 'to_bus', 'length_km', 'r_ohm_per_km', 'x_ohm_per_km']])
# Example model loading (adapt as needed)
classification_model = torch.load("data/STGAT_ATT.pt")
priority_model = torch.load("data/best_priority_model.pt")
priority_map = {
    0: 'N/A',
    1: 'Low',
    2: 'Medium',
    3: 'High'
}

def get_mode(file_path):
    mode = 0
    with lock:
        with open(file_path, 'r') as f:
            data = json.load(f)  # Load the entire JSON object
        mode = data.get('comm_mode', None)
    return mode


def get_live_data(file_path):
    with lock:
        with open(file_path, 'r') as f:
            data = json.load(f)  # Load the entire JSON object
        voltage = [entry['voltage'] for key, entry in data.items() if key.isdigit()]
        real_power = [entry['active'] for key, entry in data.items() if key.isdigit()]
        reactive_power = [entry['reactive'] for key, entry in data.items() if key.isdigit()]
        # mode = data.get('comm_mode', None)
        # print(voltage)
    return voltage, real_power, reactive_power

def load_existing_data(file_path):
    with lock:
        with open(file_path, 'r') as f:
            return json.load(f)

def update_priority(json_data, priority_labels, priority_indices, mode):
    priority_indices = priority_indices.cpu().numpy()
    sorted_indices = sorted(range(len(priority_indices)), key=lambda i: priority_indices[i])
    sorted_indices = sorted_indices[::-1]
    sorted_arr = [priority_indices[i] for i in sorted_indices]
    print(sorted_arr)
    # print(sorted_indices[::-1])
    # print(sorted_indices)
    # indexed_dict = {index: value for index, value in enumerate(priority_indices)}
    # print(indexed_dict)
    priority_truncated= np.zeros(33)
    if mode == 0: capacity = 33
    elif mode == 1: capacity = 21
    else: capacity = 9 
    temp_arr = []
    print(sorted_indices)
    
    print(sorted_arr)
    for i in range(capacity):
        priority_truncated[sorted_indices[i]] = sorted_arr[i]

    print(priority_truncated)
    temp_arr = []
    for i in range(len(priority_truncated)):
        temp_arr.append(priority_map[priority_truncated[i]])
    print( temp_arr )

    for i in range(33):
        node_id = str(i + 1)
        if node_id in json_data:
            json_data[node_id]["priority"] = temp_arr[i]
            json_data[node_id]["class"] = NODE_TYPE[i]
        else:
            # Initialize node data if it doesn't exist
            json_data[node_id] = {"priority": temp_arr[i]}
            json_data[node_id]["class"] = NODE_TYPE[i]
    return json_data

def save_json_data(file_path, data):
    with lock:
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)

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
        voltage, real_power, reactive_power = get_live_data("voltages_loads.json")
        mode = get_mode("node_values.json")

        real_power = np.array(real_power).reshape(-1,1)
        reactive_power = np.array(reactive_power).reshape(-1,1)
        load = np.hstack((real_power, reactive_power))

        # print(load)
        load = torch.tensor(load, dtype=torch.float)
        # Convert class_indices to float so we can stack
        # mode = get_mode("node_values.json")
        mode = torch.tensor(np.full((33,), mode))  # e.g. 0,1,2
        mode = mode.reshape(-1,1)

        # Create edge index (remains constant for all time steps)
        edge_index = np.vstack((line_data[:, 0], line_data[:, 1])).astype(int)
        edge_features = line_data[:, 2:5]  # Edge attributes (x, r, length)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_features = torch.tensor(edge_features, dtype=torch.float)

        # print(random_loads.shape, node_type.shape, mode.shape)
        priority_input = torch.hstack([load, torch.tensor(node_type.reshape(-1,1)), mode])
        # print(priority_input.shape)

        data = Data(
            x=priority_input,
            edge_index=edge_index,
            edge_attr=edge_features,
        ).to(device)

        # Get priority predictions
        priority_output = model(data)  # shape [33, #priority_classes]
        priority_indices = priority_output.argmax(dim=1)
        _, predicted_indices = torch.max(priority_output, dim=1)
        # print(priority_indices)
        priority_labels = [priority_map[idx.item()] for idx in priority_indices]



        mode = int(mode.numpy()[0])
        # Load existing JSON data
        json_data = load_existing_data("voltages_loads.json")

        # Update priority fields
        json_data = update_priority(json_data, priority_labels, priority_indices, mode)

        # Update comm_mode
        json_data["comm_mode"] = mode 
        # Save updated JSON data
        save_json_data("node_values.json", json_data)

        # Small delay before next loop
        time.sleep(1)

if __name__ == "__main__":
    main()
