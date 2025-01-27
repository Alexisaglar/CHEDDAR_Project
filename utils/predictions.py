import time
import torch
import json

# Example model loading (adapt as needed)
classification_model = torch.load("data/STGAT_ATT.pt")
priority_model = torch.load("data/best_priority_model.pt")

# Classification is only needed once
class_input = torch.randn(24, 33, 2)
class_output = classification_model(class_input)  # Suppose [1, 33] or [33]

class_map = ["Residential", "Commercial", "Industrial"]
class_labels = [class_map[idx.item()] for idx in class_indices]

priority_map = ["Low", "Medium", "High"]

def get_live_voltage_and_load():
    # Replace this with real USRP reading / calculation
    # Return two tensors: shape [33] for voltage, load
    voltage = torch.rand(33)  # placeholder
    load = torch.rand(33)     # placeholder
    return voltage, load

def get_mode():
    # Return an integer or tensor representing the mode
    return 1  # placeholder

def main_loop():
    while True:
        # Acquire live voltage and load
        voltage, load = get_live_voltage_and_load()

        # Suppose you also have P, Q from somewhere, shape [33]
        P = torch.rand(33)
        Q = torch.rand(33)

        # Build [33, 4] input for priority model: [P, Q, Class, Mode]
        # Convert class_indices to float so we can stack
        Mode = torch.randint(0, 3, (33,))  # e.g. 0,1,2
        priority_input = torch.stack([P, Q, class_output, Mode], dim=1)

        # Get priority predictions
        priority_output = priority_model(priority_input)  # shape [33, #priority_classes]
        priority_indices = priority_output.argmax(dim=1)
        priority_labels = [priority_map[idx.item()] for idx in priority_indices]

        # Prepare JSON data
        json_data = {}
        for i in range(33):
            node_id = str(i + 1)
            json_data[node_id] = {
                "class": class_labels[i],           # from one-time classification
                "priority": priority_labels[i],     # from priority model
                "voltage": float(voltage[i]),
                "load": float(load[i])
            }

        # Overall comm_mode, or anything else you want at top-level
        json_data["comm_mode"] = int(Mode[0].item())

        # Write to file
        with open("node_values.json", "w") as f:
            json.dump(json_data, f, indent=2)

        # Small delay (e.g., 2 seconds) before next loop
        time.sleep(1)

if __name__ == "__main__":
    main_loop()
