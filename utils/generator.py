import json
import time
import numpy as np

file_path = "node_values.json"
num_nodes = 33  # Number of nodes in your 33-bus system

# Possible static classes (each node keeps its class once assigned)
CLASSES = ["Residential", "Commercial", "Industrial"]
# Possible dynamic priorities
PRIORITIES = ["High", "Medium", "Low"]
# Possible communication modes
COMM_MODES = [0, 1, 2]

def generate_data(num_nodes, previous_data=None):
    """
    Generate a dictionary with the following top-level structure:
      {
        "comm_mode": <0 or 1 or 2>,
        "1": {"voltage": float, "load": float, "class": str, "priority": str},
        "2": {...},
        ...
      }
    If `previous_data` is None, assign random classes. Otherwise, keep classes.
    Voltage, Load, Priority, and comm_mode can change each iteration.
    """
    new_data = {}

    # Randomly choose a new comm_mode each iteration (or keep it stable if you prefer)
    new_data["comm_mode"] = int(np.random.choice(COMM_MODES))

    for node_id in range(1, num_nodes + 1):
        node_str = str(node_id)

        if previous_data is None or node_str not in previous_data:
            # First time initialization
            voltage = 1.0 + np.random.uniform(-0.05, 0.05)  # around 1.0 p.u.
            real_power = np.random.uniform(0.2, 1.5)             # random load
            reactive_power = np.random.uniform(0.2, 1.5)             # random load

            cls = np.random.choice(CLASSES)                # pick a class once
            prio = np.random.choice(PRIORITIES)            # initial priority
        else:
            # We have previous data. Keep the same class, but vary other fields.
            old_node = previous_data[node_str]
            cls = old_node["class"]

            # Slightly adjust voltage
            old_voltage = old_node["voltage"]
            voltage = voltage + np.random.uniform(-0.02, 0.02)
            # Clamp voltage between 0.90 and 1.10 for realism
            voltage = max(0.90, min(1.10, voltage))

            # Slightly adjust load
            old_real_power = old_node["real_power"]
            old_reactive_power = old_node["reactive_power"]
            real_power = old_load + np.random.uniform(-0.1, 0.1)
            reactive_power = old_load + np.random.uniform(-0.1, 0.1)
            # Clamp load between 0.0 and 2.0 (example range)
            
            real_power = max(0.0, min(2.0, real_power))
            reactive_power = max(0.0, min(2.0, reactive_power))
            # Priority can change each iteration
            prio = np.random.choice(PRIORITIES)

        # Construct the per-node data dict
        new_data[node_str] = {
            "voltage": round(float(voltage), 4),
            "real_power": round(float(real_power), 4),
            "reactive_power": round(float(real_power), 4),
            "class": cls,
            "priority": prio
        }

    return new_data


def main():
    previous_data = None

    while True:
        # Generate new data based on the previous iteration
        data = generate_data(num_nodes, previous_data)

        # Write new data to JSON
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"Updated node data written to {file_path}.")
        # Save for the next loop iteration
        previous_data = data

        # Update every 2 seconds (or 1 second, per your needs)
        time.sleep(2)

if __name__ == "__main__":
    main()
