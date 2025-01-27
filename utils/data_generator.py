import json
import time
import numpy as np

# Simulate node values and classes
def generate_data(num_nodes, previous_data=None):
    classes = ["Residential", "Commercial", "Industrial"]  # Possible classes
    priority = ["High", "Medium", "Low"]  # Possible classes
    if previous_data is None:
        # Start with random values and assign random classes if there's no previous data
        return {
            str(i): {
                "value": np.random.rand() * 100,
                "class": np.random.choice(classes),
                "priority": np.random.choice(priority)
            }
            for i in range(num_nodes)
        }
    else:
        # Vary values randomly, keeping them within a reasonable range, and keep the class constant
        return {
            str(i): {
                "value": max(200, min(220, previous_data[str(i)]["value"] + np.random.uniform(-10, 10))),
                # "class": previous_data[str(i)]["class"],
                "priority": previous_data[str(i)]["priority"]
            }
            for i in range(num_nodes)
        }

# Write to JSON file
file_path = "node_values.json"
num_nodes = 33  # Number of nodes in your graph

# Initialize previous data
previous_data = None

while True:
    # Generate new data based on the previous data
    data = generate_data(num_nodes, previous_data)
    
    # Write new data to the JSON file
    with open(file_path, "w") as f:
        json.dump(data, f)
    print(f"Updated node data: {data}")
    
    # Update previous data for the next iteration
    previous_data = data
    
    # Update every 2 seconds
    time.sleep(1)
