import matplotlib.pyplot as plt
import networkx as nx
import json
import time
from matplotlib.animation import FuncAnimation
import numpy as np
from matplotlib import cm

# Create the graph (replace with your specific network structure)
# Define the graph
G = nx.Graph()

# Add edges based on the 33-bus system
edges = [
    (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10),
    (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16), (16, 17),
    (17, 18), (3, 19), (19, 20), (20, 21), (21, 22), (5, 23), (23, 24), (24, 25),
    (7, 26), (26, 27), (27, 28), (28, 29), (29, 30), (30, 31), (31, 32), (32, 33)
]
G.add_edges_from(edges)

# Define positions for the nodes
pos = {
    1: (0, 0), 2: (1, 0), 3: (2, 0), 4: (3, 0), 5: (4, 0), 6: (5, 0), 7: (6, 0),
    8: (7, 0), 9: (8, 0), 10: (9, 0), 11: (10, 0), 12: (11, 0), 13: (12, 0),
    14: (13, 0), 15: (14, 0), 16: (15, 0), 17: (16, 0), 18: (17, 0),
    19: (2, -1), 20: (3, -1), 21: (4, -1), 22: (5, -1), 23: (4, 2), 24: (5, 2),
    25: (6, 2), 26: (6, 1), 27: (7, 1), 28: (8, 1), 29: (9, 1), 30: (10, 1),
    31: (11, 1), 32: (12, 1), 33: (13, 1)
}

# File path for dynamic data updates
file_path = "node_values.json"

# Normalize values for color mapping
cmap = cm.get_cmap('Greens')

# Initialize figure and axis
fig, ax = plt.subplots(figsize=(14, 8))
# ax.set_title("33-Bus System Layout with Dynamic Updates", fontsize=16)

# Draw static edges
nx.draw_networkx_edges(G, pos, edge_color="gray", alpha=0.7, ax=ax)

# Draw initial nodes and labels
node_scatters = []
value_labels = {}
id_labels = {}
class_labels = {}

for node, (x, y) in pos.items():
    # Scatter for the node
    scatter = ax.scatter(x, y, s=700, color="white", edgecolors="black", zorder=2)
    node_scatters.append(scatter)

    # Display the node ID
    id_labels[node] = ax.text(
        x, y, str(node), fontsize=10, color="white", ha="center", va="center", zorder=3
    )

    # Placeholder for dynamic value (blue, above node)
    value_labels[node] = ax.text(
        x, y + 0.1, "Voltage:", fontsize=9, color="blue", ha="center", zorder=4
    )

    # Placeholder for dynamic class (red, below node)
    class_labels[node] = ax.text(
        x, y - 0.2, "Class:", fontsize=9, color="red", ha="center", zorder=4
    )

# Function to update the graph dynamically
def update(frame):
    try:
        # Load new node values and classes from JSON
        with open(file_path, "r") as f:
            node_data = json.load(f)

        # Update node colors, values, and classes
        for i, node in enumerate(G.nodes):
            # Get value and class
            value = float(node_data.get(str(node), {}).get("value", 0))  # Default value
            class_label = node_data.get(str(node), {}).get("class", "N/A")  # Default class

            # Update node color
            color = cmap(value / 220)  # Normalize to [0, 1]
            node_scatters[i].set_facecolor(color)

            # Update value label
            value_labels[node].set_text(f"Voltage: \n{value:.1f}")

            # Update class label
            class_labels[node].set_text(f"Class: \n{class_label}")
    except FileNotFoundError:
        print("Data file not found. Waiting for updates...")

# Animate the plot every 20 seconds
ani = FuncAnimation(fig, update, interval=1000)

plt.axis("off")
plt.show()
