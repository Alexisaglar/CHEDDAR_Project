import matplotlib.pyplot as plt
import networkx as nx
import json
import time
import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.patches as mpatches
from matplotlib.widgets import Button  # <--- IMPORTANT

plt.style.use('seaborn-v0_8-darkgrid')

# 1) CREATE THE BASE 33-BUS GRAPH

G = nx.Graph()
edges = [
    (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10),
    (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16), (16, 17),
    (17, 18), (3, 19), (19, 20), (20, 21), (21, 22), (5, 23), (23, 24), (24, 25),
    (7, 26), (26, 27), (27, 28), (28, 29), (29, 30), (30, 31), (31, 32), (32, 33)
]
G.add_edges_from(edges)

pos = {
    1: (0, 0), 2: (1, 0), 3: (2, 0), 4: (3, 0), 5: (4, 0), 6: (5, 0), 7: (6, 0),
    8: (7, 0), 9: (8, 0), 10: (9, 0), 11: (10, 0), 12: (11, 0), 13: (12, 0),
    14: (13, 0), 15: (14, 0), 16: (15, 0), 17: (16, 0), 18: (17, 0),
    19: (2, -0.5), 20: (3, -0.5), 21: (4, -0.5), 22: (5, -0.5), 23: (4, 1), 24: (5, 1),
    25: (6, 1), 26: (6, 0.5), 27: (7, 0.5), 28: (8, 0.5), 29: (9, 0.5), 30: (10, 0.5),
    31: (11, 0.5), 32: (12, 0.5), 33: (13, 0.5)
}

file_path = "node_values.json"

# 2) DEFINE APPEARANCE MAPPINGS

# Priority -> face color
priority_color_map = {
    "High": "red",
    "Medium": "orange",
    "Low": "green"
}
default_priority_color = "gray"

# Bus class -> marker shape
class_shape_map = {
    "sesidential": "o",   # circle
    "commercial": "s",    # square
    "industrial": "^",    # triangle
}
default_class_shape = "o"  # fallback shape

# 3) READ (STATIC) BUS CLASS FROM JSON (assuming it doesn't change in real‐time)
try:
    with open(file_path, "r") as f:
        initial_data = json.load(f)
except FileNotFoundError:
    print("Warning: node_values.json not found at initialization.")
    initial_data = {}

global_comm_mode = initial_data.get("comm_mode", 0)
node_classes = {}
for n in G.nodes():
    node_str = str(n)
    node_info = initial_data.get(node_str, {})
    node_classes[n] = node_info.get("class", "Unknown")

# Group nodes by class so we can give them different shapes
class_groups = {}
for n in G.nodes():
    cls = node_classes[n]
    class_groups.setdefault(cls, []).append(n)

# 4) PREPARE MATPLOTLIB FIGURE

fig, ax = plt.subplots(figsize=(14, 8))
# ax.imshow(img, extent=[-20,20,-10,10])
ax.set_facecolor("white")
ax.set_title("33-Bus Smart Grid System", fontsize=16)

nx.draw_networkx_edges(G, pos, edge_color="gray", alpha=0.8, ax=ax)

scatter_dict = {}
facecolors_dict = {}
sizes_dict = {}
node_to_scatter_idx = {}
text_labels = {}

# We’ll only color nodes by priority (no more voltage colormap).

# 5) INITIALIZE SCATTERS (by class)
for cls_name, nodes_in_class in class_groups.items():
    xvals = [pos[n][0] for n in nodes_in_class]
    yvals = [pos[n][1] for n in nodes_in_class]

    # Initialize with some default face color (e.g., gray) and size
    init_facecolors = [default_priority_color]*len(nodes_in_class)
    init_sizes = [300]*len(nodes_in_class)  # or 200 if you want them smaller

    scatter = ax.scatter(
        xvals,
        yvals,
        s=init_sizes,
        c=init_facecolors,
        edgecolors="black",
        marker=class_shape_map.get(cls_name, default_class_shape),
        linewidths=1.5,
        zorder=3,
        alpha=0.9,
        label=cls_name  # for the class legend
    )

    scatter_dict[cls_name] = scatter
    facecolors_dict[cls_name] = np.array(init_facecolors, dtype=object)
    sizes_dict[cls_name] = np.array(init_sizes)
    
    for i, n in enumerate(nodes_in_class):
        node_to_scatter_idx[n] = (cls_name, i)
        x, y = pos[n]
        # We'll show Bus ID, Voltage, and Load in the label— but NO Priority line
        label = ax.text(
            x, y + 0.03,
            f"Bus {n}\nV=?\nLoad=?",
            fontsize=8,
            color="black",
            ha="center",
            va="bottom",
            zorder=5
        )
        text_labels[n] = label

# 6) LEGENDS (Class shapes, Priority colors)

# Make legend for bus classes (shapes). 
# Increase labelspacing so shapes don't overlap
class_legend = ax.legend(
    handles=[scatter_dict[k] for k in scatter_dict],
    loc="upper right",
    title="Bus Class (shapes)",
    fontsize=9,
    labelspacing=1.5  # <--- extra spacing to avoid overlap
)
ax.add_artist(class_legend)

# Priority color legend
priority_patches = [
    mpatches.Patch(color="red",     label="High Priority"),
    mpatches.Patch(color="orange",  label="Medium Priority"),
    mpatches.Patch(color="green",   label="Low Priority"),
    mpatches.Patch(color="gray",    label="Unknown Priority"),
]
priority_legend = ax.legend(
    handles=priority_patches,
    loc="center right",
    title="Node Priority",
    fontsize=9
)
ax.add_artist(priority_legend)

# Communication mode in a corner
comm_text_box = ax.text(
    0.02, 0.95, 
    f"Comm Mode: {global_comm_mode}",
    transform=ax.transAxes,
    fontsize=12,
    color="purple",
    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3')
)

# 7) ANIMATION FUNCTION

def update(frame):
    global global_comm_mode
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Data file not found. Waiting for updates...")
        return

    # Update communication mode
    new_comm_mode = data.get("comm_mode", global_comm_mode)
    if new_comm_mode != global_comm_mode:
        global_comm_mode = new_comm_mode
    comm_text_box.set_text(f"Comm Mode: {global_comm_mode}")

    for n in G.nodes():
        node_str = str(n)
        node_info = data.get(node_str, {})

        voltage = float(node_info.get("voltage", 1.0))
        load = float(node_info.get("load", 0.5))
        priority = node_info.get("priority", "Unknown")

        cls_name = node_classes[n]  # static class
        if cls_name in scatter_dict:
            scatter_handle = scatter_dict[cls_name]
            face_arr = facecolors_dict[cls_name]
            size_arr = sizes_dict[cls_name]
            idx = node_to_scatter_idx[n][1]

            # Face color now determined by priority (not voltage anymore)
            new_face_color = priority_color_map.get(priority, default_priority_color)
            face_arr[idx] = new_face_color

            # Scale size by load (optional)
            new_size = 300 + 300*load
            size_arr[idx] = new_size

        # Update text label WITHOUT the priority line
        text_labels[n].set_text(
            f"Bus {n}\n"
            f"V={voltage:.3f}\n"
            f"Load={load:.2f}"
        )

    # Push updated arrays back to each scatter
    for cls_name in scatter_dict:
        scatter_handle = scatter_dict[cls_name]
        scatter_handle.set_facecolors(facecolors_dict[cls_name])
        scatter_handle.set_sizes(sizes_dict[cls_name])


# 8) RUN THE ANIMATION
ani = FuncAnimation(fig, update, interval=2000)

# --- BUTTON CODE STARTS HERE ---
# Create an axis for our "Toggle Mode" button
button_ax = fig.add_axes([0.8, 0.02, 0.1, 0.05])
# Create the Button
comm_button = Button(
    button_ax,
    "Toggle Mode",
    # color="#EAEAF2",       # match the seaborn background or your preference
    hovercolor="#D0D0D0"
)

# comm_button.ax.set_frame_on(True)

for spine in comm_button.ax.spines.values():
    spine.set_edgecolor('black')  # border color
    spine.set_linewidth(1)        # border thickness
# Adjust the label style
comm_button.label.set_fontsize(9)
comm_button.label.set_color("black")
def toggle_mode(event):
    global global_comm_mode
    
    # 1) Load the current JSON (if it exists)
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        data = {}

    # 2) Cycle the mode 0 -> 1 -> 2 -> 0
    global_comm_mode = (global_comm_mode + 1) % 3
    
    # 3) Update the JSON with the new comm_mode
    data["comm_mode"] = global_comm_mode

    # 4) Write the JSON file back to disk
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)

    # 5) Update the text in the corner immediately
    comm_text_box.set_text(f"Comm Mode: {global_comm_mode}")

    # (Optional) Debug print
    print(f"Button clicked! Updated comm_mode -> {global_comm_mode}")

# Register the callback
comm_button.on_clicked(toggle_mode)

plt.axis('off')
# plt.tight_layout()
plt.show()
