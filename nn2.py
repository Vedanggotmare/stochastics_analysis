import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
import networkx as nx

# ─── PARAMETERS ────────────────────────────────────────────────────────────────
k = 14
n_neurons = [10,15,15,15,15,15,15,15,15,15,15,15,15,10]   # Number of neurons in each of the k layers
p_conn = 0.1                 # Probability of connecting any two neurons between adjacent layers

total_nodes = sum(n_neurons)

# ─── ASSIGN NODE INDICES & LAYER MAPPING ───────────────────────────────────────
layer_of = {}
nodes_in_layer = []
counter = 0
for layer_idx in range(k):
    this_layer = []
    for _ in range(n_neurons[layer_idx]):
        layer_of[counter] = layer_idx
        this_layer.append(counter)
        counter += 1
    nodes_in_layer.append(this_layer)

# ─── COMPUTE 2D POSITIONS FOR PLOTTING ─────────────────────────────────────────
pos_2d = {}
for layer_idx in range(k):
    Nl = n_neurons[layer_idx]
    y_vals = np.linspace(-(Nl - 1) / 2, (Nl - 1) / 2, Nl)
    for j, node_id in enumerate(nodes_in_layer[layer_idx]):
        x = layer_idx
        y = y_vals[j]
        pos_2d[node_id] = (x, y)

# ─── ASSIGN REMOVAL PROBABILITIES ──────────────────────────────────────────────
removal_weights = np.random.rand(total_nodes)
removal_weights /= removal_weights.sum()

remaining = set(range(total_nodes))
removal_order = []         # List of nodes in the order they get removed
fraction_LCC = []          # fraction_LCC[t] = size of LCC / total_nodes after t+1 removals
edges_history = []         # edges_history[t] = list of edges (tuples) among remaining nodes after t+1 removals

# ─── FULL PRECOMPUTATION: NODE REMOVAL, EDGE RE‐SAMPLING, LCC CALC ─────────────
for t in range(total_nodes):
    # 1) Pick which node to remove (weighted by removal_weights among remaining)
    rem_list = np.array(sorted(list(remaining)))
    weights = removal_weights[rem_list]
    weights /= weights.sum()
    chosen_idx = np.random.choice(len(rem_list), p=weights)
    node_to_remove = int(rem_list[chosen_idx])

    # 2) Remove that node
    remaining.remove(node_to_remove)
    removal_order.append(node_to_remove)

    # 3) Re‐sample edges among the remaining nodes (only between adjacent layers)
    new_edges = []
    for layer_idx in range(k - 1):
        layer_l = [i for i in nodes_in_layer[layer_idx]     if i in remaining]
        layer_r = [j for j in nodes_in_layer[layer_idx + 1] if j in remaining]
        for i in layer_l:
            for j in layer_r:
                if np.random.rand() < p_conn:
                    new_edges.append((i, j))

    # 4) Build a quick graph on the remaining nodes + these new edges
    G_current = nx.Graph()
    G_current.add_nodes_from(remaining)
    G_current.add_edges_from(new_edges)

    # 5) Compute size of largest connected component (LCC)
    if len(G_current) > 0:
        largest_cc = max(nx.connected_components(G_current), key=len)
        size_lcc = len(largest_cc)
    else:
        size_lcc = 0

    fraction_LCC.append(size_lcc / total_nodes)
    edges_history.append(new_edges)

# ─── SET UP FIGURE & ARTISTS FOR ANIMATION ─────────────────────────────────────
fig, (ax_net, ax_frac) = plt.subplots(1, 2, figsize=(12, 5))

# --- Left panel: network layout ---
ax_net.set_title("Network (nodes removed → edges re-sampled every step)")
ax_net.set_xlim(-0.5, k - 0.5)
max_layer_size = max(n_neurons)
ax_net.set_ylim(-(max_layer_size / 2 + 1), (max_layer_size / 2 + 1))
ax_net.set_aspect('equal')
ax_net.axis('off')

# Scatter for all nodes; we’ll control alpha to “fade out” removed nodes
node_scatter = ax_net.scatter(
    [pos_2d[i][0] for i in range(total_nodes)],
    [pos_2d[i][1] for i in range(total_nodes)],
    s=100,
    c='C0',
    edgecolors='k',
    zorder=2
)
node_colors = np.array([[0.2, 0.4, 0.8, 1.0] for _ in range(total_nodes)])
node_scatter.set_facecolors(node_colors)

# A single LineCollection for edges; initialize with the edges after 1st removal (frame=0)
from matplotlib.collections import LineCollection
initial_segments = [
    [pos_2d[u], pos_2d[v]]
    for (u, v) in edges_history[0]
]
lc = LineCollection(initial_segments, colors='gray', linewidths=0.7, zorder=1)
ax_net.add_collection(lc)

# --- Right panel: fraction plot setup ---
ax_frac.set_title("Fraction of nodes in LCC")
ax_frac.set_xlabel("Nodes removed")
ax_frac.set_ylabel("Fraction in largest component")
ax_frac.set_xlim(0, total_nodes)
ax_frac.set_ylim(0, 1.05)
ax_frac.grid(True, linestyle=':', alpha=0.5)

frac_line, = ax_frac.plot([], [], color='C1', lw=2)
dot, = ax_frac.plot([], [], 'o', color='C1', markersize=6)  # moving dot

x_data = []
y_data = []

# ─── ANIMATION UPDATE FUNCTION ────────────────────────────────────────────────
def update(frame):
    """
    frame = 0 .. total_nodes-1.
    At frame t:
      - removal_order[t] is the node removed at this step.
      - edges_history[t] is the new list of edges among remaining nodes after that removal.
      - fraction_LCC[t] is the LCC fraction after that removal.
    We do:
      1) Fade out node removal_order[t].
      2) Update the LineCollection to use edges_history[t].
      3) Update the fraction plot (line + moving dot).
    """
    node_removed = removal_order[frame]

    # 1) Fade out the removed node by setting its alpha = 0
    node_colors[node_removed, -1] = 0.0
    node_scatter.set_facecolors(node_colors)

    # 2) Update the edge collection’s segments
    segments = [
        [pos_2d[u], pos_2d[v]]
        for (u, v) in edges_history[frame]
    ]
    lc.set_segments(segments)

    # 3) Update fraction plot
    x_data.append(frame + 1)
    y_data.append(fraction_LCC[frame])
    frac_line.set_data(x_data, y_data)
    dot.set_data([x_data[-1]], [y_data[-1]])

    return node_scatter, lc, frac_line, dot

# ─── CREATE & RUN ANIMATION ───────────────────────────────────────────────────
ani = FuncAnimation(
    fig,
    update,
    frames=total_nodes,
    interval=15,    # ms between frames; tweak for playback speed
    blit=True
)

plt.tight_layout()
plt.show()
