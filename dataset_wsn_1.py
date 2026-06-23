import random
import numpy as np
import pandas as pd
import networkx as nx
from node2vec import Node2Vec
import matplotlib
matplotlib.use('Agg')  # Must be declared before importing pyplot
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import xml.etree.ElementTree as ET
# minidom no longer needed (manual XML generation)
import os
import sys

# ----------------------------
# Initialization and Seeding
# ----------------------------
random.seed(7)
np.random.seed(7)

# =========================================================
# PHASE 0: DATASET LOADING & COLUMN NORMALIZATION
# =========================================================
DATASET_PATH = "WSN_Dataset.csv"   # <-- set your CSV filename here

if not os.path.exists(DATASET_PATH):
    print(f"ERROR: Dataset file '{DATASET_PATH}' not found.")
    print("Please download it from: https://www.kaggle.com/datasets/ziya07/wireless-sensor-network-dataset")
    print("Then place the CSV in the same directory as this script.")
    sys.exit(1)

df = pd.read_csv(DATASET_PATH)
df = df.head(100)
print(f"Loaded dataset: {df.shape[0]} rows x {df.shape[1]} cols")
print("Columns detected:", list(df.columns))

# ---- Exact column mapping for this dataset ----
# Columns: Node_ID, Timestamp, X_Coordinate, Y_Coordinate, Z_Coordinate,
#          Initial_Energy, Residual_Energy, Transmission_Power,
#          Signal_Strength, Noise_Level

df["_node_id"]  = df["Node_ID"].astype(str)
df["_x"]        = pd.to_numeric(df["X_Coordinate"], errors="coerce").fillna(0)
df["_y"]        = pd.to_numeric(df["Y_Coordinate"], errors="coerce").fillna(0)
df["_z"]        = pd.to_numeric(df["Z_Coordinate"], errors="coerce").fillna(0)

# Normalise X/Y to [0, 200] to match original topology scale
df["_x"] = (df["_x"] - df["_x"].min()) / (df["_x"].max() - df["_x"].min() + 1e-9) * 200
df["_y"] = (df["_y"] - df["_y"].min()) / (df["_y"].max() - df["_y"].min() + 1e-9) * 200

df["_initial_energy"]  = pd.to_numeric(df["Initial_Energy"],      errors="coerce").fillna(0)
df["_residual_energy"] = pd.to_numeric(df["Residual_Energy"],      errors="coerce").fillna(0)
df["_tx_power"]        = pd.to_numeric(df["Transmission_Power"],   errors="coerce").fillna(0)
df["_signal"]          = pd.to_numeric(df["Signal_Strength"],      errors="coerce").fillna(0)
df["_noise"]           = pd.to_numeric(df["Noise_Level"],          errors="coerce").fillna(0)
df["_energy"]          = df["_residual_energy"]   # alias used downstream

# ---- Node role derivation (no explicit type column) ----
# Strategy: nodes in the top 20% of Initial_Energy are infrastructure (cluster head candidates);
#           bottom 2 highest-energy nodes become sinks (origins).
#           All others are clients.
energy_80pct = df["_initial_energy"].quantile(0.80)
df["_node_type"] = df["_initial_energy"].apply(
    lambda e: "infra" if e >= energy_80pct else "client"
)
print(f"Column mapping applied directly from known schema.")
print(f"  Infra nodes (top 20% energy): {(df['_node_type']=='infra').sum()}")
print(f"  Client nodes: {(df['_node_type']=='client').sum()}")

# =========================================================
# TOPOLOGY CONFIGURATION
# =========================================================
df = df.reset_index(drop=True)
num_nodes = len(df)
positions = list(zip(df["_x"], df["_y"]))

client_indices = df[df["_node_type"] == "client"].index.tolist()
infra_indices  = df[df["_node_type"] == "infra"].index.tolist()

if len(infra_indices) >= 3:
    origin_indices = infra_indices[-2:]
    infra_nodes    = infra_indices[:-2]
else:
    origin_indices = infra_indices[-1:]
    infra_nodes    = infra_indices[:-1] if len(infra_indices) > 1 else infra_indices

if not infra_nodes:
    pool = [i for i in range(num_nodes) if i not in origin_indices]
    infra_nodes = random.sample(pool, min(8, len(pool)))

print(f"\nTopology: {len(client_indices)} clients | "
      f"{len(infra_nodes)} infra | {len(origin_indices)} origins")

def get_pos(i):
    return positions[i]

# =========================================================
# LEACH CLUSTERING ALGORITHM
# =========================================================
P = 0.3

cluster_heads = [i for i in infra_nodes if random.random() < P]
if not cluster_heads:
    cluster_heads.append(random.choice(infra_nodes))

clusters = {ch: [] for ch in cluster_heads}

for i in range(num_nodes):
    if i in cluster_heads or i in origin_indices:
        continue
    best = min(cluster_heads,
               key=lambda ch: (get_pos(i)[0] - get_pos(ch)[0])**2
                             + (get_pos(i)[1] - get_pos(ch)[1])**2)
    clusters[best].append(i)

client_to_ch = {}
for ch, members in clusters.items():
    for m in members:
        if m in client_indices:
            client_to_ch[m] = ch

print(f"LEACH: {len(cluster_heads)} cluster heads elected, "
      f"{len(client_to_ch)} clients assigned.")

# =========================================================
# GRAPH TOPOLOGY EXTRACTION
# =========================================================
G = nx.Graph()
for i in range(num_nodes):
    G.add_node(i)
for c, ch in client_to_ch.items():
    G.add_edge(c, ch)
for ch in cluster_heads:
    if origin_indices:
        G.add_edge(ch, origin_indices[0])

print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# =========================================================
# PHASE 1: DATASET SUMMARY
# =========================================================
print("\n--- Dataset Summary ---")
summary_cols = ["_node_id", "_x", "_y", "_z", "_initial_energy",
                "_residual_energy", "_tx_power", "_signal", "_noise", "_node_type"]
print(df[[c for c in summary_cols if c in df.columns]].describe(include="all").to_string())

df["_cluster_head"] = df.index.map(
    lambda i: client_to_ch.get(i, i if i in cluster_heads else -1))
df.to_csv("wsn_processed.csv", index=False)
print("\nPhase 1: Dataset processed and cluster assignments saved to 'wsn_processed.csv'.")

# =========================================================
# PHASE 1b: NETANIM-COMPATIBLE XML GENERATION
# =========================================================
def build_netanim_xml(positions, cluster_heads, origin_indices, clusters,
                      client_to_ch, num_nodes, sim_duration=20.0,
                      output_path="wsn_2106.xml"):
    """
    Generates a NetAnim XML file matching the exact format of ns-3 AnimationInterface output.
    Format verified from a real ns-3-generated file (netanim-3.109):
      <anim ver="netanim-3.109" filetype="animation" >
        <node id="N" sysId="0" locX="x" locY="y" />
        <nu p="0" t="0" id="N" r=R g=G b=B />
        <link fromId="A" toId="B" fd="..." td="..." ld="" />
        <wp .../>   (wireless packet events)
      </anim>
    Node colours:
      Sink/Origin  : black (r=0,   g=0,   b=0)
      Cluster Head : red   (r=255, g=0,   b=0)
      Client node  : blue  (r=0,   g=0,   b=255)
    """
    origin_set = set(origin_indices)
    ch_set     = set(cluster_heads)

    lines = []
    lines.append('<?xml version="1.0" encoding="utf-8"?>')
    lines.append('<anim ver="netanim-3.109" filetype="animation" >')

    # -- Node position entries --
    for i in range(num_nodes):
        x, y = positions[i]
        lines.append(f'  <node id="{i}" sysId="0" locX="{x:.4f}" locY="{y:.4f}" />')

    # -- Node colour/size/label updates at t=0 --
    for i in range(num_nodes):
        if i in origin_set:
            r, g, b, w, h, descr = 0, 0, 0, 3.0, 3.0, f"Sink-{i}"
        elif i in ch_set:
            r, g, b, w, h, descr = 255, 0, 0, 2.5, 2.5, f"CH-{i}"
        else:
            r, g, b, w, h, descr = 0, 0, 255, 1.0, 1.0, f"Node-{i}"
        lines.append(
            f'  <nu p="0" t="0" id="{i}" ' +
            f'r="{r}" g="{g}" b="{b}" ' +
            f'w="{w:.1f}" h="{h:.1f}" descr="{descr}" />')

    # -- Link entries --
    link_id = 0
    for c, ch in client_to_ch.items():
        lines.append(
            f'  <link fromId="{c}" toId="{ch}" ' +
            f'fd="Node-{c}" td="CH-{ch}" ld="" />')
        link_id += 1
    for ch in cluster_heads:
        if origin_indices:
            sink = origin_indices[0]
            lines.append(
                f'  <link fromId="{ch}" toId="{sink}" ' +
                f'fd="CH-{ch}" td="Sink-{sink}" ld="" />')
            link_id += 1

    # -- Wireless packet events (one per cluster head -> sink) --
    t = 2.0
    for ch in cluster_heads[:10]:
        sink = origin_indices[0] if origin_indices else ch
        lines.append(
            f'  <wp fId="{ch}" fbTx="{t:.6f}" lbTx="{t+0.001:.6f}" ' +
            f'range="30.0" tId="{sink}" ' +
            f'fbRx="{t+0.002:.6f}" lbRx="{t+0.003:.6f}" />')
        t += 1.0

    lines.append('</anim>')

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Phase 1b: NetAnim XML saved to '{output_path}' "
          f"({link_id} links, {num_nodes} nodes).")
build_netanim_xml(
    positions, cluster_heads, origin_indices,
    clusters, client_to_ch, num_nodes,
    output_path="wsn_2106.xml"
)

# =========================================================
# PHASE 2: NODE2VEC EMBEDDING GENERATION
# =========================================================
# Parameters tuned for large graphs (10k+ nodes) to avoid OOM:
#   num_walks  : 10  (was 200 — the main RAM killer)
#   walk_length: 10  (was 20)
#   workers    : 2   (was 4 — reduces parallel memory pressure)
#   batch_words: 64  (was 256 — smaller Word2Vec batches)
node2vec = Node2Vec(G, dimensions=64, walk_length=10,
                    num_walks=10, workers=2, seed=7,
                    quiet=True)
model = node2vec.fit(window=5, min_count=1, batch_words=64)

nodes_list = list(G.nodes())
X = np.vstack([model.wv[str(n)] for n in nodes_list])

pd.DataFrame(X, index=nodes_list).to_csv("wsn_embeddings.csv")
print("Phase 2: Graph embeddings generated securely.")

# =========================================================
# PHASE 3: GRAPHICAL VISUALIZATION
# Node types clearly differentiated:
#   - Sink/Origin : black star  (★)  size=60
#   - Cluster Head: red square  (s)  size=80
#   - Client node : small circle      size=8  (colour = cluster)
# =========================================================
fig, ax = plt.subplots(figsize=(14, 12), dpi=150)

cmap   = plt.colormaps.get_cmap('tab20').resampled(max(len(cluster_heads), 1))
pos_dict = {i: positions[i] for i in range(num_nodes)}

# 1. Draw member nodes (tiny circles, cluster colour)
for idx, (ch, members) in enumerate(clusters.items()):
    if members:
        nx.draw_networkx_nodes(G, pos_dict, ax=ax,
                               nodelist=members,
                               node_color=[cmap(idx % 20)],
                               node_size=8,
                               node_shape='o',
                               alpha=0.7)

# 2. Draw cluster heads (red squares, slightly larger)
nx.draw_networkx_nodes(G, pos_dict, ax=ax,
                       nodelist=cluster_heads,
                       node_color='red',
                       node_size=80,
                       node_shape='s',
                       alpha=1.0,
                       linewidths=0.8,
                       edgecolors='darkred')

# 3. Draw sink/origin nodes (black stars, largest)
nx.draw_networkx_nodes(G, pos_dict, ax=ax,
                       nodelist=origin_indices,
                       node_color='black',
                       node_size=200,
                       node_shape='*',
                       alpha=1.0)

# 4. Draw edges (thin, semi-transparent)
nx.draw_networkx_edges(G, pos_dict, ax=ax,
                       alpha=0.15,
                       width=0.4,
                       edge_color='gray')

# 5. Legend
legend_handles = [
    mpatches.Patch(color='black',       label='Sink / Origin node'),
    mpatches.Patch(color='red',         label='Cluster Head'),
    mpatches.Patch(color='steelblue',   label='Client node (colour = cluster)'),
]
ax.legend(handles=legend_handles, loc='upper left',
          fontsize=9, framealpha=0.85)

ax.set_title(
    "LEACH Clustering — Wireless Sensor Network Dataset\n"
    f"{num_nodes} nodes | {len(cluster_heads)} cluster heads | "
    f"{len(origin_indices)} sinks | "
    f"Role derived from Initial_Energy (top 20% = infra)",
    fontsize=11, pad=12)
ax.axis('off')

plt.tight_layout()
plt.savefig("clusters.png", dpi=150, bbox_inches="tight")
print("Phase 3: Visualization saved to 'clusters.png'.")

print("\nOutput files:")
print("  wsn_2106.xml        — NetAnim animation file")
print("  clusters.png        — cluster topology plot")
print("  wsn_embeddings.csv  — Node2Vec embeddings (64-dim)")
print("  wsn_processed.csv   — original data + cluster assignments")

import os
os._exit(0)
