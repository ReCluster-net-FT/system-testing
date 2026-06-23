import random
import numpy as np
import pandas as pd
import networkx as nx
from node2vec import Node2Vec
import matplotlib 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import xml.etree.ElementTree as ET
import os
import sys
matplotlib.use('Agg') 

random.seed(7)
np.random.seed(7)

# =========================================================
# PHASE 0: DATASET LOADING & COLUMN NORMALIZATION
# =========================================================
DATASET_PATH = "wsn_qaga_dataset.csv"  

if not os.path.exists(DATASET_PATH):
    print(f"ERROR: Dataset file '{DATASET_PATH}' not found.")
    print("Please ensure wsn_qaga_dataset.csv is in the current directory.")
    sys.exit(1)

df = pd.read_csv(DATASET_PATH)
# Using ALL 1500 nodes 
print(f"Loaded dataset: {df.shape[0]} rows x {df.shape[1]} cols")
print("Columns detected:", list(df.columns))


df["_node_id"] = df["Node_ID"].astype(str)
df["_x"] = pd.to_numeric(df["X_Location"], errors="coerce").fillna(0)
df["_y"] = pd.to_numeric(df["Y_Location"], errors="coerce").fillna(0)

# Normalize X/Y to [0, 200] to match topology scale
df["_x"] = (df["_x"] - df["_x"].min()) / (df["_x"].max() - df["_x"].min() + 1e-9) * 200
df["_y"] = (df["_y"] - df["_y"].min()) / (df["_y"].max() - df["_y"].min() + 1e-9) * 200

# Energy columns
df["_residual_energy"] = pd.to_numeric(df["Residual_Energy"], errors="coerce").fillna(0)
df["_initial_energy"] = df["_residual_energy"]  # Using residual as initial for this dataset

# Additional WSN parameters
df["_transmission_distance"] = pd.to_numeric(df["Transmission_Distance"], errors="coerce").fillna(0)
df["_hop_count"] = pd.to_numeric(df["Hop_Count"], errors="coerce").fillna(0)
df["_pdr"] = pd.to_numeric(df["Packet_Delivery_Ratio"], errors="coerce").fillna(0)
df["_cluster_head"] = pd.to_numeric(df["Cluster_Head"], errors="coerce").fillna(0)
df["_leach_epoch"] = pd.to_numeric(df["LEACH_Epoch"], errors="coerce").fillna(0)
df["_load_balanced"] = pd.to_numeric(df["Load_Balanced_Score"], errors="coerce").fillna(0)

df["_energy"] = df["_residual_energy"]  # alias used downstream

# ---- Node role derivation ----
# Strategy: nodes in the top 20% of Residual_Energy are infrastructure (cluster head candidates)
#           Top 2 highest-energy nodes become sinks (origins)
#           All others are clients
energy_80pct = df["_residual_energy"].quantile(0.80)
df["_node_type"] = df["_residual_energy"].apply(
    lambda e: "infra" if e >= energy_80pct else "client"
)

print(f"Column mapping applied for wsn_qaga_dataset.csv")
print(f"  Infra nodes (top 20% energy): {(df['_node_type']=='infra').sum()}")
print(f"  Client nodes: {(df['_node_type']=='client').sum()}")

# =========================================================
# TOPOLOGY CONFIGURATION
# =========================================================
df = df.reset_index(drop=True)
num_nodes = len(df)
positions = list(zip(df["_x"], df["_y"]))

client_indices = df[df["_node_type"] == "client"].index.tolist()
infra_indices = df[df["_node_type"] == "infra"].index.tolist()

# Sort infra by energy and take top 2 as origins
infra_sorted = df.loc[infra_indices].sort_values('_residual_energy', ascending=False)
origin_indices = infra_sorted.head(2).index.tolist()
infra_nodes = infra_sorted.iloc[2:].index.tolist()

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
# PHASE 1: DATASET 
# =========================================================
print("\n--- Dataset ---")
summary_cols = ["_node_id", "_x", "_y", "_residual_energy",
                "_transmission_distance", "_hop_count", "_pdr", 
                "_cluster_head", "_leach_epoch", "_load_balanced", "_node_type"]
print(df[[c for c in summary_cols if c in df.columns]].describe(include="all").to_string())

df["_cluster_head_assigned"] = df.index.map(
    lambda i: client_to_ch.get(i, i if i in cluster_heads else -1))
df.to_csv("wsn_processed.csv", index=False)
print("\nPhase 1: Dataset processed and cluster assignments saved to 'wsn_processed.csv'.")

# =========================================================
# PHASE 1b: NETANIM XML GENERATION
# =========================================================
def build_netanim_xml(positions, cluster_heads, origin_indices, clusters,
                      client_to_ch, num_nodes, sim_duration=20.0,
                      output_path="wsn_2106.xml"):
    """
    Generates a NetAnim XML file matching the exact format of ns-3 AnimationInterface output.
    """
    origin_set = set(origin_indices)
    ch_set = set(cluster_heads)

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

    # -- Wireless packet events --
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
print("\nGenerating Node2Vec Embeddings for all 1500 nodes...")
node2vec = Node2Vec(G, dimensions=64, walk_length=10,
                    num_walks=10, workers=2, seed=7,
                    quiet=True)
model = node2vec.fit(window=5, min_count=1, batch_words=64)

nodes_list = list(G.nodes())
X = np.vstack([model.wv[str(n)] for n in nodes_list])

pd.DataFrame(X, index=nodes_list).to_csv("wsn_embeddings.csv")
print("Phase 2: Graph embeddings generated for all 1500 nodes.")

# =========================================================
# PHASE 3: GRAPHICAL VISUALIZATION
# =========================================================
print("\nGenerating visualization...")
fig, ax = plt.subplots(figsize=(14, 12), dpi=150)

cmap = plt.colormaps.get_cmap('tab20').resampled(max(len(cluster_heads), 1))
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

# 4. Draw edges 
nx.draw_networkx_edges(G, pos_dict, ax=ax,
                       alpha=0.15,
                       width=0.4,
                       edge_color='gray')

# 5. Legend
legend_handles = [
    mpatches.Patch(color='black', label='Sink / Origin node'),
    mpatches.Patch(color='red', label='Cluster Head'),
    mpatches.Patch(color='steelblue', label='Client node (colour = cluster)'),
]
ax.legend(handles=legend_handles, loc='upper left',
          fontsize=9, framealpha=0.85)

ax.set_title(
    "LEACH Clustering — WSN QAGA Dataset (ALL 1500 Nodes)\n"
    f"{num_nodes} nodes | {len(cluster_heads)} cluster heads | "
    f"{len(origin_indices)} sinks | "
    f"Role derived from Residual_Energy (top 20% = infra)",
    fontsize=11, pad=12)
ax.axis('off')

plt.tight_layout()
plt.savefig("clusters.png", dpi=150, bbox_inches="tight")
print("Phase 3: Visualization saved to 'clusters.png'.")

print("\n" + "="*50)
print("OUTPUT FILES GENERATED (ALL 1500 NODES):")
print("="*50)
print("  wsn_2106.xml        — NetAnim animation file")
print("  clusters.png        — cluster topology plot")
print("  wsn_embeddings.csv  — Node2Vec embeddings (64-dim)")
print("  wsn_processed.csv   — original data + cluster assignments")
print("="*50)
print(f"Total nodes processed: {num_nodes}")
print(f"Cluster heads: {len(cluster_heads)}")
print(f"Edges in graph: {G.number_of_edges()}")
print("="*50)

os._exit(0)
