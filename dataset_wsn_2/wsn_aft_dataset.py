import kagglehub
import os
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random

from sklearn.neighbors import NearestNeighbors
from node2vec import Node2Vec

# =================================================
# STEP 1: LOAD DATASET
# =================================================
path = kagglehub.dataset_download("ziya07/wsn-aft-dataset")

csv_file = None
for root, _, files in os.walk(path):
    for f in files:
        if f.endswith(".csv"):
            csv_file = os.path.join(root, f)

df = pd.read_csv(csv_file)
print("Dataset loaded:", df.shape)

# =================================================
# STEP 2: PREPROCESS
# =================================================
numeric_df = df.select_dtypes(include=[np.number])

norm_df = (
    (numeric_df - numeric_df.min()) /
    (numeric_df.max() - numeric_df.min() + 1e-9)
)

# =================================================
# STEP 3: BUILD KNN GRAPH (WSN TOPOLOGY)
# =================================================
k = 5

nbrs = NearestNeighbors(n_neighbors=k).fit(norm_df.values)
distances, indices = nbrs.kneighbors(norm_df.values)

G = nx.Graph()

for i in range(len(norm_df)):
    G.add_node(i)

for i in range(len(norm_df)):
    for j in indices[i]:
        if i != j:
            G.add_edge(i, j)

print("Graph created:", G.number_of_nodes(), "nodes,", G.number_of_edges(), "edges")

# =================================================
# STEP 4: DEFINE NODE ROLES (FOR VISUALIZATION)
# =================================================
nodes = list(G.nodes())

sink = nodes[0]
cluster_heads = set(random.sample(nodes[1:], k=max(3, len(nodes)//25)))
sensors = set(nodes) - cluster_heads - {sink}

# =================================================
# STEP 5: GRAPH IMAGE (STRUCTURED VISUALIZATION)
# =================================================
pos = nx.spring_layout(G, seed=42)

plt.figure(figsize=(10, 7))

# Sensor nodes
nx.draw_networkx_nodes(
    G, pos,
    nodelist=list(sensors),
    node_color="skyblue",
    node_size=20,
    label="Sensor Nodes"
)

# Cluster heads
nx.draw_networkx_nodes(
    G, pos,
    nodelist=list(cluster_heads),
    node_color="red",
    node_size=80,
    node_shape="s",
    label="Cluster Heads"
)

# Sink node
nx.draw_networkx_nodes(
    G, pos,
    nodelist=[sink],
    node_color="gold",
    node_size=250,
    node_shape="*",
    label="Sink Node"
)

nx.draw_networkx_edges(G, pos, alpha=0.3)

plt.title("WSN Graph Topology (Dataset Driven)")
plt.axis("off")
plt.legend()
plt.savefig("wsn_graph.png", dpi=300)
plt.close()

print("Graph image saved → wsn_graph.png")

# =================================================
# STEP 6: GRAPH EMBEDDINGS (NODE2VEC)
# =================================================
node2vec = Node2Vec(
    G,
    dimensions=64,
    walk_length=20,
    num_walks=50,
    workers=2,
    seed=7
)

model = node2vec.fit(window=10, min_count=1)

nodes = list(G.nodes())
embeddings = np.array([model.wv[str(n)] for n in nodes])

pd.DataFrame(embeddings).to_csv("wsn_embeddings.csv", index=False)

print("Embeddings saved → wsn_embeddings.csv")

# =================================================
# STEP 7: NETANIM XML (STRUCTURED LAYOUT)
# =================================================
def generate_xml(filename="wsn_animation.xml"):
    pos = nx.spring_layout(G, seed=42)

    with open(filename, "w") as f:
        f.write("<ns2>\n")

        for i in G.nodes():
            x, y = pos[i]

            # scale to NetAnim space
            x = int((x + 1) * 500)
            y = int((y + 1) * 500)

            f.write(f'<node id="{i}" x="{x}" y="{y}" />\n')

        for u, v in G.edges():
            f.write(f'<link from="{u}" to="{v}" />\n')

        f.write("</ns2>")

generate_xml()

print("NetAnim XML saved → wsn_animation.xml")

# =================================================
# DONE
# =================================================
print("\nALL OUTPUTS GENERATED SUCCESSFULLY")
