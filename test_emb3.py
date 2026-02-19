import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from node2vec import Node2Vec
from ns import ns

# ----------------------------
# Seeds for reproducibility
# ----------------------------
random.seed(7)
np.random.seed(7)

# ----------------------------
# 1) Define node counts
# ----------------------------
NUM_CLIENTS = 50
NUM_ACCESS  = 8
NUM_POPS    = 6
NUM_REG     = 2
NUM_ORIGINS = 2

# ----------------------------
# 2) Create ns-3 nodes
# ----------------------------
nodes_clients = ns.NodeContainer()
nodes_clients.Create(NUM_CLIENTS)
nodes_access = ns.NodeContainer()
nodes_access.Create(NUM_ACCESS)
nodes_pops = ns.NodeContainer()
nodes_pops.Create(NUM_POPS)
nodes_reg = ns.NodeContainer()
nodes_reg.Create(NUM_REG)
nodes_origins = ns.NodeContainer()
nodes_origins.Create(NUM_ORIGINS)

# ----------------------------
# 3) Mobility / positions for NetAnim
# ----------------------------
mobility = ns.MobilityHelper()
posAlloc = ns.ListPositionAllocator()
for i in range(NUM_CLIENTS): posAlloc.Add(ns.Vector(i*5, 0, 0))
for i in range(NUM_ACCESS):  posAlloc.Add(ns.Vector(i*10, 50, 0))
for i in range(NUM_POPS):    posAlloc.Add(ns.Vector(i*15, 100, 0))
for i in range(NUM_REG):     posAlloc.Add(ns.Vector(i*20, 150, 0))
for i in range(NUM_ORIGINS): posAlloc.Add(ns.Vector(i*25, 200, 0))

mobility.SetPositionAllocator(posAlloc)
mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel")
all_nodes = ns.NodeContainer()
all_nodes.Add(nodes_clients)
all_nodes.Add(nodes_access)
all_nodes.Add(nodes_pops)
all_nodes.Add(nodes_reg)
all_nodes.Add(nodes_origins)
mobility.Install(all_nodes)

# ----------------------------
# 4) Install Internet stack
# ----------------------------
stack = ns.InternetStackHelper()
stack.Install(all_nodes)

# ----------------------------
# 5) Helper to connect nodes
# ----------------------------
p2p = ns.PointToPointHelper()
def connect_nodes(u, v, bw_mbps, delay_ms):
    p2p.SetDeviceAttribute("DataRate", ns.StringValue(f"{bw_mbps}Mbps"))
    p2p.SetChannelAttribute("Delay", ns.StringValue(f"{delay_ms}ms"))
    nc = ns.NodeContainer()
    nc.Add(u)
    nc.Add(v)
    return p2p.Install(nc)

# ----------------------------
# 6) Connect nodes and track edges
# ----------------------------
edges_info = []

# Clients -> Access
client_interfaces = []
for i in range(NUM_CLIENTS):
    a_idx = random.randint(0, NUM_ACCESS-1)
    bw = random.choice([50, 100])
    delay = random.choice([10, 20, 30])
    devs = connect_nodes(nodes_clients.Get(i), nodes_access.Get(a_idx), bw, delay)
    client_interfaces.append(devs)
    edges_info.append((nodes_clients.Get(i).GetId(), nodes_access.Get(a_idx).GetId(), bw, delay))

# Access -> PoPs
access_interfaces = []
for i in range(NUM_ACCESS):
    for _ in range(2):
        p_idx = random.randint(0, NUM_POPS-1)
        bw = random.choice([200, 500])
        delay = random.choice([5, 8, 12])
        devs = connect_nodes(nodes_access.Get(i), nodes_pops.Get(p_idx), bw, delay)
        access_interfaces.append(devs)
        edges_info.append((nodes_access.Get(i).GetId(), nodes_pops.Get(p_idx).GetId(), bw, delay))

# PoPs -> Regional
pop_interfaces = []
for i in range(NUM_POPS):
    r_idx = random.randint(0, NUM_REG-1)
    bw = random.choice([1000, 2000])
    delay = random.choice([2, 4])
    devs = connect_nodes(nodes_pops.Get(i), nodes_reg.Get(r_idx), bw, delay)
    pop_interfaces.append(devs)
    edges_info.append((nodes_pops.Get(i).GetId(), nodes_reg.Get(r_idx).GetId(), bw, delay))

# Optional PoP-PoP peering (extra feature)
for _ in range(6):
    p1_idx, p2_idx = random.sample(range(NUM_POPS), 2)
    bw = random.choice([500, 1000])
    delay = random.choice([2, 3, 4])
    devs = connect_nodes(nodes_pops.Get(p1_idx), nodes_pops.Get(p2_idx), bw, delay)
    pop_interfaces.append(devs)
    edges_info.append((nodes_pops.Get(p1_idx).GetId(), nodes_pops.Get(p2_idx).GetId(), bw, delay))

# Regional -> Origins
reg_interfaces = []
for i in range(NUM_REG):
    for o_idx in range(NUM_ORIGINS):
        devs = connect_nodes(nodes_reg.Get(i), nodes_origins.Get(o_idx), 5000, 1)
        reg_interfaces.append(devs)
        edges_info.append((nodes_reg.Get(i).GetId(), nodes_origins.Get(o_idx).GetId(), 5000, 1))

# ----------------------------
# 7) Assign IP addresses
# ----------------------------
address = ns.Ipv4AddressHelper()
address.SetBase(ns.Ipv4Address("10.1.1.0"), ns.Ipv4Mask("255.255.255.0"))
all_devices = ns.NetDeviceContainer()
for dev_list in client_interfaces + access_interfaces + pop_interfaces + reg_interfaces:
    all_devices.Add(dev_list)
interfaces = address.Assign(all_devices)

# ----------------------------
# 8) Install UDP Echo apps
# ----------------------------
echoServer = ns.UdpEchoServerHelper(9)
serverApps = echoServer.Install(nodes_origins.Get(0))
serverApps.Start(ns.Seconds(1.0))
serverApps.Stop(ns.Seconds(20.0))

origin_index = interfaces.GetN() - NUM_ORIGINS
for i in range(NUM_CLIENTS):
    remoteAddr = ns.InetSocketAddress(interfaces.GetAddress(origin_index), 9).ConvertTo()
    echoClient = ns.UdpEchoClientHelper(remoteAddr)
    echoClient.SetAttribute("MaxPackets", ns.UintegerValue(5))
    echoClient.SetAttribute("Interval", ns.TimeValue(ns.Seconds(1.0)))
    echoClient.SetAttribute("PacketSize", ns.UintegerValue(1024))
    clientApps = echoClient.Install(nodes_clients.Get(i))
    clientApps.Start(ns.Seconds(2.0))
    clientApps.Stop(ns.Seconds(19.0))

# ----------------------------
# 9) Build NetworkX graph for Node2Vec
# ----------------------------
G = nx.Graph()
node_roles = {}

# Add nodes with roles
for i in range(NUM_CLIENTS):  G.add_node(i); node_roles[i] = "client"
for i in range(NUM_ACCESS):   G.add_node(NUM_CLIENTS+i); node_roles[NUM_CLIENTS+i] = "access"
for i in range(NUM_POPS):     G.add_node(NUM_CLIENTS+NUM_ACCESS+i); node_roles[NUM_CLIENTS+NUM_ACCESS+i] = "pop"
for i in range(NUM_REG):      G.add_node(NUM_CLIENTS+NUM_ACCESS+NUM_POPS+i); node_roles[NUM_CLIENTS+NUM_ACCESS+NUM_POPS+i] = "regional"
for i in range(NUM_ORIGINS):  G.add_node(NUM_CLIENTS+NUM_ACCESS+NUM_POPS+NUM_REG+i); node_roles[NUM_CLIENTS+NUM_ACCESS+NUM_POPS+NUM_REG+i] = "origin"

# Add edges
for u, v, bw, delay in edges_info:
    G.add_edge(u, v, bw_mbps=bw, delay_ms=delay)

# ----------------------------
# 10) Node2Vec embeddings
# ----------------------------
node2vec = Node2Vec(G, dimensions=64, walk_length=20, num_walks=200, workers=4, seed=7)
model = node2vec.fit(window=10, min_count=1, batch_words=256)

nodes_list = list(G.nodes())
X = np.vstack([model.wv[str(n)] for n in nodes_list])

# Node embeddings CSV
emb_df = pd.DataFrame(X, index=nodes_list)
emb_df["role"] = [node_roles[n] for n in nodes_list]
emb_df.to_csv("cdn_node_embeddings.csv")

# Graph-level embedding (mean pooling)
graph_embedding = X.mean(axis=0)
pd.DataFrame([graph_embedding]).to_csv("cdn_graph_embedding.csv", index=False)

# Edge metadata CSV
edge_rows = []
for u, v, d in G.edges(data=True):
    edge_rows.append([u, v, d["bw_mbps"], d["delay_ms"]])
pd.DataFrame(edge_rows, columns=["src","dst","bw_mbps","delay_ms"]).to_csv("cdn_edges.csv", index=False)

# ----------------------------
# 11) PCA visualization
# ----------------------------
from sklearn.decomposition import PCA
pca = PCA(n_components=2, random_state=7)
X2 = pca.fit_transform(X)
roles = [node_roles[n] for n in nodes_list]

role_to_marker = {"client":"o","access":"s","pop":"^","regional":"D","origin":"X"}
plt.figure(figsize=(9,6))
for role in sorted(set(roles)):
    idx = [i for i, r in enumerate(roles) if r==role]
    plt.scatter(X2[idx,0], X2[idx,1], label=role, marker=role_to_marker.get(role,"o"))
plt.title("CDN Node Embeddings (Node2Vec) - PCA Projection")
plt.xlabel("PCA-1")
plt.ylabel("PCA-2")
plt.legend()
plt.tight_layout()
plt.savefig("cdn_embeddings_pca.png", dpi=200)
plt.show()

# ----------------------------
# 12) Run NetAnim simulation
# ----------------------------
anim = ns.AnimationInterface("cdn_sim.xml")
ns.Simulator.Stop(ns.Seconds(20.0))
ns.Simulator.Run()
ns.Simulator.Destroy()

print("Simulation complete. NetAnim file: cdn_sim.xml")
print("Saved CSVs: cdn_node_embeddings.csv, cdn_graph_embedding.csv, cdn_edges.csv")
print("Saved PCA plot: cdn_embeddings_pca.png")
