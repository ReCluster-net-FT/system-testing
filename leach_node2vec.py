import random
import numpy as np
import pandas as pd
import networkx as nx
from node2vec import Node2Vec
from ns import ns
import matplotlib
matplotlib.use('Agg')  # Must be declared before importing pyplot
import matplotlib.pyplot as plt
# ----------------------------
# Initialization and Seeding
# ----------------------------
random.seed(7)
np.random.seed(7)

# ----------------------------
# Topology Configuration
# ----------------------------
NUM_CLIENTS = 50
NUM_ACCESS  = 8
NUM_POPS    = 6
NUM_REG     = 2
NUM_ORIGINS = 2

CLIENT_START  = 0
ACCESS_START  = NUM_CLIENTS
POP_START     = ACCESS_START + NUM_ACCESS
REG_START     = POP_START    + NUM_POPS
ORIGIN_START  = REG_START    + NUM_REG

# ----------------------------
# Node Allocation
# ----------------------------
nodes_clients = ns.NodeContainer(); nodes_clients.Create(NUM_CLIENTS)
nodes_access  = ns.NodeContainer(); nodes_access.Create(NUM_ACCESS)
nodes_pops    = ns.NodeContainer(); nodes_pops.Create(NUM_POPS)
nodes_reg     = ns.NodeContainer(); nodes_reg.Create(NUM_REG)
nodes_origins = ns.NodeContainer(); nodes_origins.Create(NUM_ORIGINS)

all_nodes = ns.NodeContainer()
all_nodes.Add(nodes_clients)
all_nodes.Add(nodes_access)
all_nodes.Add(nodes_pops)
all_nodes.Add(nodes_reg)
all_nodes.Add(nodes_origins)

num_nodes = all_nodes.GetN()

# ----------------------------
# Spatial Distribution
# ----------------------------
positions = [(random.uniform(0, 200), random.uniform(0, 200)) for _ in range(num_nodes)]

posAlloc = ns.ListPositionAllocator()
for x, y in positions:
    posAlloc.Add(ns.Vector(x, y, 0))

mobility = ns.MobilityHelper()
mobility.SetPositionAllocator(posAlloc)
mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel")
mobility.Install(all_nodes)

def get_pos(i):
    return positions[i]

# ----------------------------
# Physical and MAC Layer (802.11g)
# ----------------------------
wifi = ns.WifiHelper()
wifi.SetStandard(ns.WIFI_STANDARD_80211g)

channel = ns.YansWifiChannelHelper()
channel.SetPropagationDelay("ns3::ConstantSpeedPropagationDelayModel")
channel.AddPropagationLoss("ns3::LogDistancePropagationLossModel")

phy = ns.YansWifiPhyHelper()
phy.SetChannel(channel.Create())

mac = ns.WifiMacHelper()
mac.SetType("ns3::AdhocWifiMac")

devices = wifi.Install(phy, mac, all_nodes)

# ----------------------------
# Network Layer (OLSR)
# ----------------------------
olsr = ns.OlsrHelper()
routing = ns.Ipv4ListRoutingHelper()
routing.Add(olsr, 10)

stack = ns.InternetStackHelper()
stack.SetRoutingHelper(routing)
stack.Install(all_nodes)

# ----------------------------
# IPv4 Address Assignment
# ----------------------------
address = ns.Ipv4AddressHelper()
address.SetBase(ns.Ipv4Address("10.1.1.0"), ns.Ipv4Mask("255.255.255.0"))
interfaces = address.Assign(devices)

node_ip = {i: interfaces.GetAddress(i) for i in range(num_nodes)}

# =========================================================
# LEACH CLUSTERING ALGORITHM
# =========================================================
P = 0.3
infra_nodes = list(range(ACCESS_START, ORIGIN_START))

cluster_heads = [i for i in infra_nodes if random.random() < P]
if not cluster_heads:
    cluster_heads.append(random.choice(infra_nodes))

origin_indices = list(range(ORIGIN_START, num_nodes))

clusters = {ch: [] for ch in cluster_heads}

for i in range(num_nodes):
    if i in cluster_heads or i in origin_indices:
        continue
    best = min(cluster_heads,
               key=lambda ch: ((get_pos(i)[0] - get_pos(ch)[0])**2 + (get_pos(i)[1] - get_pos(ch)[1])**2))
    clusters[best].append(i)

client_to_ch = {}
for ch, members in clusters.items():
    for m in members:
        if m < NUM_CLIENTS:
            client_to_ch[m] = ch

# =========================================================
# TRAFFIC GENERATION (Memory Anchored)
# =========================================================
BASE_PORT = 9000
port = BASE_PORT

def next_port():
    global port
    port += 1
    return port

installed_servers = set()
persistent_helpers = []
persistent_apps = []

# Origin Sink Configuration
sink_port = 9
server = ns.UdpEchoServerHelper(sink_port)
persistent_helpers.append(server)
sink_app = server.Install(nodes_origins.Get(0))
sink_app.Start(ns.Seconds(1.0))
persistent_apps.append(sink_app)

# Cluster Head Server Configuration
ch_ports = {}
for ch in cluster_heads:
    p = next_port()
    ch_ports[ch] = p
    if ch not in installed_servers:
        srv = ns.UdpEchoServerHelper(p)
        persistent_helpers.append(srv)
        ch_app = srv.Install(all_nodes.Get(ch))
        ch_app.Start(ns.Seconds(1.0))
        persistent_apps.append(ch_app)
        installed_servers.add(ch)

# Client to Cluster Head Transmissions
for c, ch in client_to_ch.items():
    addr = ns.InetSocketAddress(node_ip[ch], ch_ports[ch]).ConvertTo()
    cli = ns.UdpEchoClientHelper(addr)
    persistent_helpers.append(cli)
    cli.SetAttribute("MaxPackets", ns.UintegerValue(5))
    cli.SetAttribute("Interval", ns.TimeValue(ns.Seconds(1.0)))
    cli.SetAttribute("PacketSize", ns.UintegerValue(512))
    cli_app = cli.Install(nodes_clients.Get(c))
    cli_app.Start(ns.Seconds(2.0))
    cli_app.Stop(ns.Seconds(18.0))
    persistent_apps.append(cli_app)

# Cluster Head to Origin Transmissions
for ch in cluster_heads:
    addr = ns.InetSocketAddress(node_ip[ORIGIN_START], sink_port).ConvertTo()
    cli = ns.UdpEchoClientHelper(addr)
    persistent_helpers.append(cli)
    cli.SetAttribute("MaxPackets", ns.UintegerValue(5))
    cli.SetAttribute("Interval", ns.TimeValue(ns.Seconds(1.0)))
    cli.SetAttribute("PacketSize", ns.UintegerValue(512))
    ch_origin_app = cli.Install(all_nodes.Get(ch))
    ch_origin_app.Start(ns.Seconds(6.0))
    ch_origin_app.Stop(ns.Seconds(18.0))
    persistent_apps.append(ch_origin_app)

# =========================================================
# GRAPH TOPOLOGY EXTRACTION
# =========================================================
G = nx.Graph()
for i in range(num_nodes):
    G.add_node(i)
for c, ch in client_to_ch.items():
    G.add_edge(c, ch)
for ch in cluster_heads:
    G.add_edge(ch, ORIGIN_START)

# =========================================================
# PHASE 1: SIMULATION EXECUTION & HEAP FLUSH
# =========================================================
anim = ns.AnimationInterface("wsn_2106.xml")

ns.Simulator.Stop(ns.Seconds(20.0))
ns.Simulator.Run()
ns.Simulator.Destroy()

# Explicitly sever Python references to prevent C++ double-free exceptions
persistent_helpers.clear()
persistent_apps.clear()

print("Phase 1: Stable simulation completed and memory cleared.")

# =========================================================
# PHASE 2: ISOLATED NODE2VEC EMBEDDING GENERATION
# =========================================================
node2vec = Node2Vec(G, dimensions=64, walk_length=20,
                    num_walks=200, workers=4, seed=7)
model = node2vec.fit(window=10, min_count=1, batch_words=256)

nodes_list = list(G.nodes())
X = np.vstack([model.wv[str(n)] for n in nodes_list])

pd.DataFrame(X).to_csv("wsn_embeddings.csv")
print("Phase 2: Graph embeddings generated securely.")

# =========================================================
# GRAPHICAL VISUALIZATION
# =========================================================
plt.figure(figsize=(10, 8))
colors = plt.colormaps.get_cmap('tab10').resampled(len(cluster_heads))

pos_dict = {i: positions[i] for i in range(num_nodes)}

for idx, (ch, members) in enumerate(clusters.items()):
    nx.draw_networkx_nodes(G, pos_dict, nodelist=members,
                           node_color=[colors(idx)], node_size=40)
    nx.draw_networkx_nodes(G, pos_dict, nodelist=[ch],
                           node_color=[colors(idx)], node_size=200, node_shape='s')

nx.draw_networkx_nodes(G, pos_dict, nodelist=origin_indices,
                       node_color='black', node_size=300, node_shape='*')

nx.draw_networkx_edges(G, pos_dict, alpha=0.3)

plt.title("Stable LEACH Clustering")
plt.axis('off')

# Save the file directly to disk
plt.savefig("clusters.png")

# TEXTBOOK FIX: Remove or comment out plt.show()
# plt.show() 

print("Phase 3: Visualization saved successfully. Script complete.")

import os

print("Finished successfully.")
os._exit(0)
