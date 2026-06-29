# Thermax-PID-Automation
AI-assisted PID Automation and Machine Learning Framework for Biomass Pyrolysis Process Optimization
## Objective
This project develops a machine learning and optimization workflow for chemical process automation, focusing on P&ID automation, biomass pyrolysis, product yield prediction, and multi-objective optimization.Pyrolysis is a thermochemical conversion process used for the production of biofuels such as biochar, bio-oil, and syngas. The quality and yield of products obtained during biomass pyrolysis are dependent on several factors, such as biomass composition (proximate and ultimate analysis) and reaction conditions (temperature, heating rate, time). Therefore, it is imperative to understand the relationship between biomass composition, reaction conditions, and product yield during biomass pyrolysis.

## Key Features
- Data preprocessing for biomass/process datasets
- ANN-based surrogate model development
- Product yield prediction
- NSGA-II based multi-objective optimization
- Pareto front generation
- Model evaluation using parity plots and error metrics

## Folder Structure
- `notebooks/` - Google Colab and Jupyter notebooks
- `data/` - Raw and processed datasets
- `src/` - Python source codes
- `models/` - Trained ML models and scalers
- `results/` - Prediction and optimization outputs
- `figures/` - Graphs and plots
- `docs/` - Reports and documentation

## Tools Used
Python, Google Colab, PyTorch, Scikit-learn, Pymoo, Pandas, NumPy, Matplotlib

## Status
Under development.

## Note
This repository is private and intended for research and internal development purposes only.
import numpy as np
### Code
def find_tag_xy(df_text, tag):
    pat = rf"\b{tag.replace('-','[-\\s]?')}\b"
    hits = df_text[df_text["text"].astype(str).str.contains(pat, case=False, regex=True)]
    if len(hits) == 0:
        raise ValueError(f"Tag not found: {tag}")
    return hits.iloc[0][["x","y"]].values.astype(float)

def nearest_insert_index(df_main, xy):
    pts = df_main[["x","y"]].values.astype(float)
    d = np.sqrt(((pts - xy)**2).sum(axis=1))
    return df_main.iloc[np.argmin(d)].name

reactor_tags = ["R-100","R-200","R-300"]
reactor_nodes = {}

for t in reactor_tags:
    xy = find_tag_xy(df_text, t)
    ins_idx = nearest_insert_index(df_main, xy)
    reactor_nodes[t] = f"ins_{ins_idx}"
    # stamp role
    G.nodes[reactor_nodes[t]]["unit_tag"] = t
    G.nodes[reactor_nodes[t]]["unit_type"] = "reactor"

G.nodes[reactor_nodes["R-300"]]["reactor_stage"] = "catalytic_fixed_bed"
G.nodes[reactor_nodes["R-100"]]["reactor_stage"] = "acidification"
G.nodes[reactor_nodes["R-200"]]["reactor_stage"] = "acidification"

reactor_nodes
!pip -q install ezdxf shapely networkx pandas matplotlib
!pip -q install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
!pip -q install torch-geometric -q
import ezdxf
import pandas as pd
import numpy as np

doc = ezdxf.readfile(DXF_FILE)
msp = doc.modelspace()

# 1A) TEXT / MTEXT
texts = []
for e in msp.query("TEXT MTEXT"):
    if e.dxftype() == "TEXT":
        s = e.dxf.text
        x, y = e.dxf.insert.x, e.dxf.insert.y
        layer = e.dxf.layer
    else:
        s = e.text
        x, y = e.dxf.insert.x, e.dxf.insert.y
        layer = e.dxf.layer
    texts.append({"text": str(s).strip(), "x": x, "y": y, "layer": layer})

df_text = pd.DataFrame(texts)

# 1B) INSERT (Blocks: valves/instruments/equipment symbols)
inserts = []
for e in msp.query("INSERT"):
    inserts.append({
        "block": e.dxf.name,
        "x": e.dxf.insert.x,
        "y": e.dxf.insert.y,
        "layer": e.dxf.layer,
        "rotation": float(getattr(e.dxf, "rotation", 0.0)),
        "scale_x": float(getattr(e.dxf, "xscale", 1.0)),
        "scale_y": float(getattr(e.dxf, "yscale", 1.0)),
    })
df_ins = pd.DataFrame(inserts)

# 1C) Lines and polylines (pipes)
pipes = []
for e in msp.query("LINE LWPOLYLINE"):
    if e.dxftype() == "LINE":
        p1 = (e.dxf.start.x, e.dxf.start.y)
        p2 = (e.dxf.end.x, e.dxf.end.y)
        pts = [p1, p2]
    else:
        # LWPOLYLINE points
        pts = [(p[0], p[1]) for p in e.get_points("xy")]
    pipes.append({"type": e.dxftype(), "layer": e.dxf.layer, "pts": pts})

df_pipe = pd.DataFrame(pipes)

print("TEXT:", len(df_text), "INSERT:", len(df_ins), "PIPES:", len(df_pipe))
df_ins.head()
TEXT: 1123 INSERT: 153 PIPES: 1193
block	x	y	layer	rotation	scale_x	scale_y
0	FTGRETRE	8339.640165	599.716281	0	0.0	1.0	1.0
1	FTGRETRE	36766.244310	-2546.664658	0	0.0	1.0	1.0
2	FTGRETRE	-10791.121490	23219.614819	0	0.0	1.0	1.0
3	FTGRETRE	18459.698260	48777.148305	0	0.0	1.0	1.0
4	FTGRETRE	67163.689461	-22130.767509	0	0.0	1.0	1.0
from shapely.geometry import LineString, Point
import networkx as nx

# Convert pipes to shapely lines
pipe_lines = []
for i,r in df_pipe.iterrows():
    if len(r["pts"]) >= 2:
        pipe_lines.append((i, r["layer"], LineString(r["pts"])))

# Create graph nodes from inserts
G = nx.Graph()
for i,r in df_ins.iterrows():
    G.add_node(
        f"ins_{i}",
        kind="insert",
        block=r["block"],
        layer=r["layer"],
        x=float(r["x"]),
        y=float(r["y"])
    )

# Helper: distance from node to pipe
def node_to_pipe_dist(node_id, pipe_geom):
    x, y = G.nodes[node_id]["x"], G.nodes[node_id]["y"]
    return pipe_geom.distance(Point(x, y))

# Connect nodes that lie close to the same pipe segment
DIST_THRESH = 5.0  # you may tune depending on drawing units
for pid, layer, geom in pipe_lines:
    near = []
    for n in G.nodes:
        d = node_to_pipe_dist(n, geom)
        if d <= DIST_THRESH:
            near.append((n, d))
    near = sorted(near, key=lambda t: t[1])

    # connect near nodes along that pipe (pairwise)
    for a in range(len(near)-1):
        u, du = near[a]
        v, dv = near[a+1]
        if u != v:
            G.add_edge(u, v, edge_type="process_pipe", pipe_id=int(pid), pipe_layer=layer)

print("Graph nodes:", G.number_of_nodes(), "edges:", G.number_of_edges())
Graph nodes: 153 edges: 50
import matplotlib.pyplot as plt

plt.figure(figsize=(8,8))
# plot pipes
for _,_,geom in pipe_lines[:500]:  # cap if huge
    xs, ys = geom.xy
    plt.plot(xs, ys, linewidth=0.5)

# plot nodes
xs = [G.nodes[n]["x"] for n in G.nodes]
ys = [G.nodes[n]["y"] for n in G.nodes]
plt.scatter(xs, ys, s=8)

plt.title("DXF P&ID: Pipes + Insert Nodes (sanity check)")
plt.axis("equal")
plt.show()
import matplotlib.pyplot as plt

plt.figure(figsize=(8,8))
# plot pipes
for _,_,geom in pipe_lines[:500]:  # cap if huge
    xs, ys = geom.xy
    plt.plot(xs, ys, linewidth=0.5)

# plot nodes
xs = [G.nodes[n]["x"] for n in G.nodes]
ys = [G.nodes[n]["y"] for n in G.nodes]
plt.scatter(xs, ys, s=8)

plt.title("DXF P&ID: Pipes + Insert Nodes (sanity check)")
plt.axis("equal")
plt.show()
import matplotlib.pyplot as plt

plt.figure(figsize=(8,8))
# plot pipes
for _,_,geom in pipe_lines[:500]:  # cap if huge
    xs, ys = geom.xy
    plt.plot(xs, ys, linewidth=0.5)

# plot nodes
xs = [G.nodes[n]["x"] for n in G.nodes]
ys = [G.nodes[n]["y"] for n in G.nodes]
plt.scatter(xs, ys, s=8)

plt.title("DXF P&ID: Pipes + Insert Nodes (sanity check)")
plt.axis("equal")
plt.show()
import matplotlib.pyplot as plt

plt.figure(figsize=(8,8))
# plot pipes
for _,_,geom in pipe_lines[:500]:  # cap if huge
    xs, ys = geom.xy
    plt.plot(xs, ys, linewidth=0.5)

# plot nodes
xs = [G.nodes[n]["x"] for n in G.nodes]
ys = [G.nodes[n]["y"] for n in G.nodes]
plt.scatter(xs, ys, s=8)

plt.title("DXF P&ID: Pipes + Insert Nodes (sanity check)")
plt.axis("equal")
plt.show()
<img width="683" height="700" alt="image" src="https://github.com/user-attachments/assets/28534c81-e113-4774-ac98-9071220aad12" />
xs = df_ins["x"].values
ys = df_ins["y"].values

print("X range:", xs.min(), xs.max())
print("Y range:", ys.min(), ys.max())
X range: -49365.18283157935 4060125.101519575
Y range: -27729.92177225067 133592.4991103569
import numpy as np

# Robust center
mx, my = np.median(xs), np.median(ys)

# Distance from center
dist = np.sqrt((xs-mx)**2 + (ys-my)**2)

# Keep only nodes within reasonable radius
KEEP_RADIUS = np.percentile(dist, 95)  # keeps 95% of symbols
mask = dist <= KEEP_RADIUS

df_ins_clean = df_ins[mask].copy()
print("Before:", len(df_ins), "After:", len(df_ins_clean))
Before: 153 After: 145
import matplotlib.pyplot as plt

plt.figure(figsize=(6,6))
plt.scatter(df_ins_clean["x_norm"], df_ins_clean["y_norm"], s=10)
plt.title("Normalized P&ID Insert Nodes (Sanity Check)")
plt.axis("equal")
plt.show()
from sklearn.cluster import DBSCAN
import numpy as np

coords = df_ins_clean[["x_norm", "y_norm"]].values

# DBSCAN groups spatial clusters
clustering = DBSCAN(eps=0.6, min_samples=5).fit(coords)
df_ins_clean["cluster"] = clustering.labels_

df_ins_clean["cluster"].value_counts()
 

    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }


  
    
      
      count
    
    
      cluster
      
    
  
  
    
      0
      142
    
    
      -1
      3
    
  

dtype: int64
 





PID_Flow



S

START



A

P&ID Source (DWG)
• Original engineering drawing
• Human readable



S->A





B

Convert DWG → DXF
• Export CAD geometry
• Entities: LINE/POLYLINE/INSERT/TEXT



A->B





C

Parse DXF (ezdxf)
Extract:
• Pipes (LINE/POLYLINE)
• Equip symbols (INSERT)
• Tags (TEXT)
• Layers = process/signal



B->C





D

Build P&ID JSON
nodes[] + edges[]
• node: id,type,(x,y)
• edge: from,to,type



C->D





E

Normalize + Clean Geometry
• remove outliers
• rescale coords



D->E





F

KNN Connectivity (sanity)
• local edges only
• remove long-range false links



E->F





G

ML-ready Graph
G=(V,E)
• nodes: equipment/reactors/instruments
• edges: process pipes + signal lines



F->G





H

GNN + pyMOO
• learn KPI surrogate
• Pareto-optimal setpoints



G->H

T

STOP



H->T
import numpy as np

def find_tag_xy(df_text, tag):
    pat = rf"\b{tag.replace('-','[-\s]?')}\b"
    hits = df_text[df_text["text"].astype(str).str.contains(pat, case=False, regex=True)]
    if len(hits) == 0:
        raise ValueError(f"Tag not found: {tag}")
    return hits.iloc[0][["x","y"]].values.astype(float)

def nearest_insert_index(df_main, xy):
    pts = df_main[["x","y"]].values.astype(float)
    d = np.sqrt(((pts - xy)**2).sum(axis=1))
    return df_main.iloc[np.argmin(d)].name

reactor_tags = ["R-100","R-200","R-300"]
reactor_nodes = {}

for t in reactor_tags:
    xy = find_tag_xy(df_text, t)
    ins_idx = nearest_insert_index(df_main, xy)
    reactor_nodes[t] = f"ins_{ins_idx}"
    # stamp role
    G.nodes[reactor_nodes[t]]["unit_tag"] = t
    G.nodes[reactor_nodes[t]]["unit_type"] = "reactor"

G.nodes[reactor_nodes["R-300"]]["reactor_stage"] = "catalytic_fixed_bed"
G.nodes[reactor_nodes["R-100"]]["reactor_stage"] = "acidification"
G.nodes[reactor_nodes["R-200"]]["reactor_stage"] = "acidification"

reactor_nodes
import torch, numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from torch_geometric.nn import SAGEConv

# --- index mapping
node_list = list(G.nodes())
idx = {n:i for i,n in enumerate(node_list)}

r100 = idx[reactor_nodes["R-100"]]
r200 = idx[reactor_nodes["R-200"]]
r300 = idx[reactor_nodes["R-300"]]

def block_hash(s, k=16):
    h = abs(hash(s)) % (2**31)
    v = np.zeros(k, dtype=np.float32)
    v[h % k] = 1.0
    return v

# base node features: (x,y) + hashed block name
base = []
for n in node_list:
    base.append(np.concatenate([
        np.array([G.nodes[n]["x"], G.nodes[n]["y"]], np.float32),
        block_hash(G.nodes[n].get("block","UNK"), 16)
    ]))
base = torch.tensor(np.vstack(base), dtype=torch.float32)

edges = [[idx[u], idx[v]] for u,v in G.edges()]
edge_index = to_undirected(torch.tensor(edges, dtype=torch.long).t().contiguous())
data_static = Data(x=base, edge_index=edge_index)

# Dynamic inputs (example):
# R100: [T,P,acid_dose,tau]
# R200: [T,P,acid_dose,tau]
# R300: [T,P,WHSV,H2_feed,catalyst_family_id,activity]
# catalyst_family_id: encode as 0..N (NiMo,CoMo,Ni2P,Ru,...)
DYN_DIM = 4 + 4 + 6

def make_graph_sample(x_dyn):
    dyn = torch.zeros((data_static.num_nodes, DYN_DIM), dtype=torch.float32)
    dyn[r100, 0:4] = torch.tensor(x_dyn[0:4])
    dyn[r200, 4:8] = torch.tensor(x_dyn[4:8])
    dyn[r300, 8:14] = torch.tensor(x_dyn[8:14])
    return Data(x=torch.cat([data_static.x, dyn], dim=1), edge_index=data_static.edge_index)

class BioOilKPI_GNN(nn.Module):
    def __init__(self, in_dim, hid=96, out_dim=5):
        super().__init__()
        self.c1 = SAGEConv(in_dim, hid)
        self.c2 = SAGEConv(hid, hid)
        self.c3 = SAGEConv(hid, hid)
        self.head = nn.Sequential(nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, out_dim))
    def forward(self, g):
        h = F.relu(self.c1(g.x, g.edge_index))
        h = F.relu(self.c2(h, g.edge_index))
        h = F.relu(self.c3(h, g.edge_index))
        hg = h.mean(dim=0)               # plant-level KPI readout
        return self.head(hg)

model = BioOilKPI_GNN(in_dim=data_static.x.shape[1]+DYN_DIM, hid=96, out_dim=5)
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# -----------------------------
# 1) Fixed 5-node "P&ID" graph
# -----------------------------
N_NODES = 5
edges = [(0,1),(1,2),(2,3),(3,4)]
A = np.zeros((N_NODES,N_NODES), dtype=np.float32)
for u,v in edges:
    A[u,v] = 1.0; A[v,u] = 1.0
A += np.eye(N_NODES, dtype=np.float32)  # self loops
An = A / (A.sum(axis=1, keepdims=True) + 1e-6)        # row-normalized adjacency
An_t = torch.tensor(An, dtype=torch.float32)

# node type one-hot: Feed, Acid, Acid, 13X-bed, Product
node_types = np.array([0,1,1,2,3], dtype=np.int64)
num_types = 4
type_onehot = np.eye(num_types, dtype=np.float32)[node_types]  # (5,4)
BASE = type_onehot.shape[1]

# -----------------------------------------
# 2) Toy "truth" function (13X dries water)
# -----------------------------------------
def clamp(x, lo, hi): return np.clip(x, lo, hi)

def truth(u):
    """
    u = [T1,P1,acid1,tau1,  T2,P2,acid2,tau2,  T3,P3,WHSV,H2, zfrac, mload, act,  Fkgph,O0,W0,V0]
    outputs = [Yield, Moisture, Oxygenates, Viscosity, Stability, Q_kW, dP_bar]
    """
    (T1,P1,a1,tau1, T2,P2,a2,tau2, T3,P3,WHSV,H2, zfrac,mload,act, Fkgph,O0,W0,V0) = u

    sev1 = (a1*tau1)*(T1/480.0)**0.6
    sev2 = (a2*tau2)*(T2/500.0)**0.6

    Oa = O0*(1-0.10*clamp(sev1,0,1.5))*(1-0.12*clamp(sev2,0,1.5))
    Wa = W0*(1-0.06*clamp(sev1,0,1.5))*(1-0.06*clamp(sev2,0,1.5))
    Va = V0*(1-0.04*clamp(sev1,0,1.5))*(1-0.05*clamp(sev2,0,1.5))
    Ya = 0.62*(1-0.01*clamp(sev1,0,2))*(1-0.01*clamp(sev2,0,2))

    sev3 = (T3/650.0)**1.2*(P3/30.0)**0.5*(H2**0.6)*act/(clamp(WHSV,0.2,10.0))

    # 13X effect: strong water adsorption (higher at lower T); modest oxygenate trapping
    ads_water = zfrac*1.2*(650.0/max(T3,1e-3))**0.7
    ads_oxyg  = zfrac*0.25

    # "modified 13X" proxy improves deoxygenation slightly (metal loading)
    mod_boost = 1 + 0.15*clamp(mload,0,1.0)

    Oout = Oa*(1-0.45*clamp(sev3*mod_boost,0,1.5))*(1-ads_oxyg)
    Wout = Wa*(1-0.18*clamp(sev3,0,1.5))*(1-0.55*clamp(ads_water,0,0.8))
    Vout = Va*(1-0.30*clamp(sev3,0,1.5))*(1-0.10*clamp(ads_oxyg,0,0.3))

    over = max(0.0, sev3-1.0)           # over-severity -> cracking -> yield loss
    Yout = Ya*(1-0.12*over)

    stability = 0.3 + 1.2*(O0-Oout) + 0.8*(W0-Wout)
    stability = clamp(stability, 0.0, 2.5)

    Q  = (0.8*sev3*Fkgph*10) + (0.02*(T3-300)*Fkgph)
    dP = 0.02*Fkgph*(P3/30.0) + 0.01*Fkgph*(1/max(WHSV,0.2))

    return np.array([Yout,Wout,Oout,Vout,stability,Q,dP], dtype=np.float32)

# -----------------------------
# 3) Generate dataset (POC scale)
# -----------------------------
rng = np.random.default_rng(7)
N = 2000
X = np.zeros((N,19), dtype=np.float32)
Y = np.zeros((N,7), dtype=np.float32)

for i in range(N):
    T1=rng.uniform(430,520); P1=rng.uniform(5,20);  a1=rng.uniform(0.1,1.0); tau1=rng.uniform(0.5,3.0)
    T2=rng.uniform(440,540); P2=rng.uniform(5,25);  a2=rng.uniform(0.1,1.0); tau2=rng.uniform(0.5,3.0)
    T3=rng.uniform(560,720); P3=rng.uniform(10,60); WHSV=rng.uniform(0.5,5.0); H2=rng.uniform(0.5,5.0)
    zfrac=rng.uniform(0.2,1.0); mload=rng.uniform(0.0,1.0); act=rng.uniform(0.6,1.0)
    Fkgph=rng.uniform(0.2,1.0)                  # around 10 kg/day ≈ 0.417 kg/h
    O0=rng.uniform(0.25,0.45); W0=rng.uniform(0.15,0.35); V0=rng.uniform(50,250)

    u=np.array([T1,P1,a1,tau1,T2,P2,a2,tau2,T3,P3,WHSV,H2,zfrac,mload,act,Fkgph,O0,W0,V0],dtype=np.float32)
    X[i]=u; Y[i]=truth(u)

# normalize inputs
X_mean, X_std = X.mean(axis=0), X.std(axis=0)+1e-6
Xn = (X - X_mean)/X_std

# build node features per sample (5 nodes × (type + dynamic))
DYN = 19
Fdim = BASE + DYN

def build_node_features(xrow):
    dyn = np.zeros((N_NODES,DYN), dtype=np.float32)
    dyn[0,-4:] = xrow[-4:]      # Feed: [Fkgph, O0, W0, V0]
    dyn[1,0:4] = xrow[0:4]      # R-100
    dyn[2,4:8] = xrow[4:8]      # R-200
    dyn[3,8:15]= xrow[8:15]     # R-300 (13X bed inputs)
    return np.concatenate([type_onehot, dyn], axis=1)

X_nodes = np.stack([build_node_features(Xn[i]) for i in range(N)], axis=0)  # (N,5,Fdim)

# train/test split
tr, te = train_test_split(np.arange(N), test_size=0.2, random_state=0)
Xtr = torch.tensor(X_nodes[tr], dtype=torch.float32)
Ytr = torch.tensor(Y[tr], dtype=torch.float32)
Xte = torch.tensor(X_nodes[te], dtype=torch.float32)
Yte = torch.tensor(Y[te], dtype=torch.float32)

# -----------------------------
# 4) Message-passing GNN model
# -----------------------------
class SimpleGNN(nn.Module):
    def __init__(self, in_dim, hid=96, out_dim=7):
        super().__init__()
        self.l1 = nn.Linear(in_dim, hid)
        self.l2 = nn.Linear(hid, hid)
        self.l3 = nn.Linear(hid, hid)
        self.out= nn.Linear(hid, out_dim)

    def forward(self, x_nodes, A_norm):
        # x_nodes: (B, N, F)
        h = F.relu(self.l1(x_nodes))
        h = torch.matmul(A_norm, h)      # message passing 1
        h = F.relu(self.l2(h))
        h = torch.matmul(A_norm, h)      # message passing 2
        h = F.relu(self.l3(h))
        hg = h.mean(dim=1)               # graph pooling
        return self.out(hg)

model = SimpleGNN(Fdim, hid=96, out_dim=7)
opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)

# training
batch = 128
for epoch in range(1, 26):
    model.train()
    order = torch.randperm(Xtr.shape[0])
    for i in range(0, Xtr.shape[0], batch):
        sel = order[i:i+batch]
        pred = model(Xtr[sel], An_t)
        loss = F.mse_loss(pred, Ytr[sel])
        opt.zero_grad(); loss.backward(); opt.step()

    if epoch % 5 == 0:
        model.eval()
        with torch.no_grad():
            P = model(Xte, An_t)
        rmse = torch.sqrt(torch.mean((P - Yte)**2, dim=0)).numpy()
        print(f"epoch {epoch:02d} test_RMSE:", np.round(rmse, 4))

# report KPIs
names = ["Yield","Moisture","Oxygenates","Viscosity","Stability","Q_kW","dP_bar"]
model.eval()
with torch.no_grad():
    P = model(Xte, An_t).numpy()
Yte_np = Yte.numpy()

for j,nm in enumerate(names):
    print(nm, "R2=", round(r2_score(Yte_np[:,j], P[:,j]), 4),
              "MAE=", round(mean_absolute_error(Yte_np[:,j], P[:,j]), 4))

# SCADA-style what-if: change only R-300 temperature (T3) and see outputs change
def predict_one(u_raw):
    x = (u_raw - X_mean)/X_std
    xnodes = torch.tensor(build_node_features(x)[None,:,:], dtype=torch.float32)
    with torch.no_grad():
        out = model(xnodes, An_t).numpy().ravel()
    return dict(zip(names, out))

u0 = X[0].copy()
base = predict_one(u0)
u1 = u0.copy(); u1[8] = 700.0   # T3 index in vector = 8
hot  = predict_one(u1)

print("\nBase:", base)
print("R-300 T up:", hot)
epoch 05 test_RMSE: [2.56600e-01 8.24000e-02 6.06000e-02 4.15745e+01 2.14500e-01 4.60180e+00
 2.56000e-02]
epoch 10 test_RMSE: [ 0.112   0.0437  0.0571 28.9114  0.1774  4.7977  0.0303]
epoch 15 test_RMSE: [ 0.1455  0.0443  0.0571 20.142   0.2085  4.9847  0.0325]
epoch 20 test_RMSE: [ 0.071   0.0496  0.0599 15.0776  0.1589  5.1392  0.0301]
epoch 25 test_RMSE: [ 0.0679  0.0395  0.0578 13.359   0.1344  4.8137  0.014 ]
Yield R2= -3.826 MAE= 0.0539
Moisture R2= -0.0409 MAE= 0.0324
Oxygenates R2= 0.0267 MAE= 0.0468
Viscosity R2= 0.9156 MAE= 10.1161
Stability R2= -1.5691 MAE= 0.1064
Q_kW R2= -0.3528 MAE= 3.6229
dP_bar R2= -1.5453 MAE= 0.0115

Base: {'Yield': np.float32(0.6626199), 'Moisture': np.float32(0.14777331), 'Oxygenates': np.float32(0.17960395), 'Viscosity': np.float32(121.85239), 'Stability': np.float32(0.6617842), 'Q_kW': np.float32(7.4516315), 'dP_bar': np.float32(0.0070812725)}
R-300 T up: {'Yield': np.float32(0.6616269), 'Moisture': np.float32(0.14702849), 'Oxygenates': np.float32(0.17959537), 'Viscosity': np.float32(122.35664), 'Stability': np.float32(0.65987396), 'Q_kW': np.float32(7.4634533), 'dP_bar': np.float32(0.007340195)}

import pandas as pd

# --- Table 6 (design variables) ---
design = pd.DataFrame({
    "Run": list(range(1, 16)),
    "SiO2_Al2O3": [50,80,80,80,50,80,23,50,23,50,23,23,50,50,50],
    "ZSM5_wt_pct": [22,30,22,13,30,22,22,22,13,30,30,22,13,13,22],
    "Temp_C": [500,500,450,500,450,550,550,500,500,550,500,450,550,450,500],
    "CTO": [5,6,5,4,6,5,5,5,4,6,6,5,4,4,5],
})

# --- Table 12 (measured responses) ---
resp = pd.DataFrame([
    [1,  73.73, 51.64, 4.14, 15.05, 23.28, 7.27, 14.82, 9.17],
    [2,  85.58, 64.78, 4.74, 19.21, 29.31, 8.34, 12.47, 11.52],
    [3,  71.53, 53.85, 2.42, 14.99, 25.30, 10.02, 7.66, 11.14],
    [4,  74.71, 56.30, 3.17, 15.20, 27.93, 10.63, 7.78, 10.00],
    [5,  59.08, 46.67, 2.93, 13.79, 21.73, 6.65, 5.77, 8.22],
    [6,  66.22, 53.87, 5.27, 18.00, 21.99, 6.33, 6.02, 8.61],
    [7,  70.84, 57.06, 6.57, 17.23, 23.70, 5.90, 7.87, 9.56],
    [8,  66.57, 40.77, 2.37, 12.05, 18.88, 7.63, 18.17, 7.47],
    [9,  63.59, 53.22, 5.67, 12.52, 23.47, 6.14, 4.23, 11.55],
    [10, 72.95, 52.09, 6.24, 12.48, 22.88, 6.66, 14.20, 10.50],
    [11, 72.02, 43.86, 5.47, 10.15, 19.04, 8.01, 20.14, 9.21],
    [12, 65.47, 43.52, 1.45, 11.33, 23.35, 10.27, 11.67, 7.39],
    [13, 73.08, 49.91, 5.36, 14.05, 21.83, 7.73, 15.44, 8.68],
    [14, 81.27, 50.21, 2.42, 12.84, 24.88, 12.54, 18.52, 10.08],
    [15, 70.50, 42.90, 2.77, 10.74, 21.80, 9.06, 18.54, 7.58],
], columns=[
    "Run","MassBalance_pct","Conversion_pct","DryGas_pct","LPG_pct",
    "GasolineNaphtha_pct","LCO_pct","BottomFrac_pct","Coke_pct"
])

df = design.merge(resp, on="Run")
df.to_csv("fcc_biooil_zeolite_dataset.csv", index=False)
print(df.head())
import pandas as pd

# --- Table 6 (design variables) ---
design = pd.DataFrame({
    "Run": list(range(1, 16)),
    "SiO2_Al2O3": [50,80,80,80,50,80,23,50,23,50,23,23,50,50,50],
    "ZSM5_wt_pct": [22,30,22,13,30,22,22,22,13,30,30,22,13,13,22],
    "Temp_C": [500,500,450,500,450,550,550,500,500,550,500,450,550,450,500],
    "CTO": [5,6,5,4,6,5,5,5,4,6,6,5,4,4,5],
})

# --- Table 12 (measured responses) ---
resp = pd.DataFrame([
    [1,  73.73, 51.64, 4.14, 15.05, 23.28, 7.27, 14.82, 9.17],
    [2,  85.58, 64.78, 4.74, 19.21, 29.31, 8.34, 12.47, 11.52],
    [3,  71.53, 53.85, 2.42, 14.99, 25.30, 10.02, 7.66, 11.14],
    [4,  74.71, 56.30, 3.17, 15.20, 27.93, 10.63, 7.78, 10.00],
    [5,  59.08, 46.67, 2.93, 13.79, 21.73, 6.65, 5.77, 8.22],
    [6,  66.22, 53.87, 5.27, 18.00, 21.99, 6.33, 6.02, 8.61],
    [7,  70.84, 57.06, 6.57, 17.23, 23.70, 5.90, 7.87, 9.56],
    [8,  66.57, 40.77, 2.37, 12.05, 18.88, 7.63, 18.17, 7.47],
    [9,  63.59, 53.22, 5.67, 12.52, 23.47, 6.14, 4.23, 11.55],
    [10, 72.95, 52.09, 6.24, 12.48, 22.88, 6.66, 14.20, 10.50],
    [11, 72.02, 43.86, 5.47, 10.15, 19.04, 8.01, 20.14, 9.21],
    [12, 65.47, 43.52, 1.45, 11.33, 23.35, 10.27, 11.67, 7.39],
    [13, 73.08, 49.91, 5.36, 14.05, 21.83, 7.73, 15.44, 8.68],
    [14, 81.27, 50.21, 2.42, 12.84, 24.88, 12.54, 18.52, 10.08],
    [15, 70.50, 42.90, 2.77, 10.74, 21.80, 9.06, 18.54, 7.58],
], columns=[
    "Run","MassBalance_pct","Conversion_pct","DryGas_pct","LPG_pct",
    "GasolineNaphtha_pct","LCO_pct","BottomFrac_pct","Coke_pct"
])

df = design.merge(resp, on="Run")
df.to_csv("fcc_biooil_zeolite_dataset.csv", index=False)
print(df.head())
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = df[["SiO2_Al2O3","ZSM5_wt_pct","Temp_C","CTO"]].values.astype(np.float32)
Y = df[["Conversion_pct","GasolineNaphtha_pct","Coke_pct","DryGas_pct","BottomFrac_pct"]].values.astype(np.float32)

Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.25, random_state=7)

sx = StandardScaler().fit(Xtr)
sy = StandardScaler().fit(Ytr)

Xtr_s = sx.transform(Xtr); Xte_s = sx.transform(Xte)
Ytr_s = sy.transform(Ytr); Yte_s = sy.transform(Yte)

Xtr_t = torch.tensor(Xtr_s); Ytr_t = torch.tensor(Ytr_s)
Xte_t = torch.tensor(Xte_s); Yte_t = torch.tensor(Yte_s)

model = nn.Sequential(
    nn.Linear(4, 64), nn.ReLU(),
    nn.Linear(64, 64), nn.ReLU(),
    nn.Linear(64, Y.shape[1])
)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

for epoch in range(1500):
    model.train()
    pred = model(Xtr_t)
    loss = loss_fn(pred, Ytr_t)
    opt.zero_grad(); loss.backward(); opt.step()

model.eval()
with torch.no_grad():
    te_pred = sy.inverse_transform(model(Xte_t).numpy())
print("Sample predictions (test):")
print(te_pred[:3])
pip -q install pymoo

import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize

def surrogate_predict(x_row):
    # x_row: [SiO2/Al2O3, ZSM5_wt%, Temp_C, CTO]
    x = np.array(x_row, dtype=np.float32).reshape(1, -1)
    x_s = sx.transform(x)
    with torch.no_grad():
        y_s = model(torch.tensor(x_s)).numpy()
    y = sy.inverse_transform(y_s)[0]
    # outputs: [Conversion, Gasoline, Coke, DryGas, BottomFrac]
    return y

class FCCOpt(Problem):
    def __init__(self):
        super().__init__(
            n_var=4, n_obj=4, n_constr=0,
            xl=np.array([23, 13, 450, 4]),
            xu=np.array([80, 30, 550, 6])
        )
    def _evaluate(self, X, out, *args, **kwargs):
        F = []
        for row in X:
            conv, gaso, coke, drygas, bottom = surrogate_predict(row)
            # minimize objectives (so negate gasoline for maximize)
            F.append([
                -gaso,
                coke,
                drygas,
                bottom
            ])
        out["F"] = np.array(F)

problem = FCCOpt()
algo = NSGA2(pop_size=80, sampling=FloatRandomSampling())

res = minimize(problem, algo, ('n_gen', 120), verbose=False)
pareto = res.X

print("Top 5 Pareto candidates:")
for i in range(5):
    print(pareto[i], "=>", surrogate_predict(pareto[i]))

