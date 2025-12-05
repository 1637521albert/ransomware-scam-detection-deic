import torch
import matplotlib.pyplot as plt
from torch_geometric.nn import HeteroConv, Linear, GATConv, SAGEConv, LayerNorm, HGTConv, RGCNConv, HANConv
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer
import numpy as np
import wandb
import random
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.utils import to_scipy_sparse_matrix
import scipy.sparse as sp
import pandas as pd
import json
from torch_geometric.data import HeteroData

## Read the dataset

prefix = '/usr/local/src/BlockSci/Notebooks/Ransomware - Mario i Albert/'
def get_parameters():
    print("Type of seed addresses:\n---------------------------------")
    print("1. Licit\n2. Illicit\n3. Licit and Illicit (50/50)")
    seed_options = {"1": "licit", "2": "illicit", "3": "licit and illicit"}
    seed = seed_options[input("Option: ")]

    print("\nDirection of the expansion:\n---------------------------------")
    print("1. Forward Backward\n2. All over")
    direction_options = {"1": "fw bw", "2": "all over"}
    direction = direction_options[input("Option: ")]

    print("\nApproach of the expansion:\n---------------------------------")
    print("1. Transaction-based\n2. Address-based")
    graph_options = {"1": " tx", "2": " addr"}
    graph = graph_options[input("Option: ")]

    if direction == "fw bw":
        print("\nTransaction addresses proportion of the expansion:\n---------------------------------")
        print("1. Whole\n2. Dedicated")
        address_options = {"1": " whole", "2": " dedicated"}
        address = address_options[input("Option: ")]

        if address == " whole" and graph == " addr":
            print("\nSide addresses direction of the expansion:\n---------------------------------")
            print("1. None\n2. Same\n3. Opposite")
            side_direction_options = {"1": " none", "2": " same", "3": " opposite"}
            side_direction = side_direction_options[input("Option: ")]
        else:
            side_direction = ""
    else:
        address = ""
        side_direction = ""

    exp_alg = f'{direction}{graph}{address}{side_direction}'

    
    print("\nLimit mode:\n---------------------------------")
    print("1. Random node\n2. Random hop\n3. None")
    limit_mode_options = {"1": "random node", "2": "random hop", "3": ""}
    limit_mode = limit_mode_options[input("Option: ")]

    if limit_mode != "":
        print("\nLimit value:\n---------------------------------")
        limit = int(input("Option: "))
        
    else:
        limit = "no"

    if exp_alg.startswith('fw bw'):
        print("\nNumber of forward hops:\n---------------------------------")
        for_hops = int(input("Option: "))
        
        print("\nNumber of backward hops:\n---------------------------------")
        back_hops = int(input("Option: "))
        hops = for_hops

    else:
        print("\nNumber of hops:\n---------------------------------")
        hops = int(input("Option: "))
        for_hops = hops
        back_hops = hops

    print("\nNumber of train samples:\n---------------------------------")
    train_samples = int(input("Option: "))

    print("\nNumber of validation samples:\n---------------------------------")
    val_samples = int(input("Option: "))

    print("\nNumber of test samples:\n---------------------------------")
    test_samples = int(input("Option: "))
    print()

    space = "" if limit_mode == "" else " "

    data_path = f'{prefix}Heterogeneous/{seed}/{train_samples}-{val_samples}-{test_samples} {exp_alg} {hops} hops {limit}{space}{limit_mode} limit/'

    return space, exp_alg, limit_mode, limit, for_hops, back_hops, hops, train_samples, val_samples, test_samples, prefix, data_path, seed

space, exp_alg, limit_mode, limit, for_hops, back_hops, hops, train_samples, val_samples, test_samples, prefix, data_path, seed = get_parameters()


## Data Loading

train_data = torch.load(data_path + 'train/graph.pth')
val_data = torch.load(data_path + 'val/graph.pth')
test_data = torch.load(data_path + 'test/graph.pth')

### 1.- Clip Outliers

def compute_clipping_bounds(x, z_threshold=3, idx=2):
    vals = x[:, idx].cpu().numpy()
    mu, sigma = vals.mean(), vals.std()
    lo = mu - z_threshold * sigma
    hi = mu + z_threshold * sigma
    return lo, hi

def apply_clipping(data, lo, hi, idx=2):
    x = data['tx'].x.clone()
    if x[:, idx].dtype.is_floating_point:
        vals = x[:, idx].cpu().numpy()
        clipped = np.clip(vals, lo, hi)
        with torch.no_grad():
            x[:, idx] = torch.from_numpy(clipped).to(x.device, dtype=x.dtype)
    data['tx'].x = x
    return data

lo, hi = compute_clipping_bounds(train_data['tx'].x, z_threshold=3, idx=2) #locktime

train_data = apply_clipping(train_data, lo, hi, idx=2)
val_data = apply_clipping(val_data, lo, hi, idx=2)
test_data = apply_clipping(test_data, lo, hi, idx=2)

### 2.- Data Normalization

def log_transform(data, element_type, element_name, attribute_indices):
    for idx in attribute_indices:
        if element_type == 'node':
            data[element_name].x[:, idx] = torch.log1p(data[element_name].x[:, idx])
        elif element_type == 'edge':
            data[element_name].edge_attr[:, idx] = torch.log1p(data[element_name].edge_attr[:, idx])

log_transform(train_data, element_type='node', element_name='tx', attribute_indices=[1, 3])  # fee, total_size
log_transform(val_data, element_type='node', element_name='tx', attribute_indices=[1, 3])  # fee, total_size
log_transform(test_data, element_type='node', element_name='tx', attribute_indices=[1, 3])  # fee, total_size

log_transform(train_data, element_type='node',element_name='addr', attribute_indices=[1, 2, 3, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52])
log_transform(val_data, element_type='node', element_name='addr', attribute_indices=[1, 2, 3, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52])
log_transform(test_data, element_type='node', element_name='addr', attribute_indices=[1, 2, 3, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52])

log_transform(train_data, element_type='edge', element_name=('addr', 'input', 'tx'), attribute_indices=[0, 2])  # age, value
log_transform(val_data, element_type='edge', element_name=('addr', 'input', 'tx'), attribute_indices=[0, 2])  # age, value
log_transform(test_data, element_type='edge', element_name=('addr', 'input', 'tx'), attribute_indices=[0, 2])  # age, value

log_transform(train_data, element_type='edge', element_name=('tx', 'output', 'addr'), attribute_indices=[0])  # value
log_transform(val_data, element_type='edge', element_name=('tx', 'output', 'addr'), attribute_indices=[0])  # value
log_transform(test_data, element_type='edge', element_name=('tx', 'output', 'addr'), attribute_indices=[0])  # value

def norm_attr(train_data, val_data, test_data, element_type, element_name, attribute_indices):
    for idx in attribute_indices:
        scaler = MinMaxScaler()

        if element_type == 'node':
            train_data[element_name].x[:, idx] = torch.tensor(
                scaler.fit_transform(train_data[element_name].x[:, idx].unsqueeze(1).numpy()).flatten(),
                dtype=torch.float
            )

            val_data[element_name].x[:, idx] = torch.tensor(
                scaler.transform(val_data[element_name].x[:, idx].unsqueeze(1).numpy()).flatten(),
                dtype=torch.float
            )

            test_data[element_name].x[:, idx] = torch.tensor(
                scaler.transform(test_data[element_name].x[:, idx].unsqueeze(1).numpy()).flatten(),
                dtype=torch.float
            )

        elif element_type == 'edge':
            train_data[element_name].edge_attr[:, idx] = torch.tensor(
                scaler.fit_transform(train_data[element_name].edge_attr[:, idx].unsqueeze(1).numpy()).flatten(),
                dtype=torch.float
            )

            val_data[element_name].edge_attr[:, idx] = torch.tensor(
                scaler.transform(val_data[element_name].edge_attr[:, idx].unsqueeze(1).numpy()).flatten(),
                dtype=torch.float
            )

            test_data[element_name].edge_attr[:, idx] = torch.tensor(
                scaler.transform(test_data[element_name].edge_attr[:, idx].unsqueeze(1).numpy()).flatten(),
                dtype=torch.float
            )
            

norm_attr(train_data, val_data, test_data, element_type='node', element_name='addr', attribute_indices=[1, 2, 3, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52])

norm_attr(train_data, val_data, test_data, element_type='node', element_name='tx', attribute_indices=[0, 1, 2, 3])  # block_height, fee, locktime, total_size

norm_attr(train_data, val_data, test_data, element_type='edge', element_name=('addr', 'input', 'tx'), attribute_indices=[0, 2])  # age, value

norm_attr(train_data, val_data, test_data, element_type='edge', element_name=('tx', 'output', 'addr'), attribute_indices=[0])  # value

def quant_norm(train_data, val_data, test_data, element_type, element_name, attribute_indices):
    for idx in attribute_indices:
        scaler = QuantileTransformer()

        if element_type == 'node':
            train_data[element_name].x[:, idx] = torch.tensor(
                scaler.fit_transform(train_data[element_name].x[:, idx].unsqueeze(1).numpy()).flatten(),
                dtype=torch.float
            )

            val_data[element_name].x[:, idx] = torch.tensor(
                scaler.transform(val_data[element_name].x[:, idx].unsqueeze(1).numpy()).flatten(),
                dtype=torch.float
            )

            test_data[element_name].x[:, idx] = torch.tensor(
                scaler.transform(test_data[element_name].x[:, idx].unsqueeze(1).numpy()).flatten(),
                dtype=torch.float
            )

        elif element_type == 'edge':
            train_data[element_name].edge_attr[:, idx] = torch.tensor(
                scaler.fit_transform(train_data[element_name].edge_attr[:, idx].unsqueeze(1).numpy()).flatten(),
                dtype=torch.float
            )

            val_data[element_name].edge_attr[:, idx] = torch.tensor(
                scaler.transform(val_data[element_name].edge_attr[:, idx].unsqueeze(1).numpy()).flatten(),
                dtype=torch.float
            )

            test_data[element_name].edge_attr[:, idx] = torch.tensor(
                scaler.transform(test_data[element_name].edge_attr[:, idx].unsqueeze(1).numpy()).flatten(),
                dtype=torch.float
            )
            
quant_norm(train_data, val_data, test_data, element_type='node', element_name='addr', attribute_indices=[4, 8, 9])


## Model Definition

edge_dims = {'input': 3, 'output': 2}
node_dims = {'addr': 53, 'tx': 6}
md = (['addr', 'tx'], [('addr', 'input', 'tx'), ('tx', 'output', 'addr'), ('tx', 'spent_output', 'tx')])

class HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, dropout_prob):
        super().__init__()
        
        self.node_enc = torch.nn.ModuleDict({
            'addr': torch.nn.Sequential(
                Linear(node_dims['addr'], hidden_channels),
                torch.nn.ReLU(),
                torch.nn.Dropout(p=0.2)
            ),
            'tx': torch.nn.Sequential(
                Linear(node_dims['tx'], hidden_channels),
                torch.nn.ReLU(),
                torch.nn.Dropout(p=0.2)
            )
        })
        
        self.norm_addr = LayerNorm(hidden_channels)
        self.norm_tx = LayerNorm(hidden_channels)

        self.conv1 = HeteroConv({
            ('addr', 'input', 'tx'): GATConv((-1, -1), hidden_channels, heads = 4, concat = False, edge_dim = edge_dims['input'], add_self_loops = False, dropout=dropout_prob),
            ('tx', 'output', 'addr'): GATConv((-1, -1), hidden_channels, heads = 4, concat = False, edge_dim = edge_dims['output'], add_self_loops = False, dropout=dropout_prob),
            ('tx', 'spent_output', 'tx'): GATConv(-1, hidden_channels, heads = 4, concat = False, add_self_loops = False, dropout=dropout_prob)
        }, aggr='sum')

        self.conv2 = HeteroConv({
            ('addr', 'input', 'tx'): GATConv((-1, -1), hidden_channels, heads = 4, concat = False, edge_dim = edge_dims['input'], add_self_loops = False, dropout=dropout_prob),
            ('tx', 'output', 'addr'): GATConv((-1, -1), hidden_channels, heads = 4, concat = False, edge_dim = edge_dims['output'], add_self_loops = False, dropout=dropout_prob)
        }, aggr='sum')
        
        self.conv3 = HeteroConv({
            ('addr', 'input', 'tx'): GATConv((-1, -1), hidden_channels, heads = 4, concat = False, edge_dim = edge_dims['input'], add_self_loops = False, dropout=dropout_prob),
            ('tx', 'output', 'addr'): GATConv((-1, -1), hidden_channels, heads = 4, concat = False, edge_dim = edge_dims['output'], add_self_loops = False, dropout=dropout_prob),
            ('tx', 'spent_output', 'tx'): GATConv(-1, hidden_channels, heads = 4, concat = False, add_self_loops = False, dropout=dropout_prob)
        }, aggr='sum')

        self.conv4 = HeteroConv({
            ('addr', 'input', 'tx'): GATConv((-1, -1), hidden_channels, heads = 4, concat = False, edge_dim = edge_dims['input'], add_self_loops = False, dropout=dropout_prob),
            ('tx', 'output', 'addr'): GATConv((-1, -1), hidden_channels, heads = 4, concat = False, edge_dim = edge_dims['output'], add_self_loops = False, dropout=dropout_prob)
        }, aggr='sum')

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, data):
        x_dict, edge_index_dict = data.x_dict, data.edge_index_dict
        edge_attr_dict = data.edge_attr_dict
        
        x_dict = {
            'addr': self.node_enc['addr'](x_dict['addr']).squeeze(1),
            'tx':   self.node_enc['tx'](x_dict['tx'])
        }
        
        x_prev = x_dict
        x_dict = self.conv1(x_dict, edge_index_dict, edge_attr_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        x_dict = { 'addr': self.norm_addr(x_dict['addr'] + x_prev['addr']),
           'tx'  : self.norm_tx  (x_dict['tx']   + x_prev['tx']) }
        
        x_prev = x_dict
        x_dict = self.conv2(x_dict, edge_index_dict, edge_attr_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        x_dict = { 'addr': self.norm_addr(x_dict['addr'] + x_prev['addr']),
           'tx'  : self.norm_tx  (x_dict['tx'] + x_prev['tx']) }
        
        x_prev = x_dict
        x_dict = self.conv3(x_dict, edge_index_dict, edge_attr_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        x_dict = { 'addr': self.norm_addr(x_dict['addr'] + x_prev['addr']),
           'tx'  : self.norm_tx  (x_dict['tx'] + x_prev['tx']) }
        
        x_prev = x_dict
        x_dict = self.conv4(x_dict, edge_index_dict, edge_attr_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        x_dict = { 'addr': self.norm_addr(x_dict['addr'] + x_prev['addr']),
           'tx'  : self.norm_tx  (x_dict['tx'] + x_prev['tx']) }

        out = self.lin(x_dict['addr'])
        return out
    
class HeteroGraphTransformer(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, dropout_prob):
        super().__init__()
        
        self.node_enc = torch.nn.ModuleDict({
            'addr': torch.nn.Sequential(
                Linear(node_dims['addr'], hidden_channels),
                torch.nn.ReLU(),
                torch.nn.Dropout(p=0.2)
            ),
            'tx': torch.nn.Sequential(
                Linear(node_dims['tx'], hidden_channels),
                torch.nn.ReLU(),
                torch.nn.Dropout(p=0.2)
            )
        })
        
        self.norm_addr = LayerNorm(hidden_channels)
        self.norm_tx = LayerNorm(hidden_channels)

        self.conv1 = HGTConv(hidden_channels, hidden_channels, metadata=md, heads=4)

        self.conv2 = HGTConv(hidden_channels, hidden_channels, metadata=md, heads=4)
        
        self.conv3 = HGTConv(hidden_channels, hidden_channels, metadata=md, heads=4)
        
        self.conv4 = HGTConv(hidden_channels, hidden_channels, metadata=md, heads=4)

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, data):
        x_dict, edge_index_dict = data.x_dict, data.edge_index_dict
        
        x_dict = {
            'addr': self.node_enc['addr'](x_dict['addr']).squeeze(1),
            'tx':   self.node_enc['tx'](x_dict['tx'])
        }
        
        x_prev = x_dict
        x_norm = {k: self.norm_addr(v) if k == 'addr' else self.norm_tx(v) for k, v in x_dict.items()}
        x_new = self.conv1(x_norm, edge_index_dict)
        x_new = {key: F.relu(x) for key, x in x_new.items()}
        x_dict = {'addr': x_prev['addr'] + x_new['addr'], 'tx':   x_prev['tx'] + x_new['tx']}
        
        x_prev = x_dict
        x_norm = {k: self.norm_addr(v) if k == 'addr' else self.norm_tx(v) for k, v in x_dict.items()}
        x_new = self.conv2(x_norm, edge_index_dict)
        x_new = {key: F.relu(x) for key, x in x_new.items()}
        x_dict = {'addr': x_prev['addr'] + x_new['addr'], 'tx':   x_prev['tx'] + x_new['tx']}
        
        x_prev = x_dict
        x_norm = {k: self.norm_addr(v) if k == 'addr' else self.norm_tx(v) for k, v in x_dict.items()}
        x_new = self.conv3(x_norm, edge_index_dict)
        x_new = {key: F.relu(x) for key, x in x_new.items()}
        x_dict = {'addr': x_prev['addr'] + x_new['addr'], 'tx':   x_prev['tx'] + x_new['tx']}
        
        x_prev = x_dict
        x_norm = {k: self.norm_addr(v) if k == 'addr' else self.norm_tx(v) for k, v in x_dict.items()}
        x_new = self.conv4(x_norm, edge_index_dict)
        x_new = {key: F.relu(x) for key, x in x_new.items()}
        x_dict = {'addr': x_prev['addr'] + x_new['addr'], 'tx':   x_prev['tx'] + x_new['tx']}

        out = self.lin(x_dict['addr'])
        return out
    
class HeteroAttentionNet(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, dropout_prob):
        super().__init__()
        
        self.node_enc = torch.nn.ModuleDict({
            'addr': torch.nn.Sequential(
                Linear(node_dims['addr'], hidden_channels),
                torch.nn.ReLU(),
                torch.nn.Dropout(p=0.2)
            ),
            'tx': torch.nn.Sequential(
                Linear(node_dims['tx'], hidden_channels),
                torch.nn.ReLU(),
                torch.nn.Dropout(p=0.2)
            )
        })
        
        self.norm_addr = LayerNorm(hidden_channels)
        self.norm_tx = LayerNorm(hidden_channels)

        self.conv1 = HANConv(hidden_channels, hidden_channels, heads=4, dropout=dropout_prob, metadata= md)

        self.conv2 = HANConv(hidden_channels, hidden_channels, heads=4, dropout=dropout_prob, metadata=md)
        
        self.conv3 = HANConv(hidden_channels, hidden_channels, heads=4, dropout=dropout_prob, metadata=md)
        
        self.conv4 = HANConv(hidden_channels, hidden_channels, heads=4, dropout=dropout_prob, metadata=md)
        
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, data):
        x_dict, edge_index_dict = data.x_dict, data.edge_index_dict
        
        x_dict = {
            'addr': self.node_enc['addr'](x_dict['addr']).squeeze(1),
            'tx':   self.node_enc['tx'](x_dict['tx'])
        }
        
        x_prev = x_dict
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        x_dict = { 'addr': self.norm_addr(x_dict['addr'] + x_prev['addr']),
           'tx'  : self.norm_tx  (x_dict['tx']   + x_prev['tx']) }
        
        x_prev = x_dict
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        x_dict = { 'addr': self.norm_addr(x_dict['addr'] + x_prev['addr']),
           'tx'  : self.norm_tx  (x_dict['tx']   + x_prev['tx']) }
        
        x_prev = x_dict
        x_dict = self.conv3(x_dict, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        x_dict = { 'addr': self.norm_addr(x_dict['addr'] + x_prev['addr']),
           'tx'  : self.norm_tx  (x_dict['tx']   + x_prev['tx']) }
        
        x_prev = x_dict
        x_dict = self.conv4(x_dict, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        x_dict = { 'addr': self.norm_addr(x_dict['addr'] + x_prev['addr']),
           'tx'  : self.norm_tx  (x_dict['tx']   + x_prev['tx']) }
        

        out = self.lin(x_dict['addr'])
        return out

## Training

hidden_channels = 32
out_channels = 2
lr = 0.001
dropout_prob = 0.2
criterion = torch.nn.CrossEntropyLoss()
epochs = 1000

train_labeled_mask = (train_data['addr'].y != -1)
test_labeled_mask = (test_data['addr'].y != -1)
val_labeled_mask = (val_data['addr'].y != -1)

use_wandb = input("\nUse wandb? (yes/no): ").strip().lower() == "yes"
debug_mode = input("Debug mode? (yes/no): ").strip().lower() == "yes"


#model = HeteroGNN(hidden_channels, out_channels, dropout_prob)
#model = HeteroGraphTransformer(hidden_channels, out_channels, dropout_prob)
model = HeteroAttentionNet(hidden_channels, out_channels, dropout_prob)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


if use_wandb:
    run = wandb.init(
        name=f"{seed} {train_samples}-{test_samples} {exp_alg} {hops} hops {limit} {limit_mode}", 
        entity="bitcoin-ransomware-addresses-detection",
        project="illicit",
        config={
            "learning_rate": lr,
            "epochs": epochs,
            "architecture": "2layers-HGT",
            "hidden_channels": hidden_channels,
            "out_channels": out_channels,
            "optimizer": "Adam",
            "loss_function": "Cross Entropy Loss",
            "dropout_prob": dropout_prob,
            "dataset": {
                "seed": seed,
                "train_samples": train_samples,
                "test_samples": test_samples,
                "expansion_algorithm": exp_alg,
                "hops": hops,
                "limit_mode": limit_mode,
                "limit_value": limit,
                "forward_hops": for_hops if exp_alg.startswith("fw bw") else None,
                "backward_hops": back_hops if exp_alg.startswith("fw bw") else None
            }
        }
    )


best_accuracy = 0.0
best_epoch = 0
patience = 10
epochs_no_improve = 0
train_losses = []
val_losses = []

def train():
    model.train()
    optimizer.zero_grad()
    out = model(train_data) 
    loss = criterion(out[train_labeled_mask], train_data['addr'].y[train_labeled_mask])
    loss.backward()
    optimizer.step()
    return loss.item(), out[train_labeled_mask]

def test():
    model.eval()
    with torch.no_grad():
        out = model(val_data)
        loss = criterion(out[val_labeled_mask], val_data['addr'].y[val_labeled_mask])
    return loss.item(), out[val_labeled_mask]

model.train()
with torch.no_grad():
    _ = model(train_data)

if use_wandb:
    wandb.watch(model, criterion, log="all", log_freq=10)
    
for epoch in range(1, epochs + 1):
    train_loss, train_logits = train()
    val_loss, val_logits = test()
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    with torch.no_grad():
        # Train metrics
        y_true_train = train_data['addr'].y[train_labeled_mask]
        y_pred_train = train_logits.softmax(dim=1).argmax(dim=1)
        accuracy_train = accuracy_score(y_true_train.cpu(), y_pred_train.cpu())
        precision_train = precision_score(y_true_train.cpu(), y_pred_train.cpu(), pos_label=1)
        recall_train = recall_score(y_true_train.cpu(), y_pred_train.cpu(), pos_label=1)
        f1_train = f1_score(y_true_train.cpu(), y_pred_train.cpu(), pos_label=1)
        cm_train = confusion_matrix(y_true_train.cpu(), y_pred_train.cpu())

        # Validation metrics
        y_true_val = val_data['addr'].y[val_labeled_mask]
        y_pred_val = val_logits.softmax(dim=1).argmax(dim=1)
        accuracy_val = accuracy_score(y_true_val.cpu(), y_pred_val.cpu())
        precision_val = precision_score(y_true_val.cpu(), y_pred_val.cpu(), pos_label=1)
        recall_val = recall_score(y_true_val.cpu(), y_pred_val.cpu(), pos_label=1)
        f1_val = f1_score(y_true_val.cpu(), y_pred_val.cpu(), pos_label=1)
        cm_val = confusion_matrix(y_true_val.cpu(), y_pred_val.cpu())

    # Save checkpoint if validation F1 improves
    if accuracy_val > best_accuracy:
        epochs_no_improve = 0
        best_accuracy = accuracy_val
        best_epoch = epoch
        torch.save(model.state_dict(), data_path + "best_model.pth")
        if use_wandb:
            wandb.run.summary["best_accuracy"] = best_accuracy
            wandb.run.summary["best_epoch"] = best_epoch
            
    else:
        epochs_no_improve += 1

    if debug_mode:
        print(f"\nEpoch {epoch} | Train Loss: {train_loss} | Validation Loss: {val_loss}")
        print(f"(Train) Accuracy: {accuracy_train} | Precision: {precision_train} | Recall: {recall_train} | F1: {f1_train}")
        print(f"Confusion Matrix (Train):\n{cm_train}")
        print(f"(Validation) Accuracy: {accuracy_val} | Precision: {precision_val} | Recall: {recall_val} | F1: {f1_val}")
        print(f"Confusion Matrix (Validation):\n{cm_val}")

    if use_wandb:
        run.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_accuracy":accuracy_train,
            "train_precision": precision_train,
            "train_recall": recall_train,
            "train_f1": f1_train,
            "val_accuracy": accuracy_val,
            "val_precision": precision_val,
            "val_recall": recall_val,
            "val_f1": f1_val
        })
        
    if epochs_no_improve >= patience:
        print(f"Early stopping at epoch {epoch} (no improvement in {patience} epochs).")
        break

if use_wandb:
    artifact = wandb.Artifact(
        name="hetero-gnn-model",
        type="model",
        metadata={
            "epochs": epochs,
            "learning_rate": lr,
            "hidden_channels": hidden_channels,
        }
    )
    artifact.add_file(data_path + "best_model.pth")
    run.log_artifact(artifact)
    
print(f"Best val accuracy: {best_accuracy:.4f} at epoch {best_epoch}")

## Model Evaluation

#model = HeteroGNN(hidden_channels, out_channels, dropout_prob)
#model = HeteroGraphTransformer(hidden_channels, out_channels, dropout_prob)
model = HeteroAttentionNet(hidden_channels, out_channels, dropout_prob)
state_dict = torch.load(data_path + "best_model.pth", map_location='cpu')
model.load_state_dict(state_dict)
model.eval()
tag = "test"

if tag == "train":
    tag_data = train_data
    labeled_mask = train_labeled_mask
elif tag == "test":
    tag_data = test_data
    labeled_mask = test_labeled_mask
elif tag == "val":
    tag_data = val_data
    labeled_mask = val_labeled_mask

@torch.no_grad()
def predict(model, data, labeled_mask):
    logits = model(data)
    logits = logits[labeled_mask]
    y_pred  = logits.softmax(1).argmax(dim=1) 
    return y_pred

def evaluate_model(model, data, labeled_mask, tag):
    model.eval()
    with torch.no_grad():
        y_pred = predict(model, data, labeled_mask)

    y_true = data['addr'].y[labeled_mask].cpu().numpy()
    y_pred = y_pred.cpu().numpy()

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label=1)
    recall = recall_score(y_true, y_pred, pos_label=1)
    f1 = f1_score(y_true, y_pred, pos_label=1)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    auc_roc = roc_auc_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print(f"Accuracy        : {accuracy:.4f}")
    print(f"Precision       : {precision:.4f}")
    print(f"Recall          : {recall:.4f}")
    print(f"F1-score        : {f1:.4f}")
    print(f"ROC-AUC         : {auc_roc:.4f}")
    print(f"Confusion Matrix:\n{cm}")

    if use_wandb:
        wandb.log({f"{tag}_{m}": v for m, v in
                  zip(["accuracy","precision","recall","f1","auc_roc", "confusion_matrix"],
                      [accuracy, precision, recall, f1, auc_roc, cm])})

print(f"\nEvaluating model on {tag} data...")
evaluate_model(model, tag_data, labeled_mask, tag=tag)

if use_wandb:
    run.finish()
