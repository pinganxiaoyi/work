import torch
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, auc, f1_score, accuracy_score
from tqdm import tqdm
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

import dgl
import dgl.nn as dglnn
import dgl.function as fn
from dgl.nn.pytorch import GraphConv
from dgl.dataloading import GraphDataLoader

######### Custom Dataset Class #########

#It recursively reads all the files ending with .pt extentsion in the home directory and the sub-directories. 
#The data is then split into 80/10/10 train-val-test split. Can be changed.
#Not all data is loaded into memory.

class MyDataset(Dataset):

    """
     It recursively reads all the files ending with .pt extentsion in the directory and the sub-directories. 
     The data is then split into 80/10/10 train-val-test split. Split ratio can be specified. Default value is set to 0.8 train size.
     Not all data is loaded into memory.
    """

    def __init__(self, path: Path, split_ratio = 0.2, split = 'train', random_seed = 42):
        super().__init__()

        # Recursively reading all .pth files from the directory and sub-directories
        self.data_list = list(path.glob("*/*[0-9].pt"))

        self.split_ratio = split_ratio
        self.split = split
        self.random_seed = random_seed
        
        # Split the data into training and test sets

        train_data_list, rem_data_list = train_test_split(self.data_list, test_size=split_ratio, random_state=random_seed)
        val_data_list, test_data_list = train_test_split(rem_data_list, test_size=0.5, random_state=random_seed)
        
        # Set the data list to the appropriate split
        if split == 'train':
            self.graphs = train_data_list
        elif split == 'test':
            self.graphs = test_data_list
        elif split == 'val':
            self.graphs = val_data_list
        else:
            raise ValueError(f"Invalid split: {split}")
      
    def __getitem__(self, idx):
        return torch.load(self.graphs[idx])
    
    def __len__(self) -> int:
        return len(self.graphs)



######### Model Architecture Class #########

class EdgeClassifier(nn.Module):

    """
    GNN network for edge classifcation.
    Default parameters have node_features = 6, edge_features = 4, hidden_channels = 64, classes = 2

    2-hop GCN for node features aggregation with activation as ReLU.
    Followed by concatenation of the neighboring nodes of an edge with it's own features
    to get a feature vector for each.
    Finally followed by linear layers and dropout.
    """

    #def __init__(self, num_node_features=4, num_edge_features=1, hidden_channels=64, num_classes=2):
    def __init__(self, num_node_features=4, num_edge_features=5, hidden_channels=64, num_classes=1):

        super(EdgeClassifier, self).__init__()
        """
        #PyG
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        """
        #DGL
        #self.conv1 = GraphConv(num_node_features, hidden_channels)
        #self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.conv1 = dglnn.SAGEConv(num_node_features, hidden_channels, aggregator_type = 'mean')
        self.conv2 = dglnn.SAGEConv(hidden_channels, hidden_channels, aggregator_type = 'mean')

        #self.conv1 = dglnn.GATConv(in_feats=num_edge_features,out_feats=hidden_channels, num_heads=1)
        #self.conv2 = dglnn.GATConv(in_feats = hidden_channels,out_feats = hidden_channels, num_heads = 1)
        self.lin1 = nn.Linear(num_edge_features + 2 * hidden_channels, hidden_channels)
        #self.lin1 = nn.Linear(2*hidden_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, num_classes)

#处理图中点和边的函数
        self.nodeprocess = nn.Linear(num_node_features, hidden_channels)
        self.edgeprocess = nn.Sequential(
            nn.Linear(2 * hidden_channels + num_edge_features, hidden_channels),
            nn.Tanh(),
            nn.Linear(hidden_channels, 1)
        )
    #def forward(self, data):
    def forward(self, g):

        """
        #PyG
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # Aggreagation of Node features
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))

        # Concatenation of features of nodes and the edge between them
        x = torch.cat([x[edge_index[0]], x[edge_index[1]], edge_attr], dim=1)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)

        # No softmax since Cross-Entropy Loss also applies softmax
        return x
        """
        #DGL
        # g is a DGLGraph object with edge features and node features
        # Aggreagation of Node features
        #g = data
        node_features = g.ndata['feat']
        edge_features = torch.cat([g.edata[f] for f in ['weight','dx','dy','dz','dE']],dim = 1)
        
        
        # Concatenation of features of nodes and the edge between them
        #g = dgl.remove_self_loop(g)
        #
        #e0 = g.edata['weight']#?这里可以跟据距离做出限制 ——————但是真的需要吗？（MakeGraph类的定义中已经进行了限制（层数的限制））
        #e1 = g.edata['dx']
        #e2 = g.edata['dy']
        #e3 = g.edata['dz']
        #e4 = g.edata['dE']
        #feat = g.ndata['feat']

        src, dst = g.edges()
        h = F.relu(self.conv1(g,node_features))
        h = F.relu(self.conv2(g,h))
        h_src = h[src]
        h_dst = h[dst]
        h_src_processed = F.relu(self.nodeprocess(h_src))
        h_dst_processed = F.relu(self.nodeprocess(h_dst))
        edge_feat = torch.cat([h_src_processed,h_dst_processed,edge_features],dim=1)
        w = torch.sigmoid(self.edgeprocess(edge_feat))
        edge_feat_weighted = edge_feat * w
        edge_feat_transformed = F.relu(self.lin1(edge_feat_weighted))
        edge_feat_transformed = F.dropout(edge_feat_transformed, p=0.5, training=self.training)
        
        return self.lin2(edge_feat_transformed)
        #print(h_src)
        #print(g.srcdata['h'])
        #print(h_dst)
        #print(g.dstdata['h'])
        #print(e)
        #print("hedge", h_edge)

        #src, dst = g.srcdata['h'], g.dstdata['h']
        #print(src)
        #print(dst)
        #print(e)
        #feat = torch.cat([src, e, dst], dim=1)
        #feat = torch.cat([src,  dst], dim=1)
        #feat = F.relu(self.lin1(feat))
        #feat = F.dropout(feat, p=0.5, training=self.training)
        #feat = self.lin2(feat)
        #print("graph feature", g.ndata['feat'])
        #print("conv graph feature", g.ndata['h'])
        #print("concated feature", h_edge)
        #print("graph edge", g.edata['weight'])

        #print(h_edge.dtype)
        #h_edge=h_edge.float()
        #print(h_edge.dtype)
        #h_edge = F.relu(self.lin1(h_edge))
        #h_edge = F.dropout(h_edge, p=0.5, training=self.training)
        #h_edge = self.lin2(h_edge)
        """
        print("h edge", h_edge)
        print("srcs", src)
        print("dsts", dst)
        print(g.edata['train_mask'])
        print(g.edata['val_mask'])
        print(g.edata['test_mask'])
        print("===")

        print(g.ndata['train_mask'])
        print(g.ndata['val_mask'])
        print(g.ndata['test_mask'])
        """

        '''
        e = torch.cat([g.edata[f] for f in ['weight','dx','dy','dz','dE']],dim = 1)
        h = F.relu(self.conv1(g,h).flatten(1))
        h = F.relu(self.conv2(g,h).flatten(1))
        src,dst = g.edges()
        h_edge = 
        '''
        # No softmax since Cross-Entropy Loss also applies softmax

        #return h_edge
        #return feat




######### DataLoader Setup #########

def get_data_loaders(train_dataset, val_dataset, test_dataset, batch_size=32):

    """
    Function to create the DataLoaders for train-val-test data. 
    Can specify batch size. Default value is set to 32.
    """

    # Shuffle=True for training data to get diversity in batches at each training epoch
    train_loader = GraphDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = GraphDataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = GraphDataLoader(test_dataset, batch_size=1, shuffle=False)

    return train_loader, val_loader, test_loader




######### Set up device #########

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')




######### Handling Data-Imbalance #########

def calculate_class_weights(train_loader):

    """
    Calculates weightage given to each sample in calculating loss by taking inverse of their frequency.
    Currently functional for binary labelled dataset.
    """

    sum = 0
    total = 0
    for data in tqdm(train_loader):
        sum += data.y.sum().item()
        total += len(data.y)

    freq1 = sum/total
    freq0 = 1-freq1
    weight1 = 1/(total*freq1)
    weight0 = 1/(total*freq0)

    print(f'Frequency of class 0 : {freq0}, Assigned weight : {weight0}')
    print(f'Frequency of class 1 : {freq1}, Assigned weight : {weight1}')

    return torch.tensor([weight0, weight1]).to(get_device())




######### Defining Training-Validation-Testing Methods #########


def train(model, device, train_loader, optimizer, criterion):

    model.train()
    correct = 0
    total = 0
    total_loss = 0

    for data in tqdm(train_loader):

        data = data.to(device)
        optimizer.zero_grad()

        # Forward and Backward Propagation
        out = model(data)
        loss = criterion(out, data.y.long())
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Calculation of correctly classified edges
        pred = out.argmax(dim=1)
        correct += (pred == data.y).sum().item()
        total += len(pred)

    train_loss = total_loss / len(train_loader)
    train_acc = correct / total

    print(f'Train Loss : {train_loss:.4f}')
    print(f'Train Accuracy: {train_acc:.3f}')

    return model, train_loss

def evaluate(model, device, loader):

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data in tqdm(loader):

            data = data.to(device)
            out = model(data)

            # Calculation of correctly classified edges
            pred = out.argmax(dim=1)
            correct += (pred == data.y).sum().item()
            total += len(pred)

    acc = correct / total
    print(f'Validation Accuracy {acc:.3f}')

def test(model, device, loader):

    print('Testing The Model')

    model.eval()
    y_true = []
    y_probas = []
    y_pred = []

    with torch.no_grad():
        for data in tqdm(loader):

            data = data.to(device)
            out = model(data)

            y_true += data.y.cpu().numpy().tolist()
            y_pred += out.argmax(dim=1).cpu().numpy().tolist()  # absoulte predictions
            y_probas += out[:, 1].cpu().numpy().tolist()  # probability of class 1

    
    # Calculating few metrics

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    precision, recall, thresholds = precision_recall_curve(y_true, y_probas)
    area = auc(recall, precision)

    print('\nResults\n')
    print(f'Testing Accuracy {acc:.3f}')
    print(f'F1 score: {f1:.3f}')
    print(f'AUCPR: {area:.3f}\n')

    return acc, f1, precision, recall, area


def plot_pr_curve(precision, recall, area, name='pr-curve'):

    """
    Function for plotting and saving the AUCPR curve
    AUCPR is used as metric and not ROC. 
    Since, if the dataset is highly imbalanced then FPR would be close to 0 since TN  would be very high.
    """
	
    plt.plot(recall, precision)

    plt.plot([1, 0], [0, 1], 'k--')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall curve, AUC:{area:.3f}')
    plt.savefig(f'{name}.png')


############################################################################################
