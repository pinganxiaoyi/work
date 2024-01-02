import matplotlib
matplotlib.use('Agg')
from make_graph import Make_Graphs
import dgl
import torch
import torch.nn as nn
import dgl.nn as dglnn
import ROOT
#//from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from dgl.nn import GraphConv
#from real_final_class import Making_Graphs
from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from dgl.data.utils import split_dataset
#from sklearn.metrics import f1_score, precision_recall_curve, auc, f1_score, accuracy_score
#import CERN_EdgeNetworkModel_DGL as cerndgl
#from CERN_EdgeNetworkModel_DGL import EdgeClassifier

#将模型参数导入
dataset = Make_Graphs("F:\\work\\muon_10GeV_noCalo_comp_vertical-center-nY.g4mac.root",data_part="val")
loader = GraphDataLoader(dataset, batch_size=1, shuffle=False)
for num,(graphs, labels) in enumerate(loader):
    if num>3:
        break 
    num+=1
    for i in range(5):
        edge = graphs.edges 
        print()
    
    break