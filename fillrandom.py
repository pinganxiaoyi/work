import matplotlib
matplotlib.use('Agg')
from make_graph2 import Make_Graphs
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
from sklearn.metrics import f1_score, precision_recall_curve, auc, f1_score, accuracy_score
import CERN_EdgeNetworkModel_DGL as cerndgl
from CERN_EdgeNetworkModel_DGL import EdgeClassifier
from ROOT import TCanvas, TGraph2D 

#//import matplotlib.pyplot as plt
import dgl.function as fn

from ROOT import TCanvas, TGraph2D
from ROOT import TCanvas, TGraph2D

dataset = Make_Graphs()
print("数据集中的图数量:", len(dataset))

# 处理 dataset 中的前 10 个图
for index, (graph, label) in enumerate(dataset):
    if index >= 10:  # 如果已处理 10 个图，则停止
        break



    # 获取所有节点的位置信息
    positions = graph.ndata['feat'][:, :3].numpy()
    print(f"图 {index} 的节点数:", len(positions))
    # 为每个图绘制所有节点
    c = TCanvas(f"c{index}", f"Graph {index}", 800, 600)
    g = TGraph2D()

    for i, pos in enumerate(positions):
        print(f"图 {index}，节点 {i} 的坐标: {pos}")
        g.SetPoint(i, pos[0], pos[1], pos[2])

    g.Draw('pcol')
    c.SaveAs(f'graph_{index}.png')