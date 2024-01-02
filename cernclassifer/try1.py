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
from sklearn.metrics import f1_score, precision_recall_curve, auc, f1_score, accuracy_score
import CERN_EdgeNetworkModel_DGL as cerndgl

from CERN_EdgeNetworkModel_DGL import EdgeClassifier
from fit_angle import selected_angle,fit_angle
#from try0 import ParticleGraphNetwork
in_feats = 4  # 输入特征的维度
hidden_feats = 10  # 隐藏层的维度
num_classes = 2  # 输出类别的数量
num_layers = 8 # 网络层数
model = EdgeClassifier()
model.load_state_dict(torch.load("F:\\work\\model_parameter0.pth"))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()
print("model loaded")
dataset = Make_Graphs("F:\\work\\muon_10GeV_noCalo_comp_vertical-center-nx.g4mac.root",data_part="val")
#print ( dataset )

#设置数据集验证集测试集
trainset, valset, testset = split_dataset(dataset, frac_list=[0.7, 0.2,0.1])
#train_loader, val_loader, test_loader = cerndgl.get_data_loaders(trainset, valset, testset, batch_size=32)
train_loader = GraphDataLoader(trainset, batch_size=32,shuffle=True)
test_loader = GraphDataLoader(testset, batch_size=1,shuffle=True)
val_loader = GraphDataLoader(dataset,batch_size = 1,shuffle = False)

total_nodes = 0
total_edges = 0
none_count = 0
# Calculate angle resolution
accuracy_list = []
xy_pred_list = []
xz_pred_list = []
yz_pred_list = []
tot_edep_pred_list = []
def resolution(model2, graph,labels, features, efeatures,label):
    #logits = model2(graph, features, efeature)
    model2.to(device)
    graph = graph.to(device)
    labels = labels.to(device) #graph's Number 
    features = features.to(device)
    efeatures = efeatures.to(device)
    label = label.to(device)
    logits = model2(graph, features)
    #logits = logits
    #labels = labels
    logit_classes = torch.sigmoid(logits) > 0.5
    print("logit_classes",logit_classes)
    edges = graph.edges()
    correct = torch.sum(logit_classes == label)
    correct = correct.item() *1.0 / len(label) 
    src_nodes,dst_nodes = edges[0].cpu().numpy(),edges[1].cpu().numpy()
    selected_edges = []
    for i in range(len(logit_classes)):
        if logit_classes[i].item() ==0:
            selected_edges.append(( src_nodes[i], dst_nodes[i]))
    xy_pred, xz_pred, yz_pred, tot_edep_pred  = selected_angle(graph,selected_edges)
    xy_truth, xz_truth, yz_truth, tot_edep_truth, err_truth = fit_angle(graph,labels)
    #角分辨# 检查预测值和真实值是否有效
    global none_count  
    if xy_pred is None  or xz_pred is None or yz_pred is None  or tot_edep_pred is None :
        none_count += 1 
        return correct, None, None, None, None  # 返回 None 值
    
    # 计算 sigma 值
    
    global total_nodes , total_edges
    total_nodes += graph.number_of_nodes()
    total_edges += graph.number_of_edges()
    print(f"Graph label is {labels},Number of nodes: {graph.number_of_nodes()}, Number of edges: {graph.number_of_edges()},xy_pred is {xy_pred}")
    return correct,xy_pred,xz_pred,yz_pred,tot_edep_pred


# 在 evaluate 
count = 0

for graph, labels in val_loader:
    
    correct, xy_pred,xz_pred,yz_pred,tot_edep_pred = resolution(model, graph, labels, graph.ndata['feat'], graph.edata['weight'], graph.edata['label'])
    #计算角分辨
    accuracy_list.append(correct)
    if xy_pred is None  or xz_pred is None or yz_pred is None  or tot_edep_pred is None :
        count += 1
        continue
    
    xy_pred_list.append(xy_pred)
    xz_pred_list.append(xz_pred)
    yz_pred_list.append(yz_pred)
    
print("count is ",count)
'''
xz_array = np.array(xz_pred_list)
hist = ROOT.TH1F("hist", "Data Histogram", 500, min(xz_pred_list), max(xz_pred_list))
for val in xz_pred_list:
    hist.Fill(val)

# 绘制直方图
c1 = ROOT.TCanvas("c1", "Canvas", 800, 600)
hist.Draw()

# 计算累积分布和阈值
cumulative = 0
total = hist.Integral()
threshold_value = None
for bin in range(1, hist.GetNbinsX() + 1):
    cumulative += hist.GetBinContent(bin)
    if cumulative / total >= 0.682:
        threshold_value = hist.GetBinLowEdge(bin)
        break

# 在阈值处画线
if threshold_value is not None:
    line = ROOT.TLine(threshold_value, 0, threshold_value, hist.GetMaximum())
    line.SetLineColor(ROOT.kRed)
    line.SetLineStyle(2)
    line.Draw("same")

    # 添加文字说明
    pt = ROOT.TPaveText(0.2, 0.6, 0.4, 0.8, "blNDC")
    pt.AddText(f"Threshold: {threshold_value:.2f}")
    pt.SetFillColor(0)
    pt.Draw()

# 更新画布并保存
c1.Update()
c1.Draw()
c1.WaitPrimitive()
c1.SaveAs("newhistogram_with_threshold2.png")
'''
c1 = ROOT.TCanvas("c1","Accurary",1200,1800)
x_values = np.arange(len(accuracy_list), dtype=np.float64)  # x values as float64 array
y_values = np.array(accuracy_list, dtype=np.float64)        # y values as float64 array
accuracy_graph = ROOT.TGraph(len(x_values), x_values, y_values)
accuracy_graph.SetTitle("Accuracy with DIY;Test Sample;Accuracy")
accuracy_graph.SetMarkerStyle(20)
accuracy_graph.SetDrawOption("AP")
accuracy_graph.Draw("AP")      
c1.Update()
print("Entries in accuracy_list:", len(accuracy_list))
print("Total graphs resolution calculated:", count)
print(f"Total graphs discarded : {none_count}")
c1.SaveAs("Accnew10Gevmuon.png")   