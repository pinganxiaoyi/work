'''
import ROOT
from pathlib import Path

file_path = Path("F:\\work\\test-proton-vertical-full-100GeV_n-simdigi.root")
f = ROOT.TFile(str(file_path), "READ")
tree = f.Get("events")

n_fithits = ROOT.std.vector("float")()
tree.SetBranchAddress("fithits.pos.x", n_fithits)

total = 0
for i in range(tree.GetEntries()):
    tree.GetEntry(i)
    total += len(n_fithits)
print(total)
'''
'''
import ROOT
from ROOT import TMath, TF1

# 定义朗道和高斯函数
def landau(x, par):
    return TMath.Landau(x[0], par[0], par[1])

def gaussian(x, par):
    return TMath.Gaus(x[0], par[0], par[1])

# 定义卷积函数
def landau_gauss_convolution(x, par):
    # 定义积分范围和精度
    x_min = -10
    x_max = 10
    precision = 0.01

    result = 0.0
    t = x_min
    while t <= x_max:
        result += landau([t], [par[2], par[3]]) * gaussian([x[0] - t], [par[0], par[1]]) * precision
        t += precision
    
    return result

# 创建 TF1 对象
conv_func = TF1("conv_func", landau_gauss_convolution, -10, 10, 4)
conv_func.SetParameters(0, 1, 0, 1) # 设置初始参数

# 绘制卷积函数
conv_func.Draw()
'''
'''
from os import path
from ROOT import TCanvas, TFile, TPaveText
from ROOT import gROOT, gBenchmark

c1 = TCanvas( 'c1', 'The Fit Canvas', 200, 10, 700, 500 )
c1.SetGridx()
c1.SetGridy()
c1.GetFrame().SetFillColor( 21 )
c1.GetFrame().SetBorderMode(-1 )
c1.GetFrame().SetBorderSize( 5 )

gBenchmark.Start( 'fit1' )
#
# We connect the ROOT file generated in a previous tutorial
#
fill = TFile( 'py-fillrandom.root' )

#
# The function "ls()" lists the directory contents of this file
#
fill.ls()

#
# Get object "sqroot" from the file.
#

sqroot = gROOT.FindObject( 'sqroot' )
sqroot.Print()

#
# Now fit histogram h1f with the function sqroot
#
h1f = gROOT.FindObject( 'h1f' )
h1f.SetFillColor( 45 )
h1f.Fit( 'sqroot' )

# We now annotate the picture by creating a PaveText object
# and displaying the list of commands in this macro
#
fitlabel = TPaveText( 0.6, 0.3, 0.9, 0.80, 'NDC' )
fitlabel.SetTextAlign( 12 )
fitlabel.SetFillColor( 42 )
fitlabel.ReadFile(path.join(str(gROOT.GetTutorialDir()), 'pyroot', 'fit1_py.py'))
fitlabel.Draw()
c1.Update()
gBenchmark.Show( 'fit1' )
print("press enter to exit")
input()
'''


'''
import ROOT
from ROOT import TH1F, TF1,TCanvas
from ROOT import gROOT
from array import array
import heartrate
heartrate.trace(browser=True)
#c1 = TCanvas( 'c1', 'The Fit Canvas', 200, 10, 700, 500 )
x = ( 1.913521, 1.953769, 2.347435, 2.883654, 3.493567,
      4.047560, 4.337210, 4.364347, 4.563004, 5.054247,
      5.194183, 5.380521, 5.303213, 5.384578, 5.563983,
      5.728500, 5.685752, 5.080029, 4.251809, 3.372246,
      2.207432, 1.227541, 0.8597788,0.8220503,0.8046592,
      0.7684097,0.7469761,0.8019787,0.8362375,0.8744895,
      0.9143721,0.9462768,0.9285364,0.8954604,0.8410891,
      0.7853871,0.7100883,0.6938808,0.7363682,0.7032954,
      0.6029015,0.5600163,0.7477068,1.188785, 1.938228,
      2.602717, 3.472962, 4.465014, 5.177035 )

np = len(x)
h = TH1F( 'h', 'Example of several fits in subranges', np, 85, 134 )
h.SetMaximum( 7 )

for i in range(np):
   h.SetBinContent( i+1, x[i] )

par = array( 'd', 9*[0.] )
g1 = TF1( 'g1', 'gaus',  85,  95 )
g2 = TF1( 'g2', 'gaus',  98, 108 )
g3 = TF1( 'g3', 'gaus', 110, 121 )

total = TF1( 'total', 'gaus(0)+gaus(3)+gaus(6)', 85, 125 )
total.SetLineColor( 2 )
h.Fit( g1, 'R' )
h.Fit( g2, 'R+' )
h.Fit( g3, 'R+' )

par1 = g1.GetParameters()
par2 = g2.GetParameters()
par3 = g3.GetParameters()

par[0], par[1], par[2] = par1[0], par1[1], par1[2]
par[3], par[4], par[5] = par2[0], par2[1], par2[2]
par[6], par[7], par[8] = par3[0], par3[1], par3[2]

total.SetParameters( par )
h.Fit( total, 'R+' )
c1.saveas("test00.png")
'''



'''
file_path = "F:\\work\\test-proton-vertical-full-1000GeV_n-simdigi.root"
f = ROOT.TFile(str(file_path), "READ")
tree = f.Get("events")
total = 0
'''
'''
#for i in range(tree.GetEntries()):
num = 557
tree.GetEntry(num)
#tree.GetEntry(15)
branch = tree.GetBranch("fithits.trackID")
n_trackIDs = branch.GetLeaf("fithits.trackID").GetLen()
trackIDs = [branch.GetLeaf("fithits.trackID").GetValue(i) for i in range(n_trackIDs)]
print("TrackIDs for event", num, "are:", trackIDs)
non_count = 0
count = 0
for i in range(0,tree.GetEntries()):
    tree.GetEntry(i)
    branch = tree.GetBranch("fithits.trackID")
    n_trackIDs = branch.GetLeaf("fithits.trackID").GetLen()
    trackIDs = [branch.GetValue(i) for i in range(n_trackIDs)]
    if not any(trackIDs is None):
        non_count += 1
    else:
        count += 1
'''

'''
'''
'''
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

model = EdgeClassifier()
model.load_state_dict(torch.load("F:\\work\\model_parameter0.pth"))
device = cerndgl.get_device()
model.to(device)
model.eval()
print("model loaded")
dataset = Make_Graphs("F:\\work\\muon_10GeV_noCalo_comp_vertical-center.g4mac.root",data_part="val")
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
    logit_classes = logits.argmax(dim=1)
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
xy_array = np.array(xy_pred_list)
hist = ROOT.TH1F("hist", "Data Histogram", 500, min(xy_pred_list), max(xy_pred_list))
for val in xy_pred_list:
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
#c1.SaveAs("histogram_with_threshold.png")

'''
'''

graph = None
for g, l in val_loader:
    if l in [659,666,674,655,679]:
        graph,labels = g,l
        count += 1
        correct, delta_xy, delta_xz, delta_yz, delta_tot_edep = resolution(model, graph, labels, graph.ndata['feat'], graph.edata['weight'], graph.edata['label'], graph.edata['theta'], graph.edata['phi'])
        print(f"labels is {labels}",graph.edata['label'])

accuracy_list.append(correct)
delta_tot_edep_list.append(delta_tot_edep)
delta_xy_list.append(delta_xy)
delta_xz_list.append(delta_xz)
delta_yz_list.append(delta_yz)
'''
'''
delta_xy_list = [x for x in delta_xy_list if x is not None]
delta_xz_list = [x for x in delta_xz_list if x is not None]
delta_yz_list = [x for x in delta_yz_list if x is not None]
print("delta_xy_list",delta_xy_list,count,)
'''
'''

dataset = Make_Graphs("F:\\work\\test-proton-vertical-full-1000GeV_n-simdigi.root")
loader = GraphDataLoader(dataset, batch_size=1, shuffle=True)
count = 0
num = 0
labels_list = []

for graph, labels in loader:
    label = graph.edata['label']
    if torch.all(label == 1):
        count += 1
    num += 1
    labels_list.append(labels.flatten())

print("Number of graphs where all edge labels are 1:", count)
print("Total number of graphs:", num)

'''
'''
for entries in range(20):
    tree0 = tree.GetEntry(entries)
    branch = tree.GetBranch("fithits.pdgID")
    n_PdgIDs = branch.GetLeaf("fithits.PdgID").GetLen()
    PdgID = [branch.GetLeaf("fithits.pdgID")for i in range(n_PdgIDs)]
    print("PdgID for event", entries, "are:", PdgID)
'''
'''
import uproot
import awkward as ak
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import uproot
import numpy as np

file_path = "F:\\work\\muon_10GeV_noCalo_comp_vertical-center-nX.g4mac.root"
with uproot.open(file_path) as file:
    # 获取TTree
    tree = file["events"]
    e = 151
    # 读取特定条目的数据
    data = tree.arrays(
        ["fithits/fithits.pos.x","fithits/fithits.pos.y","fithits/fithits.pos.z", "fithits/fithits.trackID"],
        entry_start=e,  # 你可以指定特定的条目号
        entry_stop=e+1,  # 例如读取前10个条目
        library="ak"
    )

    # 筛选出trackID为3的pos.z值
    pos_z_id_3 = data["fithits/fithits.pos.z"][data["fithits/fithits.trackID"] == 3]
    pos_x_id_3 = data["fithits/fithits.pos.x"][data["fithits/fithits.trackID"] == 3]
    pos_z_id_3_np = pos_z_id_3.to_numpy()
    pos_x_id_3_np = pos_x_id_3.to_numpy()
    print(pos_x_id_3_np)
'''
''''''
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义图模块
class GraphModule(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GraphModule, self).__init__()
        self.layer = nn.Linear(in_feats, out_feats)
        self.gru = nn.GRUCell(out_feats, out_feats)

    def forward(self, g, h, edge_weights):
        with g.local_scope():
            g.ndata['h'] = h
            g.edata['w'] = edge_weights
            g.update_all(self.message_func, self.reduce_func)
            h_new = g.ndata['h']
            h = self.gru(h_new, h)  # 使用GRU更新隐藏表示
            return h

    def message_func(self, edges):
        return {'m': edges.src['h'] * edges.data['w']}

    def reduce_func(self, nodes):
        return {'h': torch.sum(nodes.mailbox['m'], dim=1)}

# 定义边模块
class EdgeModule(nn.Module):
    def __init__(self, in_feats):
        super(EdgeModule, self).__init__()
        self.edge_module = nn.Linear(in_feats * 2, 1)

    def forward(self, edges):
        h_u = edges.src['h']
        h_v = edges.dst['h']
        edge_h = torch.cat([h_u, h_v], dim=1)
        return {'w': torch.sigmoid(self.edge_module(edge_h))}

# 定义整个图神经网络模型
class ParticleGraphNetwork(nn.Module):
    def __init__(self, in_feats, hidden_feats, num_layers):
        super(ParticleGraphNetwork, self).__init__()
        self.input_module = nn.Linear(in_feats, hidden_feats)
        self.edge_module = EdgeModule(hidden_feats)
        self.layers = nn.ModuleList([
            GraphModule(hidden_feats, hidden_feats) for _ in range(num_layers)
        ])
        # 修改输出层以适应边的预测
        self.edge_predictor = nn.Linear(hidden_feats * 2, 1)

    def forward(self, g, features):
        h = self.input_module(features)
        g.ndata['h'] = h

        for layer in self.layers:
            g.apply_edges(self.edge_module)
            edge_weights = g.edata['w']
            h = layer(g, h, edge_weights)
            g.ndata['h'] = h

        # 收集边的源节点和目标节点的特征
        edges_h = torch.cat([g.ndata['h'][g.edges()[0]], g.ndata['h'][g.edges()[1]]], dim=1)
        # 预测每条边的存在概率
        edge_pred = self.edge_predictor(edges_h).squeeze()

        return edge_pred
'''
# 示例参数
in_feats = 5  # 输入特征的维度
hidden_feats = 64  # 隐藏层的维度
num_classes = 2  # 输出类别的数量
num_layers = 8 # 网络层数

# 创建模型实例
model = ParticleGraphNetwork(in_feats, hidden_feats, num_classes, num_layers)

# 示例图和特征

g = dgl.graph((torch.tensor([0, 1, 2]), torch.tensor([1, 2, 3])))
features = torch.randn(4, in_feats)  # 随机生成特征

# 模型前向传播
output = model(g, features)
print("feature",features)
print("graph",g)
print(output)
'''
