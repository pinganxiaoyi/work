import matplotlib
matplotlib.use('Agg')
import sys
# 添加包含 make_graph.py 的目录到模块搜索路径
import dgl
import torch
import torch.nn as nn
import dgl.nn as dglnn
import ROOT
import numpy as np
import torch.nn.functional as F
#from dgl.nn import GraphConv
#from real_final_class import Making_Graphs
from dgl.dataloading import GraphDataLoader
from dgl.data.utils import split_dataset
#from sklearn.metrics import f1_score, precision_recall_curve, auc, f1_score, accuracy_score
import CERN_EdgeNetworkModel as cerndgl
from CERN_EdgeNetworkModel import EdgeClassifier
#from try0 import ParticleGraphNetwork
#//import matplotlib.pyplot as plt
import dgl.function as fn
sys.path.append('F:\\work')

# 现在可以导入 Make_Graphs 函数
from make_graph import Make_Graphs
from fit_angle import selected_angle,fit_angle
#读数据 生成图
dataset = Make_Graphs("F:\\work\\muon_10GeV_noCalo_comp_vertical-center.g4mac.root",data_part="val")
#print ( dataset )

#设置数据集验证集测试集
trainset, valset, testset = split_dataset(dataset, frac_list=[0.7, 0.2,0.1])
#train_loader, val_loader, test_loader = cerndgl.get_data_loaders(trainset, valset, testset, batch_size=32)
train_loader = GraphDataLoader(trainset, batch_size=32,shuffle=True)
test_loader = GraphDataLoader(testset, batch_size=1,shuffle=True)
val_loader = GraphDataLoader(valset,batch_size = 1,shuffle = True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#print(f"Number of graphs in validation set: {len(valset)}")
#print(f"Number of graphs in test set: {len(testset)}")
#--另外文件验证
#dataset2 = Make_Graphs("F:\\work\\test-helium-vertical-full-10GeV_n-simdigi.root")
#设置模型

model = EdgeClassifier()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)#?准确率之前不太行
loss_list = []
accuracy_list = []
delta_xy_list = []
delta_xz_list = []
delta_yz_list = []
delta_tot_edep_list = []
nepoch=50

#分批次
it = iter(train_loader)
batch = next(it)
batched_graph, labels = batch

#转移到GPU
model = model.to(device)

#训练
for epoch in range(nepoch):
        model.train()

        total_loss = 0
        total = 0
        correct = 0
        total_graphs = 0        
        for batched_graph, labels in train_loader:  #//进度条
        #计算损失
        # 更新每个批次的特征
                total_graphs += batched_graph.batch_size
                batched_graph = batched_graph.to(device)
                features = batched_graph.ndata['feat'].to(device)
                edge_labels = batched_graph.edata['label'].long().to(device)
                pred = model(batched_graph,features.float())#model只接受graph和点特征作为输入 边特征在forward里面进行了提取 
 
                loss = torch.nn.BCEWithLogitsLoss(pred, edge_labels.float())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        #计算准确率
                total_loss += loss.item()
                pred = torch.sigmoid(pred)
                #print(pred)
                correct += torch.sum(pred == edge_labels).item()
                total += len(pred) 
        #//loss_list.append(loss.item())
        train_loss = total_loss / len(train_loader)#平均损失
        train_acc = correct / total
        #loss_list.append(loss.item())
        loss_list.append(train_loss)
        print(f"Epoch {epoch} | Train Loss {train_loss:.4f} | Train Acc {train_acc:.4f} | Total graphs {total_graphs}")
#保存参数输出图像
torch.save(model.state_dict(),"model_parameter0.pth")

total_nodes = 0
total_edges = 0
none_count = 0
#自定义的评价函数
def resolution(model2, graph,labels, features, efeatures,label):
    #logits = model2(graph, features, efeature)
    model2.to(device)
    graph = graph.to(device)
    labels = graph.to(device) #graph's Number 
    features = features.to(device)
    efeatures = efeatures.to(device)
    label = label.to(device)
    print("OK1")
    logits = model2(graph, features)
    print("OK2")
    #logits = logits
    #labels = labels
    logit_classes = torch.sigmoid(logits) 
    edges = graph.edges()
    correct = torch.sum(logit_classes == label)
    correct = correct.item() *1.0 / len(label) 
    src_nodes,dst_nodes = edges[0],edges[1]
    selected_edges = []
    print("OK3")
    for i in range(len(logit_classes)):
        if logit_classes[i].item() ==0:
            selected_edges.append(( src_nodes[i], dst_nodes[i]))
    xy_pred, xz_pred, yz_pred, tot_edep_pred  = selected_angle(graph,selected_edges)
    print("OK4")
    xy_truth, xz_truth, yz_truth, tot_edep_truth, err_truth = fit_angle(graph,labels)
    print("OK5")
    #角分辨# 检查预测值和真实值是否有效
    global none_count  
    if xy_pred is None  or xz_pred is None or yz_pred is None  or tot_edep_pred is None :
        none_count += 1  
        return correct, None, None, None, None  # 返回 None 值

    # 计算 delta 值
    delta_xy = xy_pred - xy_truth if xy_pred is not None and xy_truth is not None else None
    delta_xz = xz_pred - xz_truth if xz_pred is not None and xz_truth is not None else None
    delta_yz = yz_pred - yz_truth if yz_pred is not None and yz_truth is not None else None
    delta_tot_edep = tot_edep_pred - tot_edep_truth if tot_edep_pred is not None and tot_edep_truth is not None else None

    global total_nodes , total_edges
    total_nodes += graph.number_of_nodes()
    total_edges += graph.number_of_edges()
    print(f"Number of nodes: {graph.number_of_nodes()}, Number of edges: {graph.number_of_edges()}")
    return correct,delta_xy,delta_xz,delta_yz,delta_tot_edep

# 在 evaluate 
count = 0
for graph, labels in val_loader:
    correct, delta_xy, delta_xz, delta_yz, delta_tot_edep = resolution(model, graph, labels, graph.ndata['feat'], graph.edata['weight'], graph.edata['label'])
    #计算角分辨率
    accuracy_list.append(correct)
    delta_tot_edep_list.append(delta_tot_edep)
    delta_xy_list.append(delta_xy)
    delta_xz_list.append(delta_xz)
    delta_yz_list.append(delta_yz)
    count += 1

delta_xy_list = [x for x in delta_xy_list if x is not None]
delta_xz_list = [x for x in delta_xz_list if x is not None]
delta_yz_list = [x for x in delta_yz_list if x is not None]
print("delta_xy_list",delta_xy_list)

x1=range(0, nepoch)
y1=loss_list
c1 = ROOT.TCanvas("c1","Loss and resolution",1200,1800)
c1.Divide(2,2)
c1.cd(1)
x_values = np.arange(0,nepoch, dtype=np.float64)  # x values as float64 array
y_values = np.array(loss_list, dtype=np.float64)        # y values as float64 array
loss_graph = ROOT.TGraph(len(x_values), x_values, y_values)
loss_graph.SetTitle("Training Loss;Epoch;Loss")
loss_graph.SetMarkerStyle(20)
loss_graph.SetDrawOption("AP")
loss_graph.Draw("APL")  # APL means axis, points, and line

c1.cd(2)  
x_values = np.arange(len(accuracy_list), dtype=np.float64)  # x values as float64 array
y_values = np.array(accuracy_list, dtype=np.float64)        # y values as float64 array
accuracy_graph = ROOT.TGraph(len(x_values), x_values, y_values)
accuracy_graph.SetTitle("Accuracy with DIY;Test Sample;Accuracy")
accuracy_graph.SetMarkerStyle(20)
accuracy_graph.SetDrawOption("AP")
accuracy_graph.Draw("AP")      
c1.Update()

c1.cd(3)
x_values = np.arange(len(delta_xy_list), dtype=np.float64)
y_values = np.array(delta_xy_list, dtype=np.float64)
graph_xy = ROOT.TGraph(len(x_values), x_values, y_values)
graph_xy.SetTitle("Delta XY;Sample Index;Delta XY Value")
graph_xy.SetMarkerStyle(20)
graph_xy.Draw("AP")
line1 = ROOT.TLine(0,0,20,0)
line1.SetLineColor(ROOT.kRed)
line1.Draw()
c1.Update()

c1.cd(4)
x_values = np.arange(len(delta_yz_list), dtype=np.float64)  # x values as float64 array
y_values = np.array(delta_yz_list, dtype=np.float64)        # y values as float64 array
graph = ROOT.TGraph(len(x_values), x_values, y_values)
graph.SetTitle("Delta_yz;Sample Index;Delta_yz Value")
graph.SetMarkerStyle(20)
graph.SetDrawOption("AP")
graph.Draw("AP") 
line1.Draw()

print("Total graphs resolution calculated:", count)
print(f"Total graphs discarded : {none_count}")
print("Total nodes processed:", total_nodes)
print("Total edges processed:", total_edges)
print("Batch size:", val_loader.batch_size)
print("Total batches:", len(val_loader))
print("Entries in accuracy_list:", len(accuracy_list))
c1.SaveAs("resolution10Gevmuon.png")       



    