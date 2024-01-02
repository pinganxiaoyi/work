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
from fit_angle import selected_angle,fit_angle

#//import matplotlib.pyplot as plt
import dgl.function as fn

#读数据 生成图
dataset = Make_Graphs("F:\\work\\test-proton-vertical-full-100GeV_n-simdigi.root")
#print ( dataset )

#设置数据集验证集测试集
trainset, valset, testset = split_dataset(dataset, frac_list=[0.7, 0.1,0.2])
#train_loader, val_loader, test_loader = cerndgl.get_data_loaders(trainset, valset, testset, batch_size=32)
train_loader = GraphDataLoader(trainset, batch_size=32,shuffle=True)
test_loader = GraphDataLoader(testset, batch_size=1,shuffle=True)
val_loader = GraphDataLoader(valset,batch_size = 1,shuffle = True)
device = cerndgl.get_device()

#设置模型
model = EdgeClassifier()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)#?准确率之前不太行
loss_list = []
accuracy_list = []
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
        for batched_graph, labels in train_loader:  #?进度条
        #计算损失
        # 更新每个批次的特征
                batched_graph = batched_graph.to(device)
                features = batched_graph.ndata['feat'].to(device)
                edge_labels = batched_graph.edata['label'].long().to(device)
                pred = model(batched_graph,features.float())#model只接受graph和点特征作为输入 边特征在forward里面进行了提取 
 
                loss = F.cross_entropy(pred, edge_labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        #计算准确率
                total_loss += loss.item()
                pred = pred.argmax(dim = 1)
                #print(pred)
                correct += torch.sum(pred == edge_labels).item()
                total += len(pred) 
        #//loss_list.append(loss.item())
        train_loss = total_loss / len(train_loader)#平均损失
        train_acc = correct / total
        #loss_list.append(loss.item())
        loss_list.append(train_loss)
#保存参数输出图像
torch.save(model.state_dict(),"model_parameter.pth")
x1=range(0, nepoch)
y1=loss_list
c1 = ROOT.TCanvas("c1","Loss and Accuracy",1200,1800)
c1.Divide(2,2)
c1.cd(1)
x_values = np.arange(0,nepoch, dtype=np.float64)  # x values as float64 array
y_values = np.array(loss_list, dtype=np.float64)        # y values as float64 array
loss_graph = ROOT.TGraph(len(x_values), x_values, y_values)
loss_graph.SetTitle("Training Loss;Epoch;Loss")
loss_graph.SetMarkerStyle(20)
loss_graph.SetDrawOption("AP")
loss_graph.Draw("APL")  # APL means axis, points, and line


#定义一个计算角分辨的函数
#def calculate_anglur_resolution(theta_pred,phi_pred,theta_true,phi_true):
#    theta
#自定义的评价函数
def evaluate(model2, graph,labels, features, efeatures,label,theta,phi):
        #logits = model2(graph, features, efeature)
        model2.to(device)
        graph = graph.to(device)
        labels = graph.to(device) #graph's Number 
        features = features.to(device)
        efeatures = efeatures.to(device)
        label = label.to(device)
        theta = theta.to(device)
        phi = phi.to(device)
        logits = model2(graph, features)
        #logits = logits
        #labels = labels
        logit_classes = logits.argmax(dim=1)
        edges = graph.edges()
        src_nodes,dst_nodes = edges[0].cpu().numpy(),edges[1].cpu().numpy()
        selected_edges = []
        for i in range(len(logit_classes)):
            if logit_classes[i].item() == 0:
                selected_edges.append((src_nodes[i], dst_nodes[i]))
        xy_pred, xz_pred, yz_pred, tot_edep_pred  = selected_angle(graph,selected_edges)
        xy_truth, xz_truth, yz_truth, tot_edep_truth, err_truth = fit_angle(graph,labels)
        #角分辨
        delta_xy = xy_pred - xy_truth
        delta_xz = xz_pred - xz_truth
        delta_yz = yz_pred - yz_truth
        delta_tot_edep = tot_edep_pred - tot_edep_truth
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == label)
        correct.item()*1.0 / len(label)
        return correct,delta_xy,delta_xz,delta_yz,delta_tot_edep

# 在 evaluate 函数外部调用


for graph, labels in val_loader:
    accuracy_list.append( evaluate(model, graph, labels,graph.ndata['feat'], graph.edata['weight'], graph.edata['label'],graph.edata['theta'],graph.edata['phi']) )
    #计算角分辨率
    
    

    
print("Total test samples:", len( batched_graph.ndata['feat']))
print("Batch size:", test_loader.batch_size)
print("Total batches:", len(test_loader))
print("Entries in accuracy_list:", len(accuracy_list))
       
#画图
c1.cd(3)  # 切换到第4个子画布
x_values = np.arange(len(accuracy_list), dtype=np.float64)  # x values as float64 array
y_values = np.array(accuracy_list, dtype=np.float64)        # y values as float64 array
accuracy_graph = ROOT.TGraph(len(x_values), x_values, y_values)
accuracy_graph.SetTitle("Accuracy with DIY;Test Sample;Accuracy")
accuracy_graph.SetMarkerStyle(20)
accuracy_graph.SetDrawOption("AP")
accuracy_graph.Draw("AP")      

#画直方图
c1.cd(4)
accuracy_hist = ROOT.TH1F("accuracy_hist", "Test Accuracy;Accuracy;Frequency", 100,min(y_values), max(y_values))
for i in range(len(accuracy_list)):
    accuracy_hist.Fill(accuracy_list[i])
accuracy_hist.Draw()
#测试
def test(model,device,loader):

    print('Testing The Model')

    model.eval()
    y_true = []
    y_probas = []
    y_pred = []

    with torch.no_grad():
        for batched_graph, labels in loader:
            features = batched_graph.ndata['feat'].to(device)
            labels = labels.to(device)
            edge_labels = batched_graph.edata['label'].long().to(device)
            batched_graph = batched_graph.to(device)
            out = model(batched_graph, features.float())
            y_true += edge_labels.cpu().numpy().tolist()  # 使用 extend 而不是 append
            y_pred += out.argmax(dim=1).cpu().numpy().tolist()  # 使用 extend
            y_probas += out[:, 1].cpu().numpy().tolist() 

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    precision, recall, thresholds = precision_recall_curve(y_true, y_probas)
    area = auc(recall, precision)

    print('\nResults\n')
    print(f'Testing Accuracy {acc:.3f}')
    print(f'F1 score: {f1:.3f}')
    print(f'AUCPR: {area:.3f}\n')

    return acc, f1, precision, recall, area
acc, f1, precision, recall, area = test(model, device,test_loader)

c1.cd(2)
x_values = np.array(recall, dtype=np.float64)
y_values = np.array(precision, dtype=np.float64)
auc_graph = ROOT.TGraph(len(x_values), x_values, y_values)
auc_graph.SetTitle("AUC;Recall;Precision")
auc_graph.SetMarkerStyle(20)
auc_graph.SetDrawOption("AP")
legend = ROOT.TLegend(0.1, 0.7, 0.3, 0.9)
n_points = auc_graph.GetN()
legend.AddEntry(auc_graph, "AUC, points: %d" % n_points, "lp")
legend.Draw()
auc_graph.Draw("APL")
c1.SetLeftMargin(3)# 增加左边距

c1.Update()
c1.SaveAs("1357layer1000GeV.png")

"""
c2 = ROOT.TCanvas("c2", "LOSS", 1200, 1800)
x_values = np.arange(0, nepoch, dtype=np.float64)  # x values as float64 array
y_values = np.array(loss_list, dtype=np.float64)   # y values as float64 array

# 创建一个空的TGraph对象
filtered_loss_graph = ROOT.TGraph()

# 设置图形的标题、标记样式和绘图选项
filtered_loss_graph.SetTitle("Training Loss;Epoch;Loss")
filtered_loss_graph.SetMarkerStyle(20)
filtered_loss_graph.SetDrawOption("AP")

# 遍历y_values和x_values，仅添加满足条件的数据点到filtered_loss_graph
for i in range(len(y_values)):
    if y_values[i] < 0.5:
        filtered_loss_graph.SetPoint(filtered_loss_graph.GetN(), x_values[i], y_values[i])

# 绘制满足条件的部分
filtered_loss_graph.Draw("APL")

# 显示图形
c2.Draw()
c2.SaveAs("loss2.png")
"""