import matplotlib
matplotlib.use('Agg') 
from make_graph import Make_Graphs
import dgl
import torch
import torch.nn as nn
import dgl.nn as dglnn
import ROOT
import numpy as np
import torch.nn.functional as F
from dgl.nn import GraphConv
#from real_final_class import Making_Graphs
from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from dgl.data.utils import split_dataset
from sklearn.metrics import f1_score, precision_recall_curve, auc, f1_score, accuracy_score
import CERN_EdgeNetworkModel_DGL
from CERN_EdgeNetworkModel_DGL import EdgeClassifier


#import matplotlib.pyplot as plt
import dgl.function as fn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Contruct a two-layer GNN model
class SAGE(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super().__init__()
        self.conv1 = dglnn.SAGEConv(
            in_feats=in_feats, out_feats=hid_feats, aggregator_type='mean')
        self.conv2 = dglnn.SAGEConv(
            in_feats=hid_feats, out_feats=out_feats, aggregator_type='mean')

    def forward(self, graph, inputs):
        # inputs are features of nodes
        h = self.conv1(graph, inputs)
        h = F.relu(h)
        h = self.conv2(graph, h)
        return h

# dot product of incident node representations on each edge
class DotProductPredictor(nn.Module):
    def forward(self, graph, h, e):
        # h contains the node representations computed from the GNN defined
        # in the node classification section (Section 5.1).
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            return graph.edata['score']

#a prediction function that predicts a vector for each edge with an MLP
class MLPPredictor(nn.Module):
    def __init__(self, in_features, out_classes):
        super().__init__()
        self.W = nn.Linear(in_features * 2, out_classes)

    def apply_edges(self, edges):
        h_e = edges.data['h']
        h_u = edges.src['h']
        h_v = edges.dst['h']
        #score = self.W(torch.cat([h_e, h_u, h_v], -1)) #TODO
        score = self.W(torch.cat([ h_u, h_v], 1))
        return {'score': score}

    def forward(self, graph, h, e):
        # h contains the node representations computed from the GNN defined
        # in the node classification section (Section 5.1).
        with graph.local_scope():
            graph.ndata['h'] = h
            #graph.edata['h'] = graph.edata['weight']
            graph.edata['h'] = e #TODO
            graph.apply_edges(self.apply_edges)
            return graph.edata['score']

# an example from GAS model
# https://gitee.com/liu-runda/dgl/blob/master/examples/pytorch/gas/model.py
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim)

    def apply_edges(self, edges):
        h_e = edges.data['h']
        h_u = edges.src['h']
        h_v = edges.dst['h']
        score = self.W(torch.cat([h_e, h_u, h_v], -1))
        return {'score': score}

    def forward(self, g, e_feat, u_feat, v_feat):
        with g.local_scope():
            g.edges['forward'].data['h'] = e_feat
            g.nodes['u'].data['h'] = u_feat
            g.nodes['v'].data['h'] = v_feat
            g.apply_edges(self.apply_edges, etype="forward")
            return g.edges['forward'].data['score']



#node representation computation model and an edge predictor model
class Model(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
    #def __init__(self, in_features, hidden_features, out_features, out_classes):
        super().__init__()
        self.sage = SAGE(in_features, hidden_features, out_features)
        #self.pred = DotProductPredictor()
        #self.pred = MLPPredictor(in_features, out_classes)
        self.pred = MLPPredictor(in_features, 3)
    def forward(self, g, x, e):
        h = self.sage(g, x)
        return self.pred(g, h, e)

"""
node_features = edge_pred_graph.ndata['feature']
edge_label = edge_pred_graph.edata['label']
train_mask = edge_pred_graph.edata['train_mask']
model = Model(10, 20, 5)
opt = torch.optim.Adam(model.parameters())
for epoch in range(10):
    pred = model(edge_pred_graph, node_features)
    loss = ((pred[train_mask] - edge_label[train_mask]) ** 2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    print(loss.item())
"""


dataset = Make_Graphs()
for event_number in range(len( dataset )):
        graph, label = dataset[ event_number ]
        #print(label , graph )

#print ( dataset )

num_examples = len( dataset )
num_train = int( num_examples * 0.8)

#//train_sampler = SubsetRandomSampler( torch.arange( num_train ) )
#//test_sampler = SubsetRandomSampler( torch.arange( num_train, num_examples ))

#//train_dataloader = GraphDataLoader( dataset, sampler = train_sampler, batch_size = 32, drop_last = False )
#//test_dataloader = GraphDataLoader ( dataset, sampler = test_sampler, batch_size = 1, drop_last = False )

#//trainset, testset = split_dataset(dataset, frac_list=[0.8, 0.1,0.1])
trainset, valset, testset = split_dataset(dataset, frac_list=[0.8, 0.1,0.1])
trainLoader = GraphDataLoader(trainset, batch_size=32, shuffle=True)
testLoader = GraphDataLoader(testset, batch_size=1, shuffle=True)


#print("train dataloader shape: ", train_dataloader)
#print("test dataloader: ", test_dataloader)

it = iter( trainLoader )

batch = next( it )
print( "batch is ",batch )

batched_graph, labels = batch

#print('Number of nodes for each graph element in the batch :', batched_graph.batch_num_nodes() )
#print('Number of edges for each graph element in the batch :', batched_graph.batch_num_edges() )

# Recover the original graph elements from the minibatch
#graphs = dgl.unbatch( batched_graph )
#print(len(graphs))
#print ('The original graphs in the minibatch :')



#model = GCN(4 , 10 , 2)
#model = GCN(4, 16, 2)
#model = SAGE(4 , 10 , 2)
#model = Model(4 , 10 , 4)
model = EdgeClassifier(4 , 4, 64, 2)
#将model dataset 转移到GPU
model.to(device)

optimizer = torch.optim.Adam( model.parameters() , lr =0.01)

loss_list = []
accuracy_list = []
nepoch=100

for epoch in range(nepoch):
    model.train()
    total_loss = 0
    total = 0
    correct = 0
    #for batched_graph, labels in train_dataloader:
    for batched_graph, labels in trainLoader:
        #print("batched graph: ", batched_graph)

        node_features = batched_graph.ndata['feat']
        node_labels = batched_graph.ndata['label']
        train_mask = batched_graph.ndata['train_mask']
        valid_mask = batched_graph.ndata['val_mask']
        test_mask = batched_graph.ndata['test_mask']

        train_edge_mask = batched_graph.edata['train_mask']
        valid_edge_mask = batched_graph.edata['val_mask']
        test_edge_mask = batched_graph.edata['test_mask']
        edge_labels = batched_graph.edata['label']


        batched_graph = batched_graph.to(device)
        features = batched_graph.ndata['feat'].to(device)
        efeature = batched_graph.edata['weight'].to(device)
        labels = batched_graph.edata['label'].to(device)
        test_mask = batched_graph.edata['test_mask'].to(device)

        n_features = node_features.shape[1]
        n_labels = int(node_labels.max().item() + 1)

    #for event_number in range(len( dataset )):
        #graph, label = dataset[ event_number ]
        #pred = model( batched_graph, batched_graph.ndata['feat'], batched_graph.edata['weight'] )
        #pred = model( batched_graph)
        pred = model( batched_graph, batched_graph.ndata['feat'].float() )
        #loss = F.cross_entropy( pred, torch.tensor( labels ) )
        #print("pred\n", pred)
        #print("pred masks\n", pred[train_edge_mask])
        #print("node masks\n", node_labels[train_mask])
        #print("edge masks\n", edge_labels[train_edge_mask])
        #loss = F.cross_entropy( pred[train_mask], node_labels[train_mask] ) #mx
        #loss = F.cross_entropy( pred[train_mask], edge_labels[train_mask] ) #mx
        #loss = ((pred[train_edge_mask] - edge_labels[train_edge_mask]) ** 2).mean()
        #loss = F.cross_entropy(pred[train_edge_mask] , edge_labels[train_edge_mask])
        #print(pred)
        #print(train_edge_mask)
        #print(pred[train_edge_mask])
        #print(len(pred[train_edge_mask]))
        #print(edge_labels[train_edge_mask])
        #print(len(edge_labels[train_edge_mask]))
        #loss = F.cross_entropy(pred[train_edge_mask] , edge_labels[train_edge_mask])
        edge_labels = edge_labels.long()
        edge_labels = edge_labels.to(device)

        loss = F.cross_entropy(pred , edge_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pred = pred.argmax(dim=1)
        correct += (pred == edge_labels).sum().item()
        total += len(pred)

        #print('Epoch %d | Loss: %.4f' % (epoch, loss.item()))
        loss_list.append(loss.item())
    train_loss = total_loss / len(trainLoader)
    train_acc = correct / total

    #print('Epoch %d', )
    print(f'Epoch: {epoch}, Train Loss : {train_loss:.4f}, Train Accuracy: {train_acc:.3f}')
    #print(f'Train Accuracy: {train_acc:.3f}')
#!将参数输出
torch.save(model.state_dict(),"model_parameter.pth")
x1=range(0, nepoch)
y1=loss_list
#plt.subplot(2,2,1)
#plt.plot(x1, y1, "o-")
#plt.show()
#use ROOT make canvas
c1 = ROOT.TCanvas("c1","Loss and Accuracy",1200,1600)
c1.Divide(2,2)
c1.cd(1)  # 切换到第一个子画布
x_values = np.arange(0,nepoch,dtype=np.float64)  
y_values = np.array(loss_list, dtype=np.float64)        # y values as float64 array
loss_graph = ROOT.TGraph(len(x_values), x_values, y_values)
loss_graph.SetTitle("Training Loss;Epoch;Loss")
loss_graph.SetMarkerStyle(20)
loss_graph.SetDrawOption("AP")
loss_graph.Draw("APL")  # APL means axis, points, and line

num_correct = 0
num_tests = 0
def evaluate(model2, graph, features, efeature, labels, mask):
        #logits = model2(graph, features, efeature)
        model.to(device)
        graph = graph.to(device)
        features = features.to(device)
        efeatures = features.to(device)
        labels = labels.to(device)
        mask = mask.to(device)
        logits = model2(graph, features)
        logits = logits[mask]
        labels = labels[mask]
        #logits = logits
        #labels = labels
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        ##print("feature\n", features)
        ##print("mask\n", mask)
        ##print("logits", logits)
        ##print("indices\n", indices)
        ##print("number of test edges", len(labels))
        ##print("input labels: ", labels)
        ##print("correct\n", correct)
        #print("predicated  nodes: ", indices)
        ##print("torch.max\n", torch.max(logits, dim=1))
        ##print("len lables\n", len(labels))
        #print("accuracy:", correct.item() * 1.0 / len(labels))
        return correct.item() * 1.0 / len(labels)

#for batched_graph, labels in test_dataloader:
for batched_graph, labels in testLoader:
    #pred = model( batched_graph, batched_graph.ndata['feat'].float () )
    #print(batched_graph)
    #unbatched = dgl.unbatch(batched_graph)
    #print(unbatched)
    #pred = logits.argmax(1)
    #print("predectided pos:", pred)
    #print("predectided2 pos:", pred.argmax(1))
    #print("input pos:", batched_graph.ndata['feat'].float())
    #accuracy_list.append( evaluate(model, batched_graph, batched_graph.ndata['feat'].float(), batched_graph.ndata['label'], batched_graph.ndata['test_mask']) )
    accuracy_list.append( evaluate(model, batched_graph, batched_graph.ndata['feat'], batched_graph.edata['weight'], batched_graph.edata['label'], batched_graph.edata['test_mask']) )

for batched_graph, labels in testLoader:
    with torch.no_grad():
        #scores = model(batched_graph, batched_graph.ndata['feat'].float(), batched_graph.edata['weight'])
        batched_graph = batched_graph.to(device)
        batched_graph.ndata['feat'] = batched_graph.ndata['feat'].to(device)
        scores = model(batched_graph, batched_graph.ndata['feat'].float())
        labels = batched_graph.edata['label']
        #preds = (scores[:, 1] > 0.5).long()
        preds = (scores > 0.5).long()
        #f1 = f1_score(labels.numpy(), preds.numpy())
        #f1 = f1_score(labels, scores)
        #print("labels:", labels.numpy())
        #print(scores)
        #print("argmax:",scores.argmax(dim=1))
        #print("pred:", preds.numpy())
        #print(f1)

x2 = range(0, len(accuracy_list) )
#print(x2)
y2= accuracy_list
#plt.subplot(2,2,2)
#plt.plot(x2, y2, "o-")
##plt.show()
#plt.subplot(2,2,3)
#plt.hist(accuracy_list, bins=100, density=True, alpha=0.5, color='b')
#plt.xlabel('x')
#plt.ylabel('Frequency')
#plt.title('Histogram')
#
#plt.savefig("test10Gev.png")
#plt.show()
c1.cd(2)  # 切换到第二个子画布
x_values = np.arange(len(accuracy_list), dtype=np.float64)  # x values as float64 array
y_values = np.array(accuracy_list, dtype=np.float64)        # y values as float64 array
accuracy_graph = ROOT.TGraph(len(x_values), x_values, y_values)
accuracy_graph.SetTitle("Test Accuracy;Test Sample;Accuracy")
accuracy_graph.SetMarkerStyle(20)
accuracy_graph.SetDrawOption("AP")
accuracy_graph.Draw("APL")
c1.cd(3)
accuracy_hist = ROOT.TH1F("accuracy_hist", "Test Accuracy;Accuracy;Frequency", 100,min(y_values), max(y_values))
for i in range(len(accuracy_list)):
    accuracy_hist.Fill(accuracy_list[i])
accuracy_hist.Draw()
print("ntrainLoader: ", len(trainLoader))
print("ntestLoader: ", len(testLoader))
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
            edge_labels =  batched_graph.edata['label'].long().to(device)
            batched_graph = batched_graph.to(device)
            out = model(batched_graph, features.float())
            y_true += edge_labels.cpu().numpy().tolist()
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
acc, f1, precision, recall, area = test(model, device,testLoader)
c1.cd(4)
x_values = np.array(recall, dtype=np.float64)
y_values = np.array(precision, dtype=np.float64)
auc_graph = ROOT.TGraph(len(x_values), x_values, y_values)
auc_graph.SetTitle("AUC;Recall;Precision")
auc_graph.SetMarkerStyle(20)
auc_graph.SetDrawOption("AP")
auc_graph.Draw("APL")
c1.Update()
c1.SaveAs("training_100GeV.png")
#for event_number in range(len( dataset )):
    #graph, label = dataset[ event_number ]
    #pred = model( graph, graph.ndata['feat'].float () )
    #num_correct += ( pred.argmax(1) == labels ).sum().item()
    #num_correct += ( pred.argmax(1) == batched_graph.ndata['labels'] ).sum().item() ##mx
    #num_tests += len( labels )

#print ('Test accuracy:', num_correct / num_tests )
