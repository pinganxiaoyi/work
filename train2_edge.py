from make_graph2 import Make_Graphs
import dgl
import torch
import torch.nn as nn
import dgl.nn as dglnn
import torch.nn.functional as F
from dgl.nn import GraphConv
#from real_final_class import Making_Graphs
from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from dgl.data.utils import split_dataset
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve

from CERN_EdgeNetworkModel_DGL import EdgeClassifier 



import matplotlib.pyplot as plt
import dgl.function as fn


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

# dot product of incident node representations on each edge 点积 点的表示
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
    #print(loss.item())
"""


dataset = Make_Graphs()
for event_number in range(len( dataset )):
    graph, label = dataset[ event_number ]
        ##print(label , graph )


##print ( dataset )

num_examples = len( dataset )
num_train = int( num_examples * 0.8)

train_sampler = SubsetRandomSampler( torch.arange( num_train ) )
test_sampler = SubsetRandomSampler( torch.arange( num_train, num_examples ))

train_dataloader = GraphDataLoader( dataset, sampler = train_sampler, batch_size = 32, drop_last = False )
test_dataloader = GraphDataLoader ( dataset, sampler = test_sampler, batch_size = 1, drop_last = False )

#trainset, testset = split_dataset(dataset, frac_train=0.8, frac_val=0.1, frac_test=0.1)
trainset, valset, testset = split_dataset(dataset)
trainLoader = GraphDataLoader(trainset, batch_size=512, shuffle=True)
testLoader = GraphDataLoader(testset, batch_size=1, shuffle=True)


#print("train dataloader: ", train_dataloader)
#print("test dataloader: ", test_dataloader)

it = iter( train_dataloader )

batch = next( it )
#print( batch )

batched_graph, labels = batch

#print('Number of nodes for each graph element in the batch :', batched_graph.batch_num_nodes() )
#print('Number of edges for each graph element in the batch :', batched_graph.batch_num_edges() )

# Recover the original graph elements from the minibatch
graphs = dgl.unbatch( batched_graph )
#print(len(graphs))
##print ('The original graphs in the minibatch :')
#for i in graphs:
#    #print ( i )

#model = GCN(4 , 10 , 2)
#model = GCN(4, 16, 2)
#model = SAGE(4 , 10 , 2)
#model = Model(4 , 10 , 4)
model = EdgeClassifier(4 , 4, 64, 2)

optimizer = torch.optim.Adam( model.parameters() , lr =0.01)

loss_list = []
accuracy_list = []
nepoch=50

for epoch in range(nepoch):
    
    model.train()
    total_loss = 0
    total = 0
    correct = 0
    #for batched_graph, labels in train_dataloader:
    for batched_graph, labels in trainLoader:
        ##print("batched graph: ", batched_graph)
        node_features = batched_graph.ndata['feat']
        node_labels = batched_graph.ndata['label']
        train_mask = batched_graph.ndata['train_mask']
        valid_mask = batched_graph.ndata['val_mask']
        test_mask = batched_graph.ndata['test_mask']

        train_edge_mask = batched_graph.edata['train_mask']
        valid_edge_mask = batched_graph.edata['val_mask']
        test_edge_mask = batched_graph.edata['test_mask']
        edge_labels = batched_graph.edata['label']

        n_features = node_features.shape[1]
        n_labels = int(node_labels.max().item() + 1)
        print("valid_mask",valid_mask,'\n',"test_edge_mask",test_edge_mask,'\n')
        print("n_features",n_features,'\n',"n_labels",n_labels)
    #for event_number in range(len( dataset )):
        #graph, label = dataset[ event_number ]
        #pred = model( batched_graph, batched_graph.ndata['feat'], batched_graph.edata['weight'] )
        #pred = model( batched_graph)
        pred = model( batched_graph, batched_graph.ndata['feat'].float() )
        #loss = F.cross_entropy( pred, torch.tensor( labels ) )
        print("pred\n", pred)
        print("pred masks\n", pred[train_edge_mask])
        print("node masks\n", node_labels[train_mask])
        print("edge masks\n", edge_labels[train_edge_mask])
        #loss = F.cross_entropy( pred[train_mask], node_labels[train_mask] ) #mx
        #loss = F.cross_entropy( pred[train_mask], edge_labels[train_mask] ) #mx
        #loss = ((pred[train_edge_mask] - edge_labels[train_edge_mask]) ** 2).mean()
        #loss = F.cross_entropy(pred[train_edge_mask] , edge_labels[train_edge_mask])
        ##print(pred)
        ##print(train_edge_mask)
        ##print(pred[train_edge_mask])
        ##print(len(pred[train_edge_mask]))
        ##print(edge_labels[train_edge_mask])
        ##print(len(edge_labels[train_edge_mask]))
        #loss = F.cross_entropy(pred[train_edge_mask] , edge_labels[train_edge_mask])
        edge_labels = edge_labels.long()
        loss = F.cross_entropy(pred , edge_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('Epoch %d | Loss: %.4f' % (epoch, loss.item()))
        loss_list.append(loss.item())
        #!利用CERN_Edge的函数测试
        
x1=range(0, nepoch)
y1=loss_list
plt.subplot(2,2,1)
plt.plot(x1, y1, "o-")
#plt.show()
num_correct = 0
num_tests = 0
v

for batched_graph, labels in testLoader:
    with torch.no_grad():
        #scores = model(batched_graph, batched_graph.ndata['feat'].float(), batched_graph.edata['weight'])
        scores = model(batched_graph, batched_graph.ndata['feat'].float())
        labels = batched_graph.edata['label']
        #preds = (scores[:, 1] > 0.5).long()
        preds = (scores > 0.5).long()
        #f1 = f1_score(labels.numpy(), preds.numpy())
        #f1 = f1_score(labels, scores)
        ##print("labels:", labels.numpy())
        ##print(scores)
        ##print("argmax:",scores.argmax(dim=1))
        ##print("pred:", preds.numpy())
        ##print(f1)
        
x2 = range(0, len(accuracy_list) )
##print(x2)
y2= accuracy_list
plt.subplot(2,2,2)
plt.plot(x2, y2, "o-")
#plt.show()
plt.subplot(2,2,3)
plt.hist(accuracy_list, bins=100, density=True, alpha=0.5, color='b')
plt.xlabel('x')
plt.ylabel('Frequency')
plt.title('Histogram')

plt.savefig("test_1000GeV.png")
plt.show()

#print("ntrainLoader: ", len(trainLoader))
#print("ntestLoader: ", len(testLoader))


#for event_number in range(len( dataset )):
    #graph, label = dataset[ event_number ]
    #pred = model( graph, graph.ndata['feat'].float () )
    #num_correct += ( pred.argmax(1) == labels ).sum().item()
    #num_correct += ( pred.argmax(1) == batched_graph.ndata['labels'] ).sum().item() ##mx
    #num_tests += len( labels )
#?where is the graph?

##print ('Test accuracy:', num_correct / num_tests )

