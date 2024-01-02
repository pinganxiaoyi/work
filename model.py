import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义一个图神经网络模型
class ParticleGraphNetwork(nn.Module):
    def __init__(self, in_feats, hidden_feats = 64, num_classes = 2):
        super(ParticleGraphNetwork, self).__init__()
        self.input_module = nn.Linear(in_feats, hidden_feats)
        self.edge_module = nn.Linear(hidden_feats * 2, 1) #? 边权重是一个标量
        self.graph_module = GraphModule(hidden_feats, num_classes)

    def forward(self, g):
        # 输入模块：初始化隐藏表示
        features = g.ndata['feat']
        h = self.input_module(features)
        g.ndata['h'] = h

        # 边模块：更新边权重
        with g.local_scope():
            g.apply_edges(self.edge_function)
            edge_weights = g.edata['w']

        # 图模块：传播信息
        return self.graph_module(g, h, edge_weights)

    def edge_function(self, edges):
        # 边函数：计算边权重
        h_u = edges.src['h']
        h_v = edges.dst['h']
        edge_h = torch.cat([h_u, h_v], dim=1)
        return {'w': torch.sigmoid(self.edge_module(edge_h))}

class GraphModule(nn.Module):
    def __init__(self, in_feats, num_classes):
        super(GraphModule, self).__init__()
        self.layer = nn.Linear(in_feats, num_classes)

    def forward(self, g, h, edge_weights):
        with g.local_scope():
            g.ndata['h'] = h
            g.edata['w'] = edge_weights
            g.update_all(self.message_func, self.reduce_func)
            h = g.ndata['h']
            return self.layer(h)

    def message_func(self, edges):
        return {'m': edges.src['h'] * edges.data['w']}

    def reduce_func(self, nodes):
        return {'h': torch.sum(nodes.mailbox['m'], dim=1)}

# 假设输入特征数量、隐藏特征数量和类别数量
in_feats = ... # 输入特征的维度
hidden_feats = ... # 隐藏层的维度
num_classes = ... # 输出类别的数量

# 创建模型实例
model = ParticleGraphNetwork(in_feats, hidden_feats, num_classes)
