
import numpy as np
import awkward as ak
import uproot
import pandas as pd
#import vector
from ROOT import TVector3
import torch
import dgl
from dgl.data import DGLDataset
from pathlib import Path
import ROOT
from dgl.dataloading import GraphDataLoader
#import matplotlib.pyplot as plt
#import networkx as nx

#class TVector3():
#    def __init__ ( self, df):
#        self._px = df[0]
#        self._py = df[1]
#        self._pz = df[2]
#        vec1=vector.obj(x=df[0], y=df[1], z=df[2])
#    def distance ( self , vec2 ):
#        self.rvalue = self.vec1.deltaR(vec2)
class Make_Graphs(DGLDataset):

    def __init__(self,file_path,data_part = "all"):
        self.data_part = data_part
        self.file_path = file_path
        super().__init__( name ='CustomDataset')
        
        self.count=0

    def process(self):
        #file = uproot.open("/herdfs/data/wangjunjing/top10cmx10cmVertical_mu-_100GeV_vertical_0307.root")
        #file = uproot.open("/herdfs/data/wangjunjing/top10cmx10cmVertical_gamma_0.5GeV.root")
        #file = uproot.open("/herdfs/data/tangzc/testMC/Test2/1test/v2022a-test-2/helium/test-helium-vertical-full-10GeV_n-simdigi.root")
        #file_path = Path("F:\\work\\top10cmx10cmVertical_gamma_0.5GeV.root")
        #file_path = Path("F:\\work\\top10cmx10cmVertical_gamma_0.1GeV.root")
        #file_path = Path("F:\\work\\test-proton-vertical-full-1000GeV_n-simdigi.root")
        #file_path = Path("F:\\work\\test-proton-vertical-full-100GeV_n-simdigi.root")
        #file_path = Path("F:\\work\\top10cmx10cmVertical_mu-_100GeV_vertical_0307.root")
        #file_path = Path("F:\\work\\test-helium-vertical-full-10GeV_n-simdigi.root")
        file = uproot.open(self.file_path)
        #file = uproot.open("/herdfs/data/tangzc/testMC/Test2/1test/v2022a-test-2/proton/test-proton-vertical-full-1000GeV_n-————.root")
        print("uproot file is ",self.file_path)
        events = file["events"]
# 在 process 函数中
        b_fitCellCode_np = events['fithits/fithits.cellCode'].array(library="np")
        b_fitTrackID_np = events['fithits/fithits.trackID'].array(library="np")
        num_of_events = b_fitCellCode_np.size
        fitCellCode_ak = []
        print("一共有",num_of_events)
        label_ak = [] #label from trackID, (trackID=3, label=0), (trackID!=3, label=1)
###from data_part to process different part as train val test
        start_index = 0
        end_index = num_of_events
        if self.data_part == "val":
            start_index,end_index = self.calculate_val_indices(num_of_events)
### convert cell code to layer code into     array
        for jentry in range(num_of_events):
            
        #for jentry in range(100):
#for jentry in range(1):
            b_fitCellCode_np_1d = ((b_fitCellCode_np[jentry]%10000)/100).astype(int)
            b_label_np_1d = b_fitTrackID_np[jentry]
            #??b_label_np_1d = np.where(b_label_np_1d==3, 0, 1) #git, trouble shooting issue #2
            b_label_np_1d = np.where(b_label_np_1d == 3, 0, 1)
            fitCellCode_ak.append(b_fitCellCode_np_1d)
            label_ak.append(b_label_np_1d)
            
### awkward array to pandas data frame

        fitCellCode_ak = ak.from_iter(fitCellCode_ak)
        label_ak = ak.from_iter(label_ak)

### input tree --> awkward --> pandas data frame
        fithits0 = events.arrays(["fithits/fithits.cellCode", "fithits/fithits.pos.x", "fithits/fithits.pos.y", "fithits/fithits.pos.z", "fithits/fithits.edep", "fithits/fithits.trackID"], library="ak")
        

#fithits0 = ak.to_dataframe(fithits0, anonymous = 'cellCode', 'posX', 'posY', 'posZ', 'edep', 'trackID')
        fithits0 = ak.to_dataframe(fithits0)

        gn_trackID = [1,2] # geantino trackID 
        #track_layer= [0,1,2,3,4,5,6,7,8,9,10,11,12,13]
        #track_layer= [0,1,3,5,7,9,11,13] # keep (x,y) layers into 3d point/layer
        #track_layer= [2,4,6,8,10,12]

        fithits1 = fithits0.drop(columns='fithits/fithits.cellCode')


### awkward array to pandas data frame
        fitLayerID_df = ak.to_dataframe(fitCellCode_ak, anonymous='layerID')
        fitTrackID_df = ak.to_dataframe(label_ak, anonymous='label')
### concat fithits1[posXYZ, edep] with layerID(for layer pair grouping), trackID( for node label)
        fithits = pd.concat([fithits1, fitLayerID_df, fitTrackID_df], axis=1)
        
        #=-print(fithits)
        for i in range(1010,1011):
            print("Event", i)
            print("X Position:", fithits0["fithits/fithits.pos.x"][i])
            
       # print("==========pre-processing of the data frame done==========")
       # print("move into event loop-------->")
        layer_pairs = [(2,4), (4,6), (6,8), (8,10), (10,12),(1,3),(3,5),(5,7),(7,9),(9,11),(11,13),(0,2)]
#layer_pairs = [(6,8)]

## entry at event level, sub entry at number of hits in one specific event
        sub_entry = 0
        self.graphs = []
        self.labels = []
#for jentry in range(num_of_events): ### loop over all events
        #for jentry in range(num_of_events):
        #for jentry in range(start_index,end_index):
        for jentry in range(1010,1011):
        #for jentry in range(100):
        #for jentry in range(95,96):
        #for jentry in range(1000):
            print("processing evt:", jentry)

            fithits_per_event = b_fitCellCode_np[jentry].size #sub entry, hits per event
            #print("sub hits per events:", fithits_per_event)
            if(fithits_per_event < 1) :
                continue

            # locate rows in corresponding to the sub entries by [0 : sub_entry] 
            # and all the columns of hits pos, trackID, layerID   
            # return hits0 as pandas data frame  
            
            hits0 = fithits.iloc[ sub_entry:sub_entry + fithits_per_event, 0:7]
            hits0 = hits0[hits0[ 'fithits/fithits.trackID' ].isin(gn_trackID) == False ] # remove geantino entries from pandas dataframe
            #hits0 = hits0[hits0[ 'layerID' ].isin(track_layer) == False ] # remove geantino entries from pandas dataframe
            #print(hits0)
            for i in range(len(hits0)):
                print("subEvent", i)
                print("X Position:", hits0.iloc[i]["fithits/fithits.pos.x"])

            # add hit index[0-n] for graph construction
            hitIndex = hits0['label'].copy(deep=True)
            for nhits in range(len(hits0.axes[0])):
                hitIndex.iloc[ nhits ] = nhits
            hitIndex = hitIndex.rename('hitIndex')
            hits0 = pd.concat([hits0, hitIndex], axis=1)
            for i in range(len(hits0)):
                print("subEvent", i)
                print("X Position:", hits0.iloc[i]["fithits/fithits.pos.x"])
            if(len(hits0.axes[0])) < 1:
                continue #TODO, overlapping

            # group the sub entry by layerID   
            group = hits0.groupby('layerID')

            #print("==========fit hits information:==========\n", hits0)

            #graph edge variables
            src_list = []
            dst_list = []

            R_list = []
            deltaX_list = []
            deltaY_list = []
            deltaZ_list = []
            deltaE_list = []

            theta_list = []
            phi_list = []
            src_layer = []
            dst_layer = []

            elable_list = []
            #hit_list = []
            #node_labels = []

            #hit_list = hits0.loc[:,'fithits/fithits.pos.x', 'fithits/fithits.pos.y', 'fithits/fithits.pos.z', 'fithits/fithits.edep'] #hit_list --> node features(posXYZ, edep)
            hit_list = hits0.iloc[:, 0:4] #hit_list --> node features(posXYZ, edep)

            node_labels = hits0['label']
            num_of_node = len(hits0.axes[0])

            #print("---> number of valid hits:\n", num_of_node)

            for (layer1, layer2) in layer_pairs:
                try:
                    hits1=group.get_group(layer1)
                    #print("hits in layer1:\n", hits1)
                    hits2=group.get_group(layer2)
                    #print("hits in layer2:\n", hits2)
                    #print(pd.concat([hits1, hits2],axis=0))
                    #combined_hits = pd.concat([hits1, hits2],axis=0)
                    nhits1 = len(hits1.axes[0])
                    nhits2 = len(hits2.axes[0])
                    #hit_list=combined_hits[['fithits/fithits.pos.x', 'fithits/fithits.pos.y', 'fithits/fithits.pos.z', 'fithits/fithits.edep']] #hit_list --> node features(posXYZ, edep)
                    #node_labels = combined_hits['label']

                    ### to build edge information
                    for m in range (nhits1):
                        for n in range (nhits2):
                            src_list.append(hits1['hitIndex'].values[m])
                            vec1 = TVector3()
                            vec1.SetXYZ( hits1['fithits/fithits.pos.x'].values[m],
                                         hits1['fithits/fithits.pos.y'].values[m],
                                         hits1['fithits/fithits.pos.z'].values[m] )

                            dst_list.append(hits2['hitIndex'].values[n])
                            vec2 = TVector3()
                            vec2.SetXYZ( hits2['fithits/fithits.pos.x'].values[n],
                                         hits2['fithits/fithits.pos.y'].values[n],
                                         hits2['fithits/fithits.pos.z'].values[n] )

                            R_list.append((vec2-vec1).Mag())
                            deltaX_list.append(vec2.X() - vec1.X())
                            deltaY_list.append(vec2.Y() - vec1.Y())
                            deltaZ_list.append(vec2.Z() - vec1.Z())
                            deltaE_list.append(hits2['fithits/fithits.edep'].values[n] - hits1['fithits/fithits.edep'].values[m])
                            theta = vec1.Angle(vec2)
                            theta_list.append(theta)
                            phi_list.append(theta * 180.0 / ROOT.TMath.Pi())
                            src_layer.append(layer1)
                            dst_layer.append(layer2)

                            if(hits1['label'].values[m]==0 and hits2['label'].values[n]==0):
                                elable_list.append(0)
                            if(hits1['label'].values[m]==1 and hits2['label'].values[n]==0):
                                elable_list.append(1)
                            if(hits1['label'].values[m]==0 and hits2['label'].values[n]==1):
                                elable_list.append(1)
                            if(hits1['label'].values[m]==1 and hits2['label'].values[n]==1):
                                elable_list.append(1)
                            #else:
                            #    elable_list.append(1)

                except KeyError as e:
                    continue
            sub_entry = sub_entry + fithits_per_event

            ##TODO, transformed torch or not ? in Graph
            hit_features = torch.from_numpy(np.array(hit_list)).T
            hit_features2 = torch.from_numpy(np.array(hit_list))
            for i in range(len(hits0)):
                print("subEvent", i)
                print("X Position:", hits0.iloc[i]["fithits/fithits.pos.x"])

            edges_src = torch.from_numpy(np.array(src_list))
            edges_dst = torch.from_numpy(np.array(dst_list))
            edges_features = torch.from_numpy( np.array(R_list) )
            edges_features = torch.unsqueeze(edges_features, 1)

            deltaX = torch.from_numpy(np.array(deltaX_list))
            deltaX = torch.unsqueeze(deltaX, 1)

            deltaY = torch.from_numpy(np.array(deltaY_list))
            deltaY = torch.unsqueeze(deltaY, 1)
            
            deltaZ = torch.from_numpy(np.array(deltaZ_list))
            deltaZ = torch.unsqueeze(deltaZ, 1)

            deltaE = torch.from_numpy(np.array(deltaE_list))
            deltaE = torch.unsqueeze(deltaE, 1)

            theta = torch.from_numpy(np.array(theta_list))
            theta = torch.unsqueeze(theta,1)

            phi = torch.from_numpy(np.array(phi_list))
            phi = torch.unsqueeze(phi,1)

            

            node_labels = torch.from_numpy(np.array(node_labels))
            edge_labels = torch.from_numpy(np.array(elable_list))
            #print('hit feature tensor2', hit_features2, hit_features2.size(dim=0))
            #print(hit_features2[edges_src])
            #print('node label tensor', node_labels, node_labels.size(dim=0))

            #print(f"--->edeg src tensor:{edges_src}, size:{edges_src.size(dim=0)}")
            #print(f"--->edeg dst tensor:{edges_dst}, size:{edges_dst.size(dim=0)}")
            #print(f"--->edeg feature tensor:{edges_features}, size:{edges_features.size(dim=0)}")

            #print(f"hit feature tensor:{hit_features}, size:{hit_features.size(dim=0)}")
            #print(f"hit feature tensor2:{hit_features2}, size:{hit_features2.size(dim=0)}")
            #print(f"node label tensor:{node_labels}, size{node_labels.size(dim=0)}")
            #print('graph details--->')

            if( edges_src.size(dim=0) < 1 ) : ##dgl._ffi.base.DGLError: [11:06:33] /opt/dgl/src/graph/unit_graph.cc:69: Check failed: aten::IsValidIdArray(src):
                continue
            gr = dgl.graph(( edges_src , edges_dst ) , num_nodes = node_labels.size( dim = 0 ) )
            gr.ndata['feat'] = hit_features2
            gr.ndata['label'] = node_labels
            gr.edata['weight'] = edges_features
            gr.edata['dx'] = deltaX
            gr.edata['dy'] = deltaY
            gr.edata['dz'] = deltaZ
            gr.edata['dE'] = deltaE
            gr.edata['theta'] = theta
            gr.edata['phi'] = phi 
            #gr.edata['src_layer'] = torch.from_numpy(np.array(src_layer))
            #gr.edata['dst_layer'] = torch.from_numpy(np.array(dst_layer))


            gr.edata['label'] = edge_labels

            #gr.ndata['u'] = hit_features2[edges_src]
            #gr.ndata['v'] = hit_features2[edges_dst]

            n_nodes = node_labels.size( dim =0)
            n_train = int ( n_nodes * 0.6)
            n_val = int ( n_nodes * 0.2)
            train_mask = torch.zeros ( n_nodes, dtype = torch.bool )
            val_mask = torch.zeros ( n_nodes, dtype = torch.bool )
            test_mask = torch.zeros ( n_nodes, dtype = torch.bool )
            train_mask[:n_train] = True
            val_mask[ n_train : n_train + n_val ] = True
            test_mask[ n_train + n_val :] = True

            gr.ndata['train_mask'] = train_mask
            gr.ndata['val_mask'] = val_mask
            gr.ndata['test_mask'] = test_mask

            n_edges = edge_labels.size( dim =0)
            n_train = int ( n_edges * 0.6)
            n_val = int ( n_edges * 0.2)
            train_edge_mask = torch.zeros ( n_edges, dtype = torch.bool )
            val_edge_mask = torch.zeros ( n_edges, dtype = torch.bool )
            test_edge_mask = torch.zeros ( n_edges, dtype = torch.bool )
            train_edge_mask[:n_train] = True
            val_edge_mask[ n_train : n_train + n_val ] = True
            test_edge_mask[ n_train + n_val :] = True

            gr.edata['train_mask'] = train_edge_mask
            gr.edata['val_mask'] = val_edge_mask
            gr.edata['test_mask'] = test_edge_mask

            gr = dgl.add_self_loop(gr)
            self.graphs.append(gr)
            #label=1
            #self.labels.append(label)
            self.labels.append(jentry)
            print("labels",jentry)
            '''
            print(gr)
            print('Node features')
            print(gr.ndata['feat'])
            print('edge features-->weight\n')
            print(gr.edata['weight'])
            print('edge coo')
            print(gr.edges())

            print('Edge features')
            print(gr.edata)
            #gr = dgl.graph(( edges_src , edges_dst ) , num_nodes = 2 )
            gr = dgl.graph(( edges_src , edges_dst ))
            '''
            #print('next event----', jentry)
    def calculate_val_indices(self, num_of_events):
        # 根据您的需求计算验证集的起止索引
        # 例如，如果验证集是数据的 70% 到 90%
        start_index = int(num_of_events * 0.1)
        end_index = int(num_of_events * 0.12)
        return start_index, end_index
    def __getitem__(self, i):
        graph, label=self.graphs[i], self.labels[i]
        #print(f"Graph{i} node feature shape:{graph.ndata['feat'].shape}")
        return graph , label
    def __len__(self):
        return len(self.graphs)

dataset = Make_Graphs("F:\\work\\muon_10GeV_noCalo_comp_vertical-center-nY.g4mac.root",data_part="val")
loader = GraphDataLoader(dataset, batch_size=1, shuffle=False)
for num,(graphs, labels) in enumerate(loader): 
    if labels in range(1000,1020):
        print("labels",labels)
        print("graphs",graphs)
        print("graphs.ndata['feat']",graphs.ndata['feat'])
        continue

'''
print(dataset.count)
print(len(dataset))
#dgl.data.utils.save_info()
#dgl.save_graphs('test.bin',g,l)
count = 0
for event_number in range(len( dataset )):
    graph, label = dataset[event_number]
    count+=1
print("events/graph count ",count)
#    print(label , graph )
#    print( graph.ndata['feat'].shape )
#    print( graph.edata )
graph,label = dataset[5]
print("图是",graph)
print("labels有",label)
#dataset.process()  # 处理数据以构建图

# 选择要可视化的图（例如第一个图）
#graph, label = dataset[0]

# 将 DGL 图转换为 NetworkX 图
#nx_graph = graph.to_networkx().to_undirected()

# 绘制图形
#plt.figure(figsize=(12, 12))
#?nx.draw(nx_graph, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10)
#plt.title(f"Event {label}")
#plt.show()
    '''