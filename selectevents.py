import numpy as np
import awkward as ak
import ROOT
import pandas as pd
import torch
import dgl
import dgl.function as fn
import dgl.data as DGLDataset
from pathlib import Path

#在events层面挑选事例
#gn的方向限制
class select_event(DGLDataset):
    def __init__(self):
        super().__init__(name='CustomDataset')
    def process(self):
        file_path = Path("F:\\work\\top10cmx10cmVertical_gamma_0.5GeV.root")
        events = ROOT.TChain("events")
        events.Add(str(file_path))
#---------------------------------------------------------------------------------#
        #select mcparts to get gamma convert to electron and positron
        b_mcparts_pdgID = events['mcparts/mcparts.pdgID'].array(library="np")
        b_mcparts_trackID = events['mcparts/mcparts.trackID'].array(library="np")
#---------------------------------------------------------------------------------#

#----#

        b_fitCellCode_np = events['fithits/fithits.cellCode'].array(library="np")
        b_fitTrackID_np = events['fithits/fithits.trackID'].array(library="np")
        num_of_events = b_fitCellCode_np.size

        fitCellCode_ak = []

        label_ak = [] 
        for jentry in range(num_of_events):
#for jentry in range(1):
            b_fitCellCode_np_1d = ((b_fitCellCode_np[jentry]%10000)/100).astype(int)
            b_mcparts_pdgID_np_1d = b_mcparts_pdgID[jentry]
            b_mcparts_trackID_np_1d = b_mcparts_trackID[jentry]
            print("b_mcparts_pdgID_np_1d", b_mcparts_pdgID_np_1d)
            print("b_mcparts_trackID_np_1d", b_mcparts_trackID_np_1d)


            b_label_np_1d = np.where(b_label_np_1d==3, 0, 1) #git, trouble shooting issue #2 

            fitCellCode_ak.append(b_fitCellCode_np_1d)
            label_ak.append(b_label_np_1d)
        fitCellCode_ak = ak.from_iter(fitCellCode_ak)
        label_ak = ak.from_iter(label_ak)

        fithits0 = events.arrays(["fithits/fithits.cellCode", "fithits/fithits.pos.x", "fithits/fithits.pos.y", "fithits/fithits.pos.z", "fithits/fithits.edep", "fithits/fithits.trackID"], library="ak")
#fithits0 = ak.to_dataframe(fithits0, anonymous = 'cellCode', 'posX', 'posY', 'posZ', 'edep', 'trackID')
        fithits0 = ak.to_dataframe(fithits0)

        gn_trackID = [1,2] # geantino trackID 
        #track_layer= [0,1,2,3,4,5,6,7,8,9,10,11,12,13]
        track_layer= [0,1,3,5,7,9, 11,13] # keep (x,y) layers into 3d point/layer
        #track_layer= [2,4,6,8,10,12]

        fithits1 = fithits0.drop(columns='fithits/fithits.cellCode')
        fitLayerID_df = ak.to_dataframe(fitCellCode_ak, anonymous='layerID')
        fitTrackID_df = ak.to_dataframe(label_ak, anonymous='label')
### concat fithits1[posXYZ, edep] with layerID(for layer pair grouping), trackID( for node label)
        fithits = pd.concat([fithits1, fitLayerID_df, fitTrackID_df], axis=1)
        #print(fithits)

       # print("==========pre-processing of the data frame done==========")
       # print("move into event loop-------->")
        layer_pairs = [(2,4), (4,6), (6,8), (8,10), (10,12),(2,6),(4,8),(8,12),(2,8),(4,10),(6,12)]
        #layer_pairs = [(2,4), (4,6), (6,8), (8,10), (10,12)]
        #layer_pairs = [(1,3),(3,5),(5,7),(7,9),(9,11),(11,13)]
#layer_pairs = [(6,8)]

## entry at event level, sub entry at number of hits in one specific event
        sub_entry = 0
        self.graphs = []
        self.labels = []

dataset = select_event()