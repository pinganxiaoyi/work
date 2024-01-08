import matplotlib
matplotlib.use('Agg')
import dgl
import torch
import torch.nn as nn
import dgl.nn as dglnn
import ROOT
#//from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from make_graph import Make_Graphs
from ROOT import TCanvas, TGraph2D 
import math
from make_graph import Make_Graphs
from dgl.dataloading import GraphDataLoader
from ROOT import TCanvas, TGraph2D

ROOT.gROOT.SetBatch(True)
def fit_angle(graph,labels):

    #for index, (graph, label) in enumerate(dataset):  
       #if  index >= 10:  # 如果处理了 20 个图，则停止
       #    break
       #if label not in [157,98,29,57,896,817,904,430]:
        #   continue 
       #获取所有边的label 选出边的label为0的边 画图
    edge_label = graph.edata['label']
    src_nodes = graph.edges()[0]
    dst_nodes = graph.edges()[1]
    err_label = []
    #print("labels:", labels, "edge_label:", edge_label)
    if not any(edge_label == 0):
        err_label.append(labels)
        return None, None, None, None, err_label     

    c1 = ROOT.TCanvas(f"c{labels}", f"Graph {labels}", 800, 1800)
    c1.Divide(1,3)
    g_xz = ROOT.TGraph()
    g_yz = ROOT.TGraph()
    g_xy = ROOT.TGraph()
    edep = []
    xz_idx = 0
    yz_idx = 0  # 点的索引
    xy_idx = 0
    tot_edep = 0
    #//points = []
    #//print("layerID", dataset.fitLayerID_df.index)
    for i in range(len(edge_label)):      
        if edge_label[i] == 0:
            src_idx = src_nodes[i]
            dst_idx = dst_nodes[i]

            src_feat = graph.ndata['feat'][src_idx][:3]
            dst_feat = graph.ndata['feat'][dst_idx][:3]
            #//points.append(src_feat)
            #//points.append(dst_feat)
            #edep.append(graph.edata['dE'][i])
            edep.append(graph.ndata['feat'][src_idx][3]+graph.ndata['feat'][dst_idx][3])
            #print(f"src Node {src_idx} is in layer:")
            #print(f"dst Node {dst_idx} is in layer:", graph.edata['dst_layer'][i].item())    
            # 为 TGraph 对象设置点
            g_xz.SetPoint(xz_idx, src_feat[0], src_feat[2])
            g_yz.SetPoint(yz_idx, src_feat[1], src_feat[2])
            g_xy.SetPoint(xy_idx, src_feat[0], src_feat[1])
            xz_idx += 1
            yz_idx += 1
            xy_idx += 1
            g_xz.SetPoint(xz_idx, dst_feat[0], dst_feat[2])
            g_yz.SetPoint(yz_idx, dst_feat[1], dst_feat[2])
            g_xy.SetPoint(xy_idx, dst_feat[0], dst_feat[1])
            xz_idx += 1
            yz_idx += 1
            xy_idx += 1
            
            # 为 TPolyLine3D 对象添加线段
            #line.SetNextPoint(src_feat[0], src_feat[1], src_feat[2])
            #line.SetNextPoint(dst_feat[0], dst_feat[1], dst_feat[2])
    tot_edep = sum(edep)   
    ROOT.gStyle.SetOptFit(1111)

    c1.cd(1)
    g_xz.SetMarkerStyle(20)
    g_xz.SetMarkerSize(1.0)
    g_xz.Draw("AP")
    
    # 线性拟合
    result_xz = g_xz.Fit("pol1","SQ")
    k1 = result_xz.Parameter(1)
    b1 = result_xz.Parameter(0)
    theta_xz = math.atan(k1)
    theta_xz = math.degrees(theta_xz)  # 夹角（度）
    pt_xz = ROOT.TPaveText(0.1, 0.7, 0.3, 0.8, "blNDC")
    pt_xz.AddText(f"XZ plane angle: {theta_xz:.2f} degrees")
    pt_xz.SetFillColor(0)
    pt_xz.Draw()
       
    c1.cd(2)
    g_yz.SetMarkerStyle(20)
    g_yz.SetMarkerSize(1.0)
    g_yz.Draw("AP")

    result_yz = g_yz.Fit("pol1","SQ")
    b2 = result_yz.Parameter(0)
    k2 = result_yz.Parameter(1)
    theta_yz = math.atan(k2)
    theta_yz = math.degrees(theta_yz)  # 夹角（度）
    
    pt_yz = ROOT.TPaveText(0.1, 0.6, 0.3, 0.7, "blNDC")
    pt_yz.AddText(f"YZ plane angle: {theta_yz:.2f} degrees")
    pt_yz.SetFillColor(0)
    pt_yz.Draw()
    #print("xz面夹角（度）:", theta_xz)
    #print("yz面夹角（度）:", theta_yz)
       
    c1.cd(3)
    g_xy.SetMarkerStyle(20)
    g_xy.SetMarkerSize(1.0)
    g_xy.Draw("AP")

    result_xy = g_xy.Fit("pol1","SQ")
    b3 = result_xy.Parameter(0)
    k3 = result_xy.Parameter(1)
    theta_xy = math.atan(k3)
    theta_xy = math.degrees(theta_xy)  # 夹角（度）
    pt_xy = ROOT.TPaveText(0.1, 0.6, 0.3, 0.7, "blNDC")
    
    theta0_xy = math.atan(math.sqrt(k1**2 + k2**2))
    theta0_xy = math.degrees(theta0_xy)
    theta0_yz = math.atan(math.sqrt(k2**2 + k3**2))
    theta0_yz = math.degrees(theta0_yz)
    theta0_xz = math.atan(math.sqrt(k1**2 + k3**2))
    theta0_xz = math.degrees(theta0_xz)
    #print("投影xy面夹角（度）:", theta_xy)
    #print("与xy面夹角",theta0_xy)
    pt_xy.AddText("XY plane angle: {:.2f} degrees".format(theta_xy))
    pt_xy.AddText("with XY plane: {:.2f} degrees".format(theta0_xy))
    pt_xy.SetFillColor(0)
    pt_xy.Draw()
    c1.SaveAs(r"C:\Users\10094\Desktop\gr\gr_{}.png".format(label))
    return theta0_xy,theta0_xz,theta0_yz,tot_edep,err_label
    

    # 保存画布
    

angle_xy = []  # 存储 XY 平面角度
angle_xz = []  # 存储 XZ 平面角度
angle_yz = []  # 存储 YZ 平面角度
edep = []      # 存储能量沉积
err_labels = []  # 存储错误标签
dataset = Make_Graphs("F:\\work\\muon_10GeV_noCalo_comp_vertical-center-nY.g4mac.root",data_part="val")
loader = GraphDataLoader(dataset, batch_size=1, shuffle=False)
for graph, label in loader: 
    if label in range(1000,1020):
        xy, xz, yz, tot_edep, err = fit_angle(graph, label)
        if err:
            err_labels.extend(err)
            continue
    angle_xy.append((label, xy))
    angle_xz.append((label, xz))
    angle_yz.append((label, yz))
    edep.append((label, tot_edep))

# 打印错误标签
print("Error labels0:", err_labels)
print("edep",edep)
print("angel_xy",angle_xy)

def selected_angle(graph,label, selected_edges):
    # 准备画布和图形
    c1 = ROOT.TCanvas("c1", "Graph", 800, 1800)
    c1.Divide(1, 3)
    g_xz = ROOT.TGraph()
    g_yz = ROOT.TGraph()
    g_xy = ROOT.TGraph()

    # 索引和能量累加
    xz_idx, yz_idx, xy_idx = 0, 0, 0
    tot_edep = 0
    edep = []
    # 遍历选中的边
    if not selected_edges:
        print("No selected edges to fit.")
        return None, None, None, None

    for (src_idx, dst_idx) in selected_edges:
        src_feat = graph.ndata['feat'][src_idx][:3].cpu().numpy()
        dst_feat = graph.ndata['feat'][dst_idx][:3].cpu().numpy()
        edep.append(graph.ndata['feat'][src_idx][3].cpu().numpy() + graph.ndata['feat'][dst_idx][3].cpu().numpy())

        # 设置 TGraph 对象的点
        g_xz.SetPoint(xz_idx, src_feat[0], src_feat[2])
        g_yz.SetPoint(yz_idx, src_feat[1], src_feat[2])
        g_xy.SetPoint(xy_idx, src_feat[0], src_feat[1])
        xz_idx += 1
        yz_idx += 1
        xy_idx += 1
        g_xz.SetPoint(xz_idx, dst_feat[0], dst_feat[2])
        g_yz.SetPoint(yz_idx, dst_feat[1], dst_feat[2])
        g_xy.SetPoint(xy_idx, dst_feat[0], dst_feat[1])
        xz_idx += 1
        yz_idx += 1
        xy_idx += 1

    # 总能量
    tot_edep = sum(edep)
    ROOT.gStyle.SetOptFit(1111)

    c1.cd(1)
    g_xz.SetMarkerStyle(20)
    g_xz.SetMarkerSize(1.0)
    g_xz.Draw("AP")
    
    # 线性拟合
    result_xz = g_xz.Fit("pol1","SQ")
    k1 = result_xz.Parameter(1)
    b1 = result_xz.Parameter(0)
    theta_xz = math.atan(k1)
    theta_xz = math.degrees(theta_xz)  # 夹角（度）
    pt_xz = ROOT.TPaveText(0.1, 0.7, 0.3, 0.8, "blNDC")
    pt_xz.AddText(f"XZ plane angle: {theta_xz:.2f} degrees")
    pt_xz.SetFillColor(0)
    pt_xz.Draw()
       
    c1.cd(2)
    g_yz.SetMarkerStyle(20)
    g_yz.SetMarkerSize(1.0)
    g_yz.Draw("AP")

    result_yz = g_yz.Fit("pol1","SQ")
    b2 = result_yz.Parameter(0)
    k2 = result_yz.Parameter(1)
    theta_yz = math.atan(k2)
    theta_yz = math.degrees(theta_yz)  # 夹角（度）
    
    pt_yz = ROOT.TPaveText(0.1, 0.6, 0.3, 0.7, "blNDC")
    pt_yz.AddText(f"YZ plane angle: {theta_yz:.2f} degrees")
    pt_yz.SetFillColor(0)
    pt_yz.Draw()
    #print("xz面夹角（度）:", theta_xz)
    #print("yz面夹角（度）:", theta_yz)
       
    c1.cd(3)
    g_xy.SetMarkerStyle(20)
    g_xy.SetMarkerSize(1.0)
    g_xy.Draw("AP")

    result_xy = g_xy.Fit("pol1","SQ")
    b3 = result_xy.Parameter(0)
    k3 = result_xy.Parameter(1)
    theta_xy = math.atan(k3)
    theta_xy = math.degrees(theta_xy)  # 夹角（度）
    pt_xy = ROOT.TPaveText(0.1, 0.6, 0.3, 0.7, "blNDC")
    
    theta0_xy = math.atan(math.sqrt(k1**2 + k2**2))
    theta0_xy = math.degrees(theta0_xy)
    theta0_yz = math.atan(math.sqrt(k2**2 + k3**2))
    theta0_yz = math.degrees(theta0_yz)
    theta0_xz = math.atan(math.sqrt(k1**2 + k3**2))
    theta0_xz = math.degrees(theta0_xz)
    #print("投影xy面夹角（度）:", theta_xy)
    #print("与xy面夹角",theta0_xy)
    pt_xy.AddText("XY plane angle: {:.2f} degrees".format(theta_xy))
    pt_xy.AddText("with XY plane: {:.2f} degrees".format(theta0_xy))
    pt_xy.SetFillColor(0)
    pt_xy.Draw()
    c1.SaveAs(f"selected_angle{label}.png")
    
    return theta0_xy, theta0_xz, theta0_yz, tot_edep
'''
dataset = Make_Graphs("F:\\work\\muon_10GeV_noCalo_comp_vertical-center-ny.g4mac.root",data_part="val")
angle_xy = []  # 存储 XY 平面角度
angle_xz = []  # 存储 XZ 平面角度
angle_yz = []  # 存储 YZ 平面角度
edep = []      # 存储能量沉积   
err = [] 
for num, (graph, labels) in enumerate(dataset):
    if num >= 20:  # 如果处理了 20 个图，则停止
        break
    edges = graph.edges()
    src,dst = edges[0],edges[1]
    edge = []
    for i in range(len(src)):
        edge.append((src[i],dst[i]))
    xy, xz, yz, tot_edep =selected_angle(graph,labels,edge)
    angle_xy.append((labels, xy))
    angle_xz.append((labels, xz))
    angle_yz.append((labels, yz))
    edep.append((labels, tot_edep))
    #err.append(err)
#print("err long is ",len(err))    
#print("edep",edep)
print("angel_xy",angle_xy)
'''