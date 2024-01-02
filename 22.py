import uproot
import awkward as ak
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from make_graph import Make_Graphs
from dgl.dataloading import GraphDataLoader
"""
selected_data = {
    "pos_x": [],
    "pos_y": [],
    "pos_z": []
}
num = 0
# 打开ROOT文件
with uproot.open("F:\\work\\muon_10GeV_noCalo_comp_vertical-center.g4mac.root") as file:
    # 获取TTree
    tree = file["events"]
    # 遍历所有entry

    entry_total = tree.num_entries
    print(entry_total)
    for entry in range(20):
        # 读取数据
        data = tree.arrays(
            ["fithits/fithits.pos.x", "fithits/fithits.pos.y", "fithits/fithits.pos.z", "fithits/fithits.trackID"],
            entry_start=entry,
            entry_stop=entry + 1,
            library="ak"
        )
        b_fitCellCode_np = tree['fithits/fithits.cellCode'].array(library="np")

        mask = data["fithits/fithits.trackID"] == 3
        if len(data["fithits/fithits.pos.x"][mask]) == 0:
            print("empty", entry)
            num += 1
            continue
        selected_data["pos_x"].append(data["fithits/fithits.pos.x"][mask])
        #selected_data["pos_y"].append(data["fithits/fithits.pos.y"][mask])
        selected_data["pos_z"].append(data["fithits/fithits.pos.z"][mask])
    print("num",num)    
# 绘图
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot()

# 绘制3D散点图
for entry in (1,11):
    x = ak.to_numpy(selected_data["pos_x"][entry])
    #y = ak.to_numpy(selected_data["pos_y"][entry])
    z = ak.to_numpy(selected_data["pos_z"][entry])  
    if x.size == 0:
        num+=1
        continue
    ax.scatter(x, z, label=f"Entry {entry}")
print("num",num)
# 设置图表标题和轴标签
ax.set_title('3D Position for trackID=3')
ax.set_xlabel('X')
ax.set_ylabel('Y')
#ax.set_zlabel('Z')


# 显示图例
#ax.legend()

# 显示图表
plt.show()
"""
dataset = Make_Graphs("F:\\work\\muon_10GeV_noCalo_comp_vertical-center.g4mac.root", data_part="val")
loader = GraphDataLoader(dataset, batch_size=1, shuffle=True)

num = 0
for g, l in loader:
    num += 1
    if num > 2:
        break
    
    src_nodes, dst_nodes = g.edges()
    src_features = []
    dst_features = []
    
    for m, n in zip(src_nodes.numpy(), dst_nodes.numpy()):
        src_feat = g.ndata['feat'][m]
        dst_feat = g.ndata['feat'][n]
        src_features.append(src_feat.numpy())  # Assuming you want to convert to numpy array
        dst_features.append(dst_feat.numpy())

    print(src_features, dst_features)

