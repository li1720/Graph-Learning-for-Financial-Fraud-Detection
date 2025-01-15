import pandas as pd

# 假设特征表和边关系表的文件路径
feature_table_path = "/Users/liyiman/coding/NodeFormer/data/geom-gcn/film/out1_node_feature_label.txt"
edges_table_path = "/Users/liyiman/coding/NodeFormer/data/geom-gcn/film/out1_graph_edges.txt"

# 读取文件
feature_table = pd.read_csv(feature_table_path, sep="\t")
edges_table = pd.read_csv(edges_table_path, sep="\t")


# 筛选标签为1的全部保留
label_1_data = feature_table[feature_table["是否造假"] == 1]


# 随机选择和标签为1数量相同的标签为0的数据
label_0_data = feature_table[feature_table["是否造假"] == 0].sample(n=len(label_1_data), random_state=42)
# label_0_data = feature_table[feature_table["是否造假"] == 0].iloc[:40000].sample(n=len(label_1_data), random_state=42)

# 合并生成平衡数据集
balanced_feature_table = pd.concat([label_1_data, label_0_data]).sort_index()

# 筛选边关系表，确保两列都存在于处理后的特征表的编号中
valid_ids = set(balanced_feature_table["编号"])
filtered_edges_table = edges_table[
    (edges_table["编号"].isin(valid_ids)) & (edges_table["供应商编号"].isin(valid_ids))
]

# 保存结果
balanced_feature_table_path = "/Users/liyiman/coding/NodeFormer/data/geom-gcn/film/out1_node_feature_label_balance_non.txt"
filtered_edges_table_path = "/Users/liyiman/coding/NodeFormer/data/geom-gcn/film/out1_graph_edges_balance_non.txt"

balanced_feature_table.to_csv(balanced_feature_table_path, sep="\t", index=False, header=True)
filtered_edges_table.to_csv(filtered_edges_table_path, sep="\t", index=False, header=True)