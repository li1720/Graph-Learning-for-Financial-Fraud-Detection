import pandas as pd

# # 去掉边关系
# input_file = "/Users/liyiman/coding/NodeFormer/data/fraud/特征表+供应商关系+客户关系+上市公司.xlsx"
# df = pd.read_excel(input_file)

# columns_to_drop = df.loc[:, "供应商金额1":df.columns[-2]].columns
# df_dropped = df.drop(columns=columns_to_drop)

# df_cleaned = df_dropped[~df_dropped.iloc[:,1].isnull()]

# df_cleaned.to_excel("/Users/liyiman/coding/NodeFormer/data/fraud/contrast_特征表.xlsx",index=False)


# 处理边关系

# 映射原始编号到新的连续编号
def remap_node_ids(edge_file, node_file, UPDATED_EDGE_FILE, UPDATED_NODE_FILE):
    # 加载节点数据，生成映射表
    nodes_df = pd.read_csv(node_file, sep='\t', header=0, names=["编号", "特征列", "是否造假"])
    unique_node_ids = nodes_df['编号'].unique()
    id_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_node_ids)}

    # 更新节点文件
    nodes_df['编号'] = nodes_df['编号'].map(id_mapping)
    nodes_df.to_csv(UPDATED_NODE_FILE, sep='\t', header=True, index=False)

    # 更新边文件
    edges_df = pd.read_csv(edge_file, sep='\t', header=0, names=['编号', '供应商编号'])
    edges_df['编号'] = edges_df['编号'].map(id_mapping)
    edges_df['供应商编号'] = edges_df['供应商编号'].map(id_mapping)
    edges_df.to_csv(UPDATED_EDGE_FILE, sep='\t', header=True, index=False)

    return UPDATED_EDGE_FILE, UPDATED_NODE_FILE
