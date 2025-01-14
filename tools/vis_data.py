import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# 加载生成的Excel文件
input_file = "/Users/liyiman/coding/NodeFormer/data/geom-gcn/film/0-1feature.xlsx"
df = pd.read_excel(input_file)

"""特征"""

# # 删除编号列
# df = df.drop(columns=["编号"])

# # 将空值替换为0
# df = df.fillna(0)

# # 提取是否造假列（作为颜色区分）
# labels = df["是否造假"]

# # 提取特征列（去掉标签列）
# features = df.drop(columns=["是否造假"])

# # 使用tsen降维到2D
# tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
# reduced_features = tsne.fit_transform(features)

# # 将降维结果加入原数据框
# df["X"] = reduced_features[:, 0]
# df["Y"] = reduced_features[:, 1]

# # 可视化
# plt.figure(figsize=(8, 6))
# for label in df["是否造假"].unique():
#     plt.scatter(
#         df[df["是否造假"] == label]["X"],
#         df[df["是否造假"] == label]["Y"],
#         label=f"Label {label}",
#         alpha=0.7
#     )

# plt.title("2D Visualization of Features (PCA Reduced)")
# plt.xlabel("X-Axis")
# plt.ylabel("Y-Axis")

# # 设置坐标轴范围
# # plt.xlim(0, 0.5)
# # plt.ylim(-1, 5)

# # 添加图例和网格

# plt.legend()
# plt.grid(True)


# # 保存图像到指定路径
# output_path = "/Users/liyiman/coding/NodeFormer/figs/data1.jpg"
# plt.savefig(output_path, format="jpg", dpi=300)
# print(f"图像已保存到 {output_path}")

# # 显示图像
# plt.show()


"""0-1"""
# 拆分 feature_blank 列为多个特征列
df_features = df["Feature_Blank"].apply(lambda x: pd.Series(list(x))).astype(float)
# df_features = df_features.iloc[:,894]
# 添加标签列用于降维后可视化
labels = df["label"]

# 使用PCA降维到2D
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
reduced_features = tsne.fit_transform(df_features)
# pca = PCA(n_components=2)
# reduced_features = pca.fit_transform(df_features)

# 将降维结果加入原数据框
df["X"] = reduced_features[:, 0]
df["Y"] = reduced_features[:, 1]

# 可视化
plt.figure(figsize=(8, 6))
for label in df["label"].unique():
    plt.scatter(
        df[df["label"] == label]["X"],
        df[df["label"] == label]["Y"],
        label=f"Label {label}",
        alpha=0.7
    )

plt.title("2D Visualization of Features (PCA Reduced)")
plt.xlabel("X-Axis")
plt.ylabel("Y-Axis")
plt.legend()
plt.grid(True)

# 保存图像到指定路径
output_path = "/Users/liyiman/coding/NodeFormer/figs/data.jpg"
plt.savefig(output_path, format="jpg", dpi=300)
print(f"图像已保存到 {output_path}")

# 显示图像
plt.show()
