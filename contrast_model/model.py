import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, roc_curve
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
import numpy as np
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from data_make import remap_node_ids


input_file = "/Users/liyiman/coding/NodeFormer/data/fraud/contrast_特征表.xlsx"
df = pd.read_excel(input_file)
# # 文件路径[正负类平衡+不含未上市公司]
# EDGE_FILE = '/Users/liyiman/coding/NodeFormer/data/geom-gcn/film/out1_graph_edges_balance.txt'  # 边关系文件
# NODE_FILE = '/Users/liyiman/coding/NodeFormer/data/geom-gcn/film/out1_node_feature_label_balance.txt'  # 节点特征和标签文件
# UPDATED_EDGE_FILE = '/Users/liyiman/coding/NodeFormer/data/geom-gcn/film/out1_graph_edges_balance_index.txt'
# UPDATED_NODE_FILE = '/Users/liyiman/coding/NodeFormer/data/geom-gcn/film/out1_node_feature_label_balance_index.txt'

# # 文件路径[不平衡]
# EDGE_FILE = '/Users/liyiman/coding/NodeFormer/data/geom-gcn/film/out1_graph_edges.txt'  # 边关系文件
# NODE_FILE = '/Users/liyiman/coding/NodeFormer/data/geom-gcn/film/out1_node_feature_label.txt'  # 节点特征和标签文件

# 文件路径[正负类平衡+包含未上市公司]
EDGE_FILE = '/Users/liyiman/coding/NodeFormer/data/geom-gcn/film/out1_graph_edges_balance_non.txt'  # 边关系文件
NODE_FILE = '/Users/liyiman/coding/NodeFormer/data/geom-gcn/film/out1_node_feature_label_balance_non.txt'  # 节点特征和标签文件
UPDATED_EDGE_FILE = '/Users/liyiman/coding/NodeFormer/data/geom-gcn/film/out1_graph_edges_balance_non_index.txt'  # 边关系文件
UPDATED_NODE_FILE = '/Users/liyiman/coding/NodeFormer/data/geom-gcn/film/out1_node_feature_label_balance_non_index.txt'  # 节点特征和标签文件
"""机器学习"""

def logistic_regression_classification(df):
    """
    Perform logistic regression classification on the input DataFrame.

    Parameters:
    df (DataFrame): Input DataFrame where the first column is the ID, middle columns are features, 
                    and the last column is the label.

    Returns:
    None: Prints evaluation metrics and displays the ROC curve.
    """
    # Ensure features and labels are correctly selected
    X = df.iloc[:, 1:-1]
    y = df.iloc[:, -1]

    # Split into training and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Create logistic regression model
    model = LogisticRegression(max_iter=2000, random_state=42, n_jobs=-1)

    # Train the model
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Compute evaluation metrics
    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    recall = recall_score(y_test, y_pred)

    print(f"lr_Accuracy: {acc:.4f}")
    print(f"lr_ROC-AUC: {roc_auc:.4f}")
    print(f"lr_Recall: {recall:.4f}")


    # # Plot ROC curve
    # fpr, tpr, _ = roc_curve(y_test, y_prob)
    # plt.figure()
    # plt.plot(fpr, tpr, label=f"ROC Curve (area = {roc_auc:.4f})")
    # plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
    # plt.xlabel("False Positive Rate")
    # plt.ylabel("True Positive Rate")
    # plt.title("Receiver Operating Characteristic")
    # plt.legend(loc="best")
    # plt.show()

def random_forest_classification(df):
    """
    Perform random forest classification on the input DataFrame.

    Parameters:
    df (DataFrame): Input DataFrame where the first column is the ID, middle columns are features, 
                    and the last column is the label.

    Returns:
    None: Prints evaluation metrics and displays the ROC curve.
    """
    # Ensure features and labels are correctly selected
    X = df.iloc[:, 1:-1]
    y = df.iloc[:, -1]

    # Split into training and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Create random forest model
    model = RandomForestClassifier(n_estimators=500, max_depth=20, random_state=42, n_jobs=-1)

    # Train the model
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Compute evaluation metrics
    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    recall = recall_score(y_test, y_pred)

    print(f"rf_Accuracy: {acc:.4f}")
    print(f"rf_ROC-AUC: {roc_auc:.4f}")
    print(f"rf_Recall: {recall:.4f}")

    # # Plot ROC curve
    # fpr, tpr, _ = roc_curve(y_test, y_prob)
    # plt.figure()
    # plt.plot(fpr, tpr, label=f"ROC Curve (area = {roc_auc:.4f})")
    # plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
    # plt.xlabel("False Positive Rate")
    # plt.ylabel("True Positive Rate")
    # plt.title("Receiver Operating Characteristic")
    # plt.legend(loc="best")
    # plt.show()

def svm_classification(df):
    """
    Perform SVM classification on the input DataFrame.

    Parameters:
    df (DataFrame): Input DataFrame where the first column is the ID, middle columns are features, 
                    and the last column is the label.

    Returns:
    None: Prints evaluation metrics and displays the ROC curve.
    """
    # Ensure features and labels are correctly selected
    X = df.iloc[:, 1:-1]
    y = df.iloc[:, -1]

    # Split into training and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Create SVM model
    model = SVC(probability=True, kernel='rbf', C=1, gamma='scale', random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Compute evaluation metrics
    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    recall = recall_score(y_test, y_pred)

    print(f"svm_Accuracy: {acc:.4f}")
    print(f"svm_ROC-AUC: {roc_auc:.4f}")
    print(f"svm_Recall: {recall:.4f}")

    # # Plot ROC curve
    # fpr, tpr, _ = roc_curve(y_test, y_prob)
    # plt.figure()
    # plt.plot(fpr, tpr, label=f"ROC Curve (area = {roc_auc:.4f})")
    # plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
    # plt.xlabel("False Positive Rate")
    # plt.ylabel("True Positive Rate")
    # plt.title("Receiver Operating Characteristic")
    # plt.legend(loc="best")
    # plt.show()

def xgboost_classification(df):
    """
    Perform XGBoost classification on the input DataFrame.

    Parameters:
    df (DataFrame): Input DataFrame where the first column is the ID, middle columns are features, 
                    and the last column is the label.

    Returns:
    None: Prints evaluation metrics and displays the ROC curve.
    """
    # Ensure features and labels are correctly selected
    X = df.iloc[:, 1:-1]
    y = df.iloc[:, -1]

    # Split into training and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Create XGBoost model
    model = XGBClassifier(n_estimators=500, max_depth=10, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, random_state=42, use_label_encoder=False, eval_metric='logloss')

    # Train the model
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Compute evaluation metrics
    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    recall = recall_score(y_test, y_pred)

    print(f"xgb_Accuracy: {acc:.4f}")
    print(f"xgb_ROC-AUC: {roc_auc:.4f}")
    print(f"xgb_Recall: {recall:.4f}")

    # # Plot ROC curve
    # fpr, tpr, _ = roc_curve(y_test, y_prob)
    # plt.figure()
    # plt.plot(fpr, tpr, label=f"ROC Curve (area = {roc_auc:.4f})")
    # plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
    # plt.xlabel("False Positive Rate")
    # plt.ylabel("True Positive Rate")
    # plt.title("Receiver Operating Characteristic")
    # plt.legend(loc="best")
    # plt.show()

"""深度学习"""
def ann_classification(df):
    # Set a manual seed for reproducibility
    torch.manual_seed(42)
    """
    Perform ANN classification using PyTorch on the input DataFrame.

    Parameters:
    df (DataFrame): Input DataFrame where the first column is the ID, middle columns are features, 
                    and the last column is the label.

    Returns:
    None: Prints evaluation metrics and displays the ROC curve.
    """
    # Ensure features and labels are correctly selected
    X = df.iloc[:, 1:-1].values
    y = df.iloc[:, -1].values

    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    X_np = X_tensor.numpy()
    y_np = y_tensor.numpy()

    X_train, X_test, y_train, y_test = train_test_split(X_np, y_np, test_size=0.3, stratify=y_np, random_state=42)

    # Convert back to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    # Recompute sample weights for training set
    class_counts = np.bincount(y_train.astype(int))
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[y_train.astype(int)]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    # Create DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=128, sampler=sampler)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # # Calculate class weights for imbalanced data
    # class_counts = np.bincount(y.astype(int))
    # class_weights = 1.0 / class_counts
    # sample_weights = class_weights[y.astype(int)]

    # sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    # # Split into training and testing datasets
    # dataset = TensorDataset(X_tensor, y_tensor)
    # train_size = int(0.7 * len(dataset))
    # test_size = len(dataset) - train_size

    # train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))

    # train_loader = DataLoader(train_dataset, batch_size=128, sampler=sampler)
    # test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # Define a simple ANN model
    class ANNModel(nn.Module):
        def __init__(self, input_size):
            super(ANNModel, self).__init__()
            self.fc1 = nn.Linear(input_size, 128)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(128, 64)
            self.relu2 = nn.ReLU()
            self.fc3 = nn.Linear(64, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            x = self.relu2(x)
            x = self.fc3(x)
            x = self.sigmoid(x)
            return x

    model = ANNModel(input_size=X.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    for epoch in range(50):  # Number of epochs
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch).squeeze()
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

    # Evaluate the model
    model.eval()
    y_true, y_scores = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            y_pred = model(X_batch).squeeze()
            y_true.extend(y_batch.tolist())
            y_scores.extend(y_pred.tolist())

    y_pred_binary = [1 if score > 0.5 else 0 for score in y_scores]

    # Compute evaluation metrics
    acc = accuracy_score(y_true, y_pred_binary)
    roc_auc = roc_auc_score(y_true, y_scores)
    recall = recall_score(y_true, y_pred_binary)

    print(f"ann_Accuracy: {acc:.4f}")
    print(f"ann_ROC-AUC: {roc_auc:.4f}")
    print(f"ann_Recall: {recall:.4f}")

    # # Plot ROC curve
    # fpr, tpr, _ = roc_curve(y_true, y_scores)
    # plt.figure()
    # plt.plot(fpr, tpr, label=f"ROC Curve (area = {roc_auc:.4f})")
    # plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
    # plt.xlabel("False Positive Rate")
    # plt.ylabel("True Positive Rate")
    # plt.title("Receiver Operating Characteristic")
    # plt.legend(loc="best")
    # plt.show()

"""GNN"""
def gnn_classification(EDGE_FILE, NODE_FILE):
    # 读取边数据
    def load_edges(file_path):
        edge_index = []
        with open(file_path, 'r') as f:
            next(f)  # 跳过标题行
            for line in f:
                node1, node2 = map(int, line.strip().split())  # 假设以空格分隔
                edge_index.append([node1, node2])
        return torch.tensor(edge_index, dtype=torch.long).t()  # 转置为 [2, num_edges]

    # 读取节点数据
    def load_nodes(file_path):
        node_features = []
        labels = []
        with open(file_path, 'r') as f:
            next(f)  # 跳过标题行
            for line in f:
                parts = line.strip().split()
                node_features.append(list(map(float, parts[1].split(','))))  # 特征以逗号分隔
                labels.append(int(parts[2]))  # 标签为整数
        return torch.tensor(node_features, dtype=torch.float), torch.tensor(labels, dtype=torch.long)

    # 加载数据
    edge_index = load_edges(EDGE_FILE)
    node_features, labels = load_nodes(NODE_FILE)

    # 构造训练、验证、测试掩码（随机划分）
    num_nodes = node_features.size(0)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_ratio, val_ratio = 0.6, 0.2
    unique_labels = labels.unique()  # 获取所有唯一标签

    for label in unique_labels:
        label_indices = (labels == label).nonzero(as_tuple=True)[0]  # 获取该标签的所有节点索引
        num_label_nodes = label_indices.size(0)
        num_train = int(num_label_nodes * train_ratio)
        num_val = int(num_label_nodes * val_ratio)

        perm = torch.randperm(num_label_nodes)  # 打乱顺序

        train_mask[label_indices[perm[:num_train]]] = True
        val_mask[label_indices[perm[num_train:num_train + num_val]]] = True
        test_mask[label_indices[perm[num_train + num_val:]]] = True

    # 构造图数据对象
    data = Data(x=node_features, edge_index=edge_index, y=labels, 
                train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

    # 定义GNN模型
    class GCN(torch.nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels):
            super(GCN, self).__init__()
            self.conv1 = GCNConv(in_channels, hidden_channels)  # 第一层卷积
            self.conv2 = GCNConv(hidden_channels, out_channels)  # 第二层卷积

        def forward(self, data):
            x, edge_index = data.x, data.edge_index
            x = self.conv1(x, edge_index)  # 第一层卷积
            x = F.relu(x)  # 激活函数
            x = F.dropout(x, p=0.5, training=self.training)  # Dropout以防止过拟合
            x = self.conv2(x, edge_index)  # 第二层卷积
            return F.log_softmax(x, dim=1)  # 输出类别的对数概率

    # 参数设置
    in_channels = node_features.size(1)  # 动态获取特征维度
    hidden_channels = 64  # 隐藏层维度
    out_channels = 2  # 类别数量

    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN(in_channels, hidden_channels, out_channels).to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

    # 模型训练
    def train():
        model.train()
        optimizer.zero_grad()
        out = model(data)  # 前向传播
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 优化模型
        return loss.item()

    # 模型评估
    def test():
        model.eval()
        with torch.no_grad():
            out = model(data)
            pred = out.argmax(dim=1)  # 获取预测类别
            probs = out.softmax(dim=1)[:, 1].cpu().numpy()  # 获取正类的概率
            accs = []
            rocs = []
            recalls = []
            for mask in [data.train_mask, data.val_mask, data.test_mask]:  # 训练、验证、测试集
                true_labels = data.y[mask].cpu().numpy()
                pred_labels = pred[mask].cpu().numpy()
                probs_masked = probs[mask.cpu().numpy()]

                acc = (pred_labels == true_labels).sum() / len(true_labels)
                roc = roc_auc_score(true_labels, probs_masked) if len(set(true_labels)) > 1 else float('nan')
                recall = recall_score(true_labels, pred_labels, zero_division=0)

                accs.append(acc)
                rocs.append(roc)
                recalls.append(recall)

            return accs, rocs, recalls

    # 训练与测试
    for epoch in range(100):
        loss = train()
        accs, rocs, recalls = test()  # 获取三个列表

        # 按顺序解包列表中的值
        train_acc, val_acc, test_acc = accs
        train_roc, val_roc, test_roc = rocs
        train_recall, val_recall, test_recall = recalls

        if epoch % 10 == 0:
            print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, '
                f'Test Acc: {test_acc:.4f}, Train ROC: {train_roc:.4f}, Val ROC: {val_roc:.4f}, Test ROC: {test_roc:.4f}, '
                f'Train Recall: {train_recall:.4f}, Val Recall: {val_recall:.4f}, Test Recall: {test_recall:.4f}')
        # print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, '
        #     f'Test Acc: {test_acc:.4f}, Train ROC: {train_roc:.4f}, Val ROC: {val_roc:.4f}, Test ROC: {test_roc:.4f}, '
        #     f'Train Recall: {train_recall:.4f}, Val Recall: {val_recall:.4f}, Test Recall: {test_recall:.4f}')

EDGE_FILE, NODE_FILE = remap_node_ids(EDGE_FILE, NODE_FILE, UPDATED_EDGE_FILE, UPDATED_NODE_FILE)
gnn_classification(EDGE_FILE, NODE_FILE)
# ann_classification(df)
# logistic_regression_classification(df)
# random_forest_classification(df)
# svm_classification(df)
# xgboost_classification(df)