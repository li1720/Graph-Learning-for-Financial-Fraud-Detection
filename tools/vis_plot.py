import matplotlib.pyplot as plt
import numpy as np

# Step 1: 读取日志文件内容
log_file = '/Users/liyiman/coding/NodeFormer/logs/log_0115_2155.txt'  # 请确保文件路径正确
epochs = []
losses = []
train_acc = []
valid_acc = []
test_acc = []
train_rocauc = []
valid_rocauc = []
test_rocauc = []
train_recall = []
valid_recall = []
test_recall = []

# 读取文件并解析数据
with open(log_file, 'r') as f:
    for line in f:
        # 从每行中提取需要的数字
        parts = line.split(', ')
        epoch = int(parts[0].split(': ')[1])
        loss = float(parts[1].split(': ')[1])
        t_acc = float(parts[2].split(': ')[1].replace('%', ''))
        v_acc = float(parts[3].split(': ')[1].replace('%', ''))
        te_acc = float(parts[4].split(': ')[1].replace('%', ''))
        t_rocauc = float(parts[5].split(': ')[1].replace('%', ''))
        v_rocauc = float(parts[6].split(': ')[1].replace('%', ''))
        te_rocauc = float(parts[7].split(': ')[1].replace('%', ''))
        t_recall = float(parts[8].split(': ')[1].replace('%', ''))
        v_recall = float(parts[9].split(': ')[1].replace('%', ''))
        te_recall = float(parts[10].split(': ')[1].replace('%', ''))

        # Append the values to the respective lists
        epochs.append(epoch)
        losses.append(loss)
        train_acc.append(t_acc)
        valid_acc.append(v_acc)
        test_acc.append(te_acc)
        train_rocauc.append(t_rocauc)
        valid_rocauc.append(v_rocauc)
        test_rocauc.append(te_rocauc)
        train_recall.append(t_recall)
        valid_recall.append(v_recall)
        test_recall.append(te_recall)

# Step 2: 使用 matplotlib 进行可视化

# 创建一个 2x2 的图形窗口
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 图1: Loss 曲线
axes[0, 0].plot(epochs, losses, color='tab:blue', label='Loss')
axes[0, 0].set_title('Loss vs Epochs')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].grid(True)

# 图2: Accuracy 曲线
axes[0, 1].plot(epochs, train_acc, color='tab:green', label='Train Accuracy')
axes[0, 1].plot(epochs, valid_acc, color='tab:orange', label='Valid Accuracy')
axes[0, 1].plot(epochs, test_acc, color='tab:red', label='Test Accuracy')
axes[0, 1].set_title('Accuracy vs Epochs')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Accuracy (%)')
axes[0, 1].legend()
axes[0, 1].grid(True)

# 图3: ROC AUC 曲线
axes[1, 0].plot(epochs, train_rocauc, color='tab:purple', label='Train ROC AUC')
axes[1, 0].plot(epochs, valid_rocauc, color='tab:brown', label='Valid ROC AUC')
axes[1, 0].plot(epochs, test_rocauc, color='tab:cyan', label='Test ROC AUC')
axes[1, 0].set_title('ROC AUC vs Epochs')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('ROC AUC (%)')
axes[1, 0].legend()
axes[1, 0].grid(True)

# 图4: Recall 曲线
axes[1, 1].plot(epochs, train_recall, color='tab:gray', label='Train Recall')
axes[1, 1].plot(epochs, valid_recall, color='tab:olive', label='Valid Recall')
axes[1, 1].plot(epochs, test_recall, color='tab:pink', label='Test Recall')
axes[1, 1].set_title('Recall vs Epochs')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Recall (%)')
axes[1, 1].legend()
axes[1, 1].grid(True)

# 自动调整子图之间的间距
plt.tight_layout()

# 显示图形
plt.savefig(f'figs/test.jpg')
