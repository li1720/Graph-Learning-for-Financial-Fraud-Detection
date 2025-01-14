import argparse
import sys
import os, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops
from sklearn.neighbors import kneighbors_graph

from logger import Logger
from dataset import load_dataset
from data_utils import load_fixed_splits, adj_mul, get_gpu_memory_map
from eval import evaluate, eval_acc, eval_rocauc, eval_f1, eval_recall
from parse import parse_method, parser_add_main_args
import time

import warnings
warnings.filterwarnings('ignore')

"""设置随机种子"""
# NOTE: for consistent data splits, see data_utils.rand_train_test_idx
def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

"""解析命令行参数"""
### Parse args ###
parser = argparse.ArgumentParser(description='General Training Pipeline')
parser_add_main_args(parser)
args = parser.parse_args()
print(args)

fix_seed(args.seed)

"""设置设备cpu or gpu"""
if args.cpu:
    device = torch.device("cpu")
else:
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
device = torch.device("cpu")

"""加载和预处理数据"""
### Load and preprocess data ###
dataset = load_dataset(args.data_dir, args.dataset, args.sub_dataset)

if len(dataset.label.shape) == 1:
    dataset.label = dataset.label.unsqueeze(1)
dataset.label = dataset.label.to(device)

"""划分数据集"""
# get the splits for all runs
if args.rand_split:
    split_idx_lst = [dataset.get_idx_split(train_prop=args.train_prop, valid_prop=args.valid_prop)
                     for _ in range(args.runs)]
elif args.rand_split_class:
    split_idx_lst = [dataset.get_idx_split(split_type='class', label_num_per_class=args.label_num_per_class)
                     for _ in range(args.runs)]
elif args.dataset in ['ogbn-proteins', 'ogbn-arxiv', 'ogbn-products', 'amazon2m']:
    split_idx_lst = [dataset.load_fixed_splits()
                     for _ in range(args.runs)]
else:
    split_idx_lst = load_fixed_splits(args.data_dir, dataset, name=args.dataset, protocol=args.protocol)

#
if args.dataset in ('mini', '20news'):
    adj_knn = kneighbors_graph(dataset.graph['node_feat'], n_neighbors=args.knn_num, include_self=True)
    edge_index = torch.tensor(adj_knn.nonzero(), dtype=torch.long)
    dataset.graph['edge_index']=edge_index


"""处理图的基本信息"""
### Basic information of datasets ###
n = dataset.graph['num_nodes']
e = dataset.graph['edge_index'].shape[1]
# infer the number of classes for non one-hot and one-hot labels
c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
d = dataset.graph['node_feat'].shape[1]

print(f"dataset {args.dataset} | num nodes {n} | num edge {e} | num node feats {d} | num classes {c}")


"""邻接矩阵的处理，对陈化、去自环、加自环————【后续改进，进行有向图处理】"""
# whether or not to symmetrize
if not args.directed and args.dataset != 'ogbn-proteins':
    dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])

dataset.graph['edge_index'], dataset.graph['node_feat'] = \
    dataset.graph['edge_index'].to(device), dataset.graph['node_feat'].to(device)


"""加载模型并设置损失函数"""
### Load method ###
model = parse_method(args, dataset, n, c, d, device)

### Loss function (Single-class, Multi-class) ###
class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # p_t is the probability of the true class
        F_loss = self.alpha * pt ** self.gamma * BCE_loss
        return F_loss.mean()
    
if args.dataset in ('yelp-chi', 'deezer-europe', 'twitch-e', 'fb100', 'ogbn-proteins'):
    criterion = nn.BCEWithLogitsLoss()
else:
    # criterion = nn.NLLLoss()
    # criterion = nn.BCELoss(torch.tensor(19))
    # criterion = nn.BCELoss()
    criterion = nn.BCELoss(weight=torch.tensor([10.0])) 
    # criterion = FocalLoss(alpha = 0.25, gamma=2.0)

"""选择评估指标"""
### Performance metric (Acc, AUC, F1) ###
if args.metric == 'rocauc':
    eval_func = eval_rocauc
elif args.metric == 'f1':
    eval_func = eval_f1
else:
    eval_func = eval_acc
    # eval_func = eval_f1


"""训练模型"""
logger = Logger(args.runs, args)

model.train()
print('MODEL:', model)

### Adj storage for relational bias ###
adjs = []
adj, _ = remove_self_loops(dataset.graph['edge_index'])
adj, _ = add_self_loops(adj, num_nodes=n)
adjs.append(adj)
for i in range(args.rb_order - 1): # edge_index of high order adjacency
    adj = adj_mul(adj, adj, n)
    adjs.append(adj)
dataset.graph['adjs'] = adjs

### Training loop ###
res_list = []
for run in range(args.runs):
    if args.dataset in ['cora', 'citeseer', 'pubmed'] and args.protocol == 'semi':
        split_idx = split_idx_lst[0]
    else:
        split_idx = split_idx_lst[run]
        # print(len(split_idx["train"]))
        # print(len(split_idx["valid"]))
        # print(len(split_idx["test"]))
    train_idx = split_idx['train'].to(device)
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(),weight_decay=args.weight_decay, lr=args.lr)
    best_val = float('-inf')

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()

        if args.method == 'nodeformer':
            out, link_loss_ = model(dataset.graph['node_feat'], dataset.graph['adjs'], args.tau)
            
        else:
            out = model(dataset)

        if args.dataset in ('yelp-chi', 'deezer-europe', 'twitch-e', 'fb100', 'ogbn-proteins'):
            if dataset.label.shape[1] == 1:
                true_label = F.one_hot(dataset.label, dataset.label.max() + 1).squeeze(1)
            else:
                true_label = dataset.label
            loss = criterion(out[train_idx], true_label.squeeze(1)[
                train_idx].to(torch.float))
        else:
            # print(out)
            out = F.softmax(out)
            # print("out : ",out)
            loss = criterion(
                out[train_idx][:,1], dataset.label.squeeze(1)[train_idx].float())
            # loss = criterion(
            #     out[train_idx], dataset.label.squeeze(1)[train_idx])
            # print(out[train_idx], dataset.label.squeeze(1)[train_idx])
            # break
        # if args.method == 'nodeformer':
        #     loss -= args.lamda * sum(link_loss_) / len(link_loss_)
        loss.backward()
        optimizer.step()

        if epoch % args.eval_step == 0:
            result = evaluate(model, dataset, split_idx, eval_acc, criterion, args)
            result1 = evaluate(model, dataset, split_idx, eval_rocauc, criterion, args)
            result2 = evaluate(model, dataset, split_idx, eval_recall, criterion, args)
            logger.add_result(run, result[:-1])

            if result[1] > best_val:
                best_val = result[1]
                if args.save_model:
                    torch.save(model.state_dict(), args.model_dir + f'{args.dataset}-{args.method}.pkl')

            res = (f'Epoch: {epoch:02d}, '
                f'Loss: {loss:.4f}, '
                f'Train acc: {100 * result[0]:.2f}%, '
                f'Valid acc: {100 * result[1]:.2f}%, '
                f'Test acc: {100 * result[2]:.2f}%, '
                f'Train rocauc: {100 * result1[0]:.2f}%, '
                f'Valid rocauc: {100 * result1[1]:.2f}%, '
                f'Test rocauc: {100 * result1[2]:.2f}%, '
                f'Train recall: {100 * result2[0]:.2f}%, '
                f'Valid recall: {100 * result2[1]:.2f}%, '
                f'Test recall: {100 * result2[2]:.2f}%')
            res_list.append(res)
            print(res)
            
            
    logger.print_statistics(run)


import datetime
current_time = datetime.datetime.now()
timestamp = current_time.strftime("%m%d_%H%M")
with open(f'logs/log_{timestamp}.txt', 'w') as f:
    for res in res_list:
        f.write(res + '\n')  # 每个结果写入一行，并且在每行的末尾加上换行符
    
# ### Save results ###
# filename = f'results/{args.dataset}.csv'
# print(f"Saving results to {filename}")
# with open(f"{filename}", 'a+') as write_obj:
#     write_obj.write(f"{args.method}," +
#                     f"{best_val.mean():.3f} ± {best_val.std():.3f}," +
#                     f"{best_test.mean():.3f} ± {best_test.std():.3f}\n")