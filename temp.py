import os
import pandas as pd
from tqdm import tqdm
from functools import reduce
from collections import Counter
import matplotlib.pyplot as plt

df = pd.read_excel("/Users/liyiman/coding/NodeFormer/data/fraud/关系表.xlsx")
supplier_list = df["供应商名称"]
stock_list = df["股票代码"]
count = Counter(supplier_list)
# 获取出现次数最多的前10个元素及其计数
top_ten = count.most_common(100)
# 打印结果
for item, frequency in top_ten:
    print(f"Item: {item}, Count: {frequency}")