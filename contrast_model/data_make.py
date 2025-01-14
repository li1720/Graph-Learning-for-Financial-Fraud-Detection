import pandas as pd


input_file = "/Users/liyiman/coding/NodeFormer/data/fraud/特征表+供应商关系+客户关系+上市公司.xlsx"
df = pd.read_excel(input_file)

columns_to_drop = df.loc[:, "供应商金额1":df.columns[-2]].columns
df_dropped = df.drop(columns=columns_to_drop)

df_cleaned = df_dropped[~df_dropped.iloc[:,1].isnull()]

df_cleaned.to_excel("/Users/liyiman/coding/NodeFormer/data/fraud/contrast_特征表.xlsx",index=False)