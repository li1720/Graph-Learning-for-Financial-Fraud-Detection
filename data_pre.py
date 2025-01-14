# -*- coding: utf-8 -*-
import os
import pandas as pd
from tqdm import tqdm
from functools import reduce
import numpy as np
import re
import gc

# 处理管理层讨论与分析
"""
# 定义文件夹路径和年份范围
base_folder = "/Users/liyiman/Desktop/毕业论文/新的论文/复现备选/dataaaaa/管理层讨论与分析"
years = [str(year) for year in range(2001, 2024)]

# 用于存储每年处理后的 DataFrame
data_frames = []

for year in tqdm(years):
    # 构造文件夹和文件路径
    folder_path = os.path.join(base_folder, year)
    file_name = f"管理层讨论与分析_{year}.xlsx"
    file_path = os.path.join(folder_path, file_name)

    # 读取 Excel 文件
    df = pd.read_excel(file_path)

    df.columns = df.iloc[0]
    df = df[1:]

    # 筛选经营分析时间为 12 月 31 日的行
    df = df[df['经营分析时间'].str.endswith('12-31')]

    # 选择需要的列
    columns_to_keep = ['股票代码', '会计年度', '正面词汇数量', '负面词汇数量']
    df = df[columns_to_keep]



    # 添加到结果列表
    data_frames.append(df)


# 合并所有 DataFrame
final_df = pd.concat(data_frames, ignore_index=True)

# 保存合并结果到新的 Excel 文件
final_df.to_excel("/Users/liyiman/coding/NodeFormer/data/try/合并后的管理层讨论与分析.xlsx", index=False)

"""

# 处理财经新闻情感
"""
# 定义文件夹路径和年份范围
base_folder = "/Users/liyiman/Desktop/毕业论文/新的论文/复现备选/dataaaaa/CFND-网络财经新闻基本信息(全部年份)"
years = [str(year) for year in range(2001, 2024)]

# 用于存储每年处理后的 DataFrame
data_frames = []

for year in years:
    # 构造文件夹和文件路径
    folder_path = os.path.join(base_folder, year)
    excel_files = excel_files = [f for f in os.listdir(folder_path) if f.startswith(f"网络财经新闻基本信息-{year}") and f.endswith(".xlsx")]
    for excel_file in excel_files:

        file_path = os.path.join(folder_path, excel_file)

        # 读取 Excel 文件
        df = pd.read_excel(file_path)

        df.columns = df.iloc[0]
        df = df[1:]

        # 选择需要的列
        columns_to_keep = ['股票代码', '报道时间', '新闻情感']
        df = df[columns_to_keep]

        df['报道时间'] = pd.to_datetime(df['报道时间']).dt.year

        # 添加到结果列表
        data_frames.append(df)

# 合并所有 DataFrame
final_df = pd.concat(data_frames, ignore_index=True)

# 按年份和股票代码计算正面词汇数量的中性、积极、消极情绪个数
final_df['新闻情感'] = final_df['新闻情感'].astype(int)
result = final_df.groupby(['股票代码', '报道时间']).apply(
    lambda group: pd.Series({
        '中性情绪': (group['新闻情感'] == 0).sum(),
        '积极情绪': (group['新闻情感'] == 1).sum(),
        '消极情绪': (group['新闻情感'] == -1).sum()
    })
).reset_index()

# 保存合并结果到新的 Excel 文件
result.to_excel("/Users/liyiman/coding/NodeFormer/data/try/合并后的网络财经新闻基本信息.xlsx", index=False)
"""


# 处理造假信息表
"""
df = pd.read_excel("/Users/liyiman/Desktop/毕业论文/新的论文/复现备选/dataaaaa/造假信息.xlsx", dtype = {"证券代码": str})
df = df[['证券代码', '违规年度']]
df_ed = df.assign(违规年度=df['违规年度'].str.split(';')).explode('违规年度')
df_ed['违规年度'] = df_ed['违规年度'].apply(lambda x: str(x) if isinstance(x, list) else x)
df_ed = df_ed.drop_duplicates()
df_ed = df_ed[df_ed['违规年度'] != "N/A"]
df_ed = df_ed.dropna(subset=["违规年度"])
df_ed.to_excel("/Users/liyiman/coding/NodeFormer/data/try/展开后的造假信息.xlsx", index=False)
"""

# 数据清洗三表
"""
df1 = pd.read_excel("/Users/liyiman/Desktop/毕业论文/新的论文/复现备选/dataaaaa/利润表.xlsx", dtype = {"证券代码": str})
df2 = pd.read_excel("/Users/liyiman/Desktop/毕业论文/新的论文/复现备选/dataaaaa/现金流量表.xlsx", dtype = {"证券代码": str})
df3 = pd.read_excel("/Users/liyiman/Desktop/毕业论文/新的论文/复现备选/dataaaaa/现金流量表2.xlsx", dtype = {"证券代码": str})
df4 = pd.read_excel("/Users/liyiman/Desktop/毕业论文/新的论文/复现备选/dataaaaa/资产负债表.xlsx", dtype = {"证券代码": str})
dfs = [df1, df2, df3, df4]
for i, df in enumerate(tqdm(dfs)):
    df = df[df["报表类型"] == "A"]
    df = df[df['统计截止日期'].str.endswith('12-31')]
    df['统计截止日期'] = pd.to_datetime(df['统计截止日期']).dt.year
    df = df.drop(columns=['报表类型', '证券简称'])
    dfs[i] = df
dfss = [dfs[0], dfs[2], dfs[3]]
df = reduce(lambda left, right: pd.merge(left, right, on=['证券代码', '统计截止日期'], how='inner'), dfss)
df_merge = pd.merge(df, dfs[1], on=['证券代码', '统计截止日期'], how='left')

df.to_excel("/Users/liyiman/coding/NodeFormer/data/try/三表.xlsx", index=False)
"""

# 合并新表
"""
df1 = pd.read_excel("/Users/liyiman/coding/NodeFormer/data/try/合并后的管理层讨论与分析.xlsx", dtype = {"证券代码": str})
df2 = pd.read_excel("/Users/liyiman/coding/NodeFormer/data/try/合并后的网络财经新闻基本信息.xlsx", dtype = {"证券代码": str})
df3 = pd.read_excel("/Users/liyiman/coding/NodeFormer/data/try/三表.xlsx", dtype = {"证券代码": str})
df4 = pd.read_excel("/Users/liyiman/coding/NodeFormer/data/try/行业数据.xlsx", dtype = {"证券代码": str})
df5 = pd.read_excel("/Users/liyiman/coding/NodeFormer/data/try/展开后的造假信息.xlsx", dtype = {"证券代码": str})
df6 = pd.read_excel("/Users/liyiman/Desktop/毕业论文/新的论文/复现备选/dataaaaa/董监高持股.xlsx", dtype = {"证券代码": str})
df7 = pd.read_excel("/Users/liyiman/Desktop/毕业论文/新的论文/复现备选/dataaaaa/股权集中度.xlsx", dtype = {"证券代码": str})
df8 = pd.read_excel("/Users/liyiman/Desktop/毕业论文/新的论文/复现备选/dataaaaa/居民消费.xlsx", dtype = {"证券代码": str})
df9 = pd.read_excel("/Users/liyiman/Desktop/毕业论文/新的论文/复现备选/dataaaaa/审计单位及审计意见.xlsx", dtype = {"证券代码": str})
df10 = pd.read_excel("/Users/liyiman/Desktop/毕业论文/新的论文/复现备选/dataaaaa/gdp.xlsx", dtype = {"证券代码": str})
dfs1 = [df4, df2, df3, df1, df5, df6, df7, df9]
df_m = reduce(lambda left, right: pd.merge(left, right, on=['证券代码', '会计年度'], how='left'), dfs1)
dfs2 = [df_m, df8, df10]
df_merge = reduce(lambda left, right: pd.merge(left, right, on=["会计年度"], how='left'), dfs2)
df_merge.to_excel("/Users/liyiman/coding/NodeFormer/data/try/合并特征表.xlsx", index=False)
"""

# 合并表清洗
"""
df = pd.read_excel("/Users/liyiman/coding/NodeFormer/data/try/合并特征表.xlsx", dtype = {"证券代码": str, "会计师事务所": str, "审计意见": str})
df["是否造假"] = df["是否造假"].fillna(0)
df = df[(df['会计年度'] >= 2003) & (df['会计年度'] <= 2023)]
# 定义需要移除的前缀列表
prefixes_to_remove = ['北京', '大连', '福建省', '福建', '福州', '甘肃', '广东', '广州', 
                      '河北', '河南', '湖北', '湖南', '江苏', '辽宁', '厦门', '南京',
                      '山东', '山西', '上海', '深圳', '四川', '天津', '武汉', '西安',
                      '云南', '浙江', '中国北京', '中国','重庆']    
def remove_prefix(row):
    for prefix in prefixes_to_remove:
        if isinstance(row, str):  
            if row.startswith(prefix):
                return row[len(prefix):]  # 移除匹配的前缀
    return row  # 如果没有匹配的前缀，则保持不变
# 移除事务所列中匹配的前缀
df['会计师事务所'] = df['会计师事务所'].apply(remove_prefix)
# 对事务所进行唯一编码
unique_firms = {name: idx for idx, name in enumerate(df['会计师事务所'].unique(), 1)}
df['会计师事务所'] = df['会计师事务所'].map(unique_firms)
# 对审计意见进行唯一编码
unique_opinions = {opinion: idx for idx, opinion in enumerate(df['审计意见'].unique(), 1)}
df['审计意见'] = df['审计意见'].map(unique_opinions)
df.to_excel("/Users/liyiman/coding/NodeFormer/data/fraud/特征表.xlsx", index=False)
"""
# 这里空也进行了赋值 13， 4



# 关系表
"""
df1 = pd.read_excel("/Users/liyiman/Desktop/毕业论文/新的论文/复现备选/dataaaaa/供应链供应商关系.xlsx", dtype = {"统计截止日期": str, "股票代码": str, "公司股票代码": str, "供应商公司ID": str})
df2 = pd.read_excel("/Users/liyiman/Desktop/毕业论文/新的论文/复现备选/dataaaaa/供应链客户关系.xlsx", dtype = {"统计截止日期": str, "股票代码": str, "公司股票代码": str, "供应商公司ID": str})
df = pd.read_excel("/Users/liyiman/coding/NodeFormer/data/fraud/特征表.xlsx", dtype = {"证券代码": str, "会计年度": str} )
df["编号"] = df['会计年度'].astype(str) + df['证券代码'].astype(str)
columns = ['编号'] + [col for col in df.columns if col != '编号']
df = df[columns]
"""

"""
# 清洗关系表
def clean_df(df):
    df = df[df['统计截止日期'].str.endswith('12-31')]
    df['统计截止日期'] = pd.to_datetime(df['统计截止日期']).dt.year
    df = df[(df['统计截止日期'] >= 2003) & (df['统计截止日期'] <= 2023)]
    df = df[df['报表类型'] == 1]
    # 处理股票代码
    def process_code(code):
        code = str(code)  # 强制类型转换为 str
        if pd.isna(code):
            return code  # 跳过 NaN
        if ";" in code:
            return code.split(";")[0]
        return code
    df["公司股票代码"] = df["公司股票代码"].apply(process_code)
    # 替换规则：如果 "股票代码" 列不是空，则用 "股票代码" 的值替换 "编号" 列的对应行
    def not_empty(s):
        return bool(s.strip())
    df["供应商公司ID"] = df.apply(lambda row: row["公司股票代码"] if not_empty(row["公司股票代码"]) and row["公司股票代码"]!= "nan"  else row["供应商公司ID"], axis=1)
    # 补全供应商残缺
    df["编号"] = df["统计截止日期"].astype(str) + df["股票代码"].astype(str)
    return df


# 生成平齐的5个公司
def creat_five(df):

    def process_company(group):
        num_rows = len(group)
        if num_rows == 1:
            # 插入 5 行，金额和比例平分
            amount = group.iloc[0]["供应商采购额"] / 5
            ratio = group.iloc[0]["供应商采购额占比"] / 5
            new_rows = [
                {**group.iloc[0].to_dict(), "排名": i + 1, "供应商采购额": amount, "供应商采购额占比": ratio}
                for i in range(5)
            ]
            return pd.concat([group, pd.DataFrame(new_rows)], ignore_index=True)
        elif 1 < num_rows < 6:
            # 插入 6 - i 行
            existing_amount = group[group["排名"] < 6]["供应商采购额"].sum()
            existing_ratio = group[group["排名"] < 6]["供应商采购额占比"].sum()
            # rank_6_rows = group[group["排名"] == 6]
            rank_1_rows = group[group["排名"] == 1]
            print("Rank 1 rows:", rank_1_rows)
            amount_to_distribute = group[group["排名"] == 6].iloc[0]["供应商采购额"] - existing_amount
            ratio_to_distribute = group[group["排名"] == 6].iloc[0]["供应商采购额占比"] - existing_ratio
            num_new_rows = 6 - num_rows
            amount = amount_to_distribute / num_new_rows
            ratio = ratio_to_distribute / num_new_rows
            new_rows = [
                {**group.iloc[0].to_dict(), "排名": num_rows + i, "供应商采购额": amount, "供应商采购额占比": ratio}
                for i in range(num_new_rows)
            ]
            return pd.concat([group, pd.DataFrame(new_rows)], ignore_index=True)
        else:
            return group[group["排名"] <= 6]
    # 按 "上市公司" 分组并生成平齐的5个公司
    df = df.groupby("编号", group_keys=False).apply(process_company)
    # 删除 "排名" 为 6 的行
    # df = df[df["排名"] < 6]
    return df


# 将供应商和客户的交易情况匹配到特征表
# df1为特征表，df为关系表，entity_type为客户or供应商，返回df1特征表
def match_and_insert(df1, df, entity_type):
    for idx, row in df1.iterrows():
        matching_rows = df[df['编号'] == row['编号']]
        if not matching_rows.empty:
            # 按排名排序
            sorted_rows = matching_rows.sort_values(by='排名')
            for i in range(1, 6):
                rank_row = sorted_rows[sorted_rows['排名'] == i]
                if not rank_row.empty:
                    df1.at[idx, f'{entity_type}金额{i}'] = rank_row['供应商采购额'].values[0]
                    df1.at[idx, f'{entity_type}比例{i}'] = rank_row['供应商采购额占比'].values[0]
            # 插入总计
            total_row = sorted_rows[sorted_rows['排名'] == 6]
            if not total_row.empty:
                df1.at[idx, f'{entity_type}金额合计'] = total_row['供应商采购额'].values[0]
                df1.at[idx, f'{entity_type}比例合计'] = total_row['供应商采购额占比'].values[0]
    return df1


# 清理和处理供应商名称杂乱问题
def clean_supplier_names(df):
    # 筛选个数大于6或上市公司的供应商名称
    df = df[(df['供应商名称'].str.len() > 6) | df['供应商公司ID'].notnull()]

    # 清理特殊符号和地名
    # df['供应商名称'] = df['供应商名称'].str.replace(r'[。，：:.,（【】）)([]{}*- —\\s]', '', regex=True)
    tqdm.pandas(desc="Cleaning symbols")
    df['供应商名称'] = df['供应商名称'].progress_apply(lambda x: re.sub(r'[。，：:.,（【】）)([]{}*- —\\s]', '', str(x)))
    # 清理
    province_names = ["北京市","天津市","上海市","重庆市","河北省","山西省","辽宁省","吉林省","黑龙江省","江苏省","浙江省","安徽省","福建省","江西省","山东省","河南省","湖北省","湖南省","广东省","海南省","四川省","贵州省","云南省","陕西省","甘肃省","青海省","台湾","内蒙古","广西","西藏","宁夏","新疆","香港","澳门", "北京","天津","上海","重庆","河北","山西","辽宁","吉林","黑龙江","江苏","浙江","安徽","福建","江西","山东","河南","湖北","湖南","广东","海南","四川","贵州","云南","陕西","甘肃","青海", "广州市","深圳市","成都市","杭州市","重庆市","武汉市","西安市","苏州市","天津市","南京市","郑州市","长沙市","东莞市","青岛市","佛山市","沈阳市","昆明市","宁波市","合肥市","无锡市","厦门市","哈尔滨市","济南市","福州市","南昌市","大连市","长春市","石家庄市","贵阳市","南宁市","太原市","乌鲁木齐市","常州市","温州市","珠海市","泉州市","金华市","惠州市","海口市","兰州市","徐州市","绍兴市","台州市","扬州市","烟台市","潍坊市","洛阳市","嘉兴市","泰州市","镇江市","南通市","威海市", "北京","上海","广州","深圳","成都","杭州","重庆","武汉","西安","苏州","天津","南京","郑州","长沙","东莞","青岛","佛山","沈阳","昆明","宁波","合肥","无锡","厦门","哈尔滨","济南","福州","南昌","大连","长春","石家庄","贵阳","南宁","太原","乌鲁木齐","常州","温州","珠海","泉州","金华","惠州","海口","兰州","徐州","绍兴","台州","扬州","烟台","潍坊","洛阳","嘉兴","泰州","镇江","南通","威海"]  # 示例地名，需补充完整
    province_pattern = '|'.join(province_names)  # 构造正则表达式
    # df['供应商名称'] = df['供应商名称'].str.replace(province_pattern, '', regex=True)
    tqdm.pandas(desc="Cleaning province names")
    df['供应商名称'] = df['供应商名称'].progress_apply(lambda x: re.sub(province_pattern, '', str(x)))

    # 前5个字相同的供应商归一化
    df['供应商名称'] = df['供应商名称'].str[:5]
    grouped = df.groupby('供应商名称')
    for name, group in tqdm(grouped, desc="Cleaning supplier names"):
        if group['供应商公司ID'].notnull().any():
            unified_name = group.loc[group['供应商公司ID'].notnull(), '供应商名称'].iloc[0]
        else:
            unified_name = group['供应商名称'].mode().iloc[0]
        df.loc[group.index, '供应商名称'] = unified_name
        df.loc[group.index, '供应商公司ID'] = group['供应商公司ID'].iloc[0] if group['供应商公司ID'].notnull().any() else np.nan

    # 为没有编号的供应商生成6位编号
    numeric_ids = df['供应商公司ID'].apply(lambda x: str(x).isdigit())
    max_existing_id = df.loc[numeric_ids, '供应商公司ID'].astype(int).max()
    # max_existing_id = df['供应商公司ID'].dropna().astype(int).max()
    new_id_start = max_existing_id + 1 if pd.notnull(max_existing_id) else 100000
    new_ids = {name: str(new_id_start + i) for i, name in enumerate(df['供应商名称'].unique()) if name not in df['供应商公司ID']}
    df['供应商公司ID'] = df.apply(lambda x: new_ids[x['供应商名称']] if pd.isnull(x['供应商公司ID']) else x['供应商公司ID'], axis=1)

    return df


# 合并非上市公司到特征表——生成新表
def sync_and_expand(df, df3):
    # 找到 df3 中 `供应商编号` 不存在于 df 的 `编号` 中的行
    unmatched_rows = df3[~df3['供应商编号'].isin(df['编号'])]

    # 创建一个编号的分组
    grouped = unmatched_rows.groupby('供应商编号')

    # 新行列表，用于收集新生成的行
    new_rows = []

    # 使用 tqdm 添加进度条
    for supply_id, group in tqdm(grouped, desc="Processing unmatched rows", total=len(grouped)):
        if not df['编号'].isin([supply_id]).any():
            # 创建新行并添加编号
            new_row = {'编号': supply_id}
            group = group.reset_index(drop=True)
            # 动态加入供应商采购额和采购额占比列
            for idx, row in group.iterrows():
                new_row[f'非上市采购额{idx+1}'] = row['供应商采购额']
                new_row[f'非上市采购额占比{idx+1}'] = row['供应商采购额占比']
            
            # 将新行添加到 new_rows 列表
            new_rows.append(new_row)
                    # 定期释放内存
        if len(new_rows) > 10000:  # 每处理 10000 行就释放一次内存
            gc.collect()
    
    # 将收集的 new_rows 转换成 DataFrame
    new_df = pd.DataFrame(new_rows)

    return new_df


df1 = creat_five(clean_df(df1))
df2 = creat_five(clean_df(df2))

df = match_and_insert(df, df1, "供应商")
df = match_and_insert(df, df2, "客户")

df3 = pd.concat([df1, df2], ignore_index=True)

df3 = clean_supplier_names(df3)

df3["供应商编号"] = df3['统计截止日期'].astype(str) + df3['供应商公司ID'].astype(str)


# 清洗特征表
df['证券代码'] = df['证券代码'].astype(str)
df = df[~df['证券代码'].str.startswith(('8', '9'))]



# 创建一个字典，将字母映射到数字
letter_to_number = {letter: idx + 1 for idx, letter in enumerate(df['行业代码'].unique())}
# 用映射字典替换行业代码列中的字母为数字
df['行业代码'] = df['行业代码'].map(letter_to_number)



# 设置要填充平均值的列索引
indexes_to_fill_average = [4, 5, 6, 33, 34]
# 遍历这些列的索引
for idx in indexes_to_fill_average:
    # 计算该列的平均值，排除空值
    mean_value = df.iloc[:, idx].mean()
    # 用该列的平均值填充空缺值
    df.iloc[:, idx] = df.iloc[:, idx].fillna(mean_value)



# 填充索引从7到32以及35到47的列的空缺值为0
for idx in range(7, 33):  # 索引7到32
    df.iloc[:, idx] = df.iloc[:, idx].fillna(0)

for idx in range(35, 48):  # 索引35到47
    df.iloc[:, idx] = df.iloc[:, idx].fillna(0)



df = df.drop(columns=['审计费用', '证券代码'])
df['会计年度'] = df['会计年度'].astype(int)



for idx in range(-1, -25, -1):
    # 计算该列的平均值，排除空值
    mean_value = df.iloc[:, idx].mean()
    # 用该列的平均值填充空缺值
    df.iloc[:, idx] = df.iloc[:, idx].fillna(mean_value)



# 审计意见列进行独热编码替换
# audit_column_index = 49
# 生成审计意见列的独热编码
audit_dummies = pd.get_dummies(df["审计意见"], prefix='audit_opinion')
# 将布尔值转换为0和1
audit_dummies = audit_dummies.astype(int)
# 将独热编码的列添加到原数据框中，放置在"审计意见"列后面
df = pd.concat([df.iloc[:, :df.columns.get_loc('审计意见') + 1], audit_dummies, df.iloc[:, df.columns.get_loc('审计意见') + 1:]], axis=1)
df = df.drop(columns=['审计意见'])




# 将'是否造假'这一列移动到最后
is_fraud_column = df.pop('是否造假')  # 提取'是否造假'这一列
df['是否造假'] = is_fraud_column  # 将'是否造假'添加到最后



df1 = sync_and_expand(df, df3)
df1['是否造假'] = 0



columns_to_keep = ['编号', '供应商编号']
df3 = df3[columns_to_keep]


df3.to_excel("/Users/liyiman/coding/NodeFormer/data/fraud/交易关系表_end.xlsx", index=False)
df.to_excel("/Users/liyiman/coding/NodeFormer/data/fraud/特征表+供应商关系+客户关系+上市公司.xlsx", index=False)
df1.to_excel("/Users/liyiman/coding/NodeFormer/data/fraud/特征表+供应商关系+客户关系+非上市公司.xlsx", index=False)

"""


df3 = pd.read_excel("/Users/liyiman/coding/NodeFormer/data/fraud/交易关系表_end.xlsx")
df = pd.read_excel("/Users/liyiman/coding/NodeFormer/data/fraud/特征表+供应商关系+客户关系+上市公司.xlsx")
df1 = pd.read_excel("/Users/liyiman/coding/NodeFormer/data/fraud/特征表+供应商关系+客户关系+非上市公司.xlsx")
df['编号'] = df['编号'].astype(str)
df3['供应商编号'] = df3['供应商编号'].astype(str)
df3['编号'] = df3['编号'].astype(str)
df = df[~df['编号'].str.contains(r'[a-zA-Z]', na=False)]
df3 = df3[~df3['供应商编号'].str.contains(r'[a-zA-Z]', na=False)]
df3 = df3[df3['编号'].isin(df['编号'])]
# df = df.head(5)
# df1 = df1.head(5)
# df3 = df3.head(5)

df3.to_csv('/Users/liyiman/coding/NodeFormer/data/geom-gcn/film/out1_graph_edges.txt', sep='\t', index=False)  # sep='\t' 表示制表符分隔，可以更改为其他分隔符

# 为了适配训练而进行清洗
"""
# 处理审计意见
# 获取从 'audit_opinion_1' 开始的8列数据
audit_columns = df.columns[df.columns.str.startswith('audit_opinion')][:8]
# 根据规则生成 '审计意见' 列
def get_audit_opinion(row):
    for i, col in enumerate(audit_columns, start=1):
        if row[col] == 1:
            return i  # 返回对应的值
    return 0  # 如果没有1，返回0
df['审计意见'] = df.apply(get_audit_opinion, axis=1)
audit_opinion_index = df.columns.get_loc('audit_opinion_1')
df.insert(audit_opinion_index, '审计意见', df.pop('审计意见'))
df.drop(["audit_opinion_1", "audit_opinion_2", "audit_opinion_3", "audit_opinion_4", "audit_opinion_5", "audit_opinion_6", "audit_opinion_7", "audit_opinion_8"])
"""


# 拼接 df 和 df1
# df_combined = pd.concat([df, df1], ignore_index=True)
df = df.fillna(0)

df['特征列'] = df.drop(columns=['编号', '是否造假']).apply(lambda row: ','.join(row.dropna().astype(str)), axis=1)

# 按顺序排列列：编号列, 特征列, 是否造假列
df_combined = df[['编号', '特征列', '是否造假']]
df_combined['是否造假'] = df_combined['是否造假'].astype(int)


# 将数据保存为 txt 文件
df_combined.to_csv('/Users/liyiman/coding/NodeFormer/data/geom-gcn/film/out1_node_feature_label.txt', index=False, sep='\t')
