"""
max <= 100，ont-hot为7位
max > 100,ont-hot为37位
"""

import math
import numpy as np

# print(2**math.ceil(math.log2(3000)))

def int_to_unsignçed_binary(num, bit_width):
    """将整数转换为无符号二进制字符串，不支持负数"""
    if num < 0:
        binary = format(abs(num), f'0{bit_width}b')  # 固定宽度显示
        return f'1{binary}'
    else:
        binary = format(num, f'0{bit_width}b')  # 固定宽度显示
        return f'0{binary}'


def build(feature):
    feature_blank = []
    max_nums = [2023, 19, 21183, 35071, 22873, 3318170000000, 2819363000000, 295015000000, 27816000000, 57167000000,
                180291000000, 456596000000, 332948000000, 96364924000, 125317000000, 358791000000, 257699000000,
                272695000000, 1075620000000, 2048963000000, 280972000000, 732577000000, 2903330000000, 689958000000,
                1590433000000, 458113000000, 567473300000, 2172253000000, 455001000000, 1147800000000, 1630621000000,
                6068, 4014, 90, 100, 100, 49, 183021000000, 34, 33, 13, 100, 1, 1, 2, 2, 222, 1, 1, 1, 1, 1, 1, 1, 1,
                34965, 1249991, 1260582, 8304800000000, 100, 8304800000000, 100, 8304800000000, 100, 8304800000000,
                100, 8304800000000, 100, 4080410000000, 111, 3114750000000, 111, 2439780000000,
                111, 2439780000000, 111, 2439773000000, 111]

    lens_list = [math.ceil(math.log2(num//10000+1)) if num >100000 else math.ceil(math.log2(num+1)) for num in max_nums]

    # 将字符串按逗号分割成一个列表
    data_list = feature.split(',')
    # 将每个元素转换为相应的数值类型（如果是浮动数，转换为 float；如果是整数，转换为 int）
    data_list = [int(float(item)) for item in data_list]
    for i in range(len(data_list)):
        if abs(data_list[i]) <= 100000:
            feature_blank.append(int_to_unsignçed_binary(data_list[i], lens_list[i]))
        else:
            feature_blank.append(int_to_unsignçed_binary(data_list[i]//10000, lens_list[i]))
    # print(feature_blank)
    # feature_blank = np.array([list(map(int, feature)) for feature in feature_blank])
    # print(feature_blank)
    return ''.join(feature_blank)