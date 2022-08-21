import numpy as np
import random
"""描述
    dividing 完成每个client的数据划分任务
    暂定默认为共 10 个 client
    Cora数据集共2708个节点，划分20%作为测试集，划分80%为训练集，将训练集随机划分到 10 个client上
    测试集: 548
    训练集: 2160
"""


def nums_client_dividing():
    """
    将训练集的2160数据划分到10个client上
    先默认每个client上放100个数据，剩下的数据再随机分配

    Returns：
        nums_client[] 存储每个clients上的数据量
    """
    random_index = [0]  # 在区间随机插入9个数，将数据区间分为10份，存储随机生成的9个数
    for i in range(9):
        temp_index = random.randint(0, 1159)
        random_index.append(temp_index)
    random_index.append(1160)
    random_index.sort(reverse=False)  # 升序排列

    nums_client = []
    for i in range(10):
        nums = random_index[i + 1] - random_index[i]
        nums_client.append(nums + 100)
        print("第{} 个client随机分配数据量为{}".format(i + 1, nums + 100))

    return nums_client


def test_dividing():
    """测试集划分

    Returns：
        test_index[] 测试集索引

    """
    nums = 548  # 测试集数量为548
    all_index = [x for x in range(2708)]
    test_index = np.random.choice(all_index, nums, replace=False)

    return test_index


def client_train_dividing():
    """训练集数据划分到每一个client上

    Arguments:


    Returns:
        dict_clients{}: 存储每一个clients上的训练数据索引
        test_index[]: 测试数据集索引
    """
    test_index = test_dividing()  # 获取测试集数据索引
    nums_client = nums_client_dividing()  # 获取每个client的数据量
    all_index = [x for x in range(2708)]
    all_index = list(set(all_index) - set(test_index))  # 去掉测试集

    dict_clients = {}
    for i in range(10):
        dict_clients[i] = np.random.choice(all_index, nums_client[i], replace=False)  # 挑选出指定数量数据
        all_index = list(set(all_index) - set(dict_clients[i]))  # 将挑选出的数据索引从总数据中剔除

    return dict_clients, test_index



