"""
    服务器端操作
"""
import copy
import threading

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from net import GraphSage
from data import CoraData
from sampling import multihop_sampling
from dividing import client_train_dividing
from collections import namedtuple
from client import ClientUpdate
from queue import Queue
INPUT_DIM = 1433    # 输入维度
# Note: 采样的邻居阶数需要与GCN的层数保持一致
HIDDEN_DIM = [128, 7]   # 隐藏单元节点数
NUM_NEIGHBORS_LIST = [10, 10]   # 每阶采样邻居的节点数
assert len(HIDDEN_DIM) == len(NUM_NEIGHBORS_LIST)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

Data = namedtuple('Data', ['x', 'y', 'adjacency_dict'])

data = CoraData().data
x = data.x / data.x.sum(1, keepdims=True)  # 归一化数据，使得每一行和为1
# train_index = np.where(data.train_mask)[0]
# train_label = data.y
# test_index = np.where(data.test_mask)[0]
# 模型定义
model = GraphSage(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM,
                  num_neighbors_list=NUM_NEIGHBORS_LIST).to(DEVICE)
# print(model)



weight_global = model.state_dict()  # 存储全局参数


def client_train(args, client, dict_clients, weight_local_queue):
    """单个client上的模型训练

    Arguments：
        args： 模型训练需要的参数
        client： 客户端索引
        dict_clients: 各个client的数据索引
        weight_local_queue: 存储每个client的训练参数, 将本次训练参数放入其中
    """
    local = ClientUpdate(args, dict_clients[client])  # 传入args 和 训练数据即可
    print("{}号client本地训练中........".format(client))
    weight = local.train(copy.deepcopy(model).to(DEVICE), client)
    weight_local_queue.put(weight)


def parallel_train(args):
    """ 每个轮次随机选出 m 个client，m 个client之间实现并行训练
        Arguments:
            args {}: 存储训练参数，包括：
                fraction 、batch_size、 lr、epochs、rounds
        """
    dict_clients, test_index = client_train_dividing()
    accuracy_round = []
    for i in range(args['rounds']):
        print("第{}轮次通信....................................".format(i + 1))
        #  随机选择 m 个客户端
        m = max(int(args["fraction"] * 10), 1)
        clients = np.random.choice([index for index in range(10)], m, replace=False)
        num_client = []  # 每个client上的数据量
        weight_local_queue = Queue()  # 存储每个client上训练的参数
        weight_local = []  # 列表形式存储
        threads = []  # 存储每一个线程
        for client in clients:
            num_client.append(len(dict_clients[client]))

        for client in clients:
            thread = threading.Thread(target=client_train, args=[args, client, dict_clients, weight_local_queue])
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

        # 遍历所有参加训练的client，将数据放入weight_local
        for _ in range(len(clients)):
            weight_local.append(weight_local_queue.get())

        # weight_avg 存储加权平均之后的参数
        weight_avg = avg(weight_local, num_client)
        model.load_state_dict(weight_avg)
        # test(test_index)  # 测试模型
        accuracy_round.append(test(test_index))

    for i, accuracy in enumerate(accuracy_round):
        print("第{}轮训练后的模型准确率为{}".format(i, accuracy))


def avg(weight_locals, num_client):
    """
        每个client上的模型参数加权求和
    Arguments:
        weight_locals: 每个client训练后的模型参数
        num_client: 每个client上的数据量
    Returns:
        weight_avg: 聚合之后的模型参数
    """
    sum = 0
    weight_client = []  # 存储权重
    for i in num_client:
        sum += i

    for i in range(len(num_client)):
        weight_client.append(num_client[i] / sum)

    keys = weight_locals[0].keys()  # 获取keys
    weight_avg = weight_locals[0]
    for key in keys:
        weight_avg[key] = weight_avg[key] * weight_client[0]

    for key in keys:
        for i in range(1, len(weight_locals)):
            weight_avg[key] += weight_locals[i][key] * weight_client[i]

    return weight_avg


def test(test_index):
    with torch.no_grad():
        """
            item 将 tensor变量转换为python的基本数据类型（int float）
            test_sampling_result = [[10], [100], [1000]]

        """
        test_sampling_result = multihop_sampling(test_index, NUM_NEIGHBORS_LIST, data.adjacency_dict)
        test_x = [torch.from_numpy(x[idx]).float().to(DEVICE) for idx in test_sampling_result]
        test_logits = model(test_x)
        test_label = torch.from_numpy(data.y[test_index]).long().to(DEVICE)
        predict_y = test_logits.max(1)[1]  # 取得最大值位置的下标
        accuracy = torch.eq(predict_y, test_label).float().mean().item()  # 计算准确率
        print("Test Accuracy: ", accuracy)

        return accuracy


if __name__ == '__main__':
    args = {}
    args.update({"fraction": 0.4})
    args.update({"batch_size": 16})
    args.update({"lr": 0.01})
    args.update({"epoch": 10})
    args.update({"rounds": 10})
    parallel_train(args)



