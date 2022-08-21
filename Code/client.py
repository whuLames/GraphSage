import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from data import CoraData
from sampling import multihop_sampling

from collections import namedtuple
NUM_NEIGHBORS_LIST = [10, 10]   # 每阶采样邻居的节点数
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

Data = namedtuple('Data', ['x', 'y', 'adjacency_dict'])

data = CoraData().data
x = data.x / data.x.sum(1, keepdims=True)  # 归一化数据，使得每一行和为1

train_label = data.y


class ClientUpdate(object):
    def __init__(self, args, data_index):
        """
            Arguments:
                args {}: 本地训练参数，包括epoch、batch_size、lr(学习率)
                data_index []: 训练集的索引 list
                # test_index []: 测试集数据索引，用于测试每个client上运行的数据
        """
        self.args = args
        self.data_index = data_index
        # self.test_index = test_index

    def dividing(self):
        """
        将本client拥有的数据索引根据batch_size进行划分，最后不满一个batch_size的数据加入到最后一个batch之中

        Returns:
            all_batch_index: [[], []]  表示每一个batch中的数据索引
            batch_num: 表示分为的batch数目
        """
        batch_size = self.args["batch_size"]
        data_index = self.data_index  # 本client的数据索引
        batch_num = int(len(data_index) / batch_size)
        all_batch_index = []
        for i in range(batch_num - 1):
            batch_index = np.random.choice(data_index, batch_size, replace=False)
            data_index = list(set(data_index) - set(batch_index))
            all_batch_index.append(batch_index)

        all_batch_index.append(data_index)  # 剩下的直接放入最后一个batch
        return batch_num, all_batch_index

    # def test(self, model_client):
    #     model_client.eval()
    #     with torch.no_grad():
    #         """
    #             item 将 tensor变量转换为python的基本数据类型（int float）
    #             test_sampling_result = [[10], [100], [1000]]
    #         """
    #         test_index = self.test_index
    #         test_sampling_result = multihop_sampling(test_index, NUM_NEIGHBORS_LIST, data.adjacency_dict)
    #         test_x = [torch.from_numpy(x[idx]).float().to(DEVICE) for idx in test_sampling_result]
    #         test_logits = model_client(test_x)
    #         test_label = torch.from_numpy(data.y[test_index]).long().to(DEVICE)
    #         # print(test_logits.shape)
    #         predict_y = test_logits.max(1)[1]  # 取得最大值位置的下标
    #         accuarcy = torch.eq(predict_y, test_label).float().mean().item()  # 计算准确率
    #         print("Test Accuracy: ", accuarcy)

    def train(self, model_client, client):
        """
        Arguments:
            model_client: client的训练模型
            client: client编号
        Returns:
            Parameters:  训练之后模型参数

        """
        criterion = nn.CrossEntropyLoss().to(DEVICE)
        optimizer = optim.Adam(model_client.parameters(), lr=self.args['lr'], weight_decay=5e-4)
        model_client.train()
        epoch = self.args["epoch"]
        batch_size = self.args["batch_size"]
        for i in range(epoch):
            batch_num, all_batch_index = self.dividing()
            # dividing 函数暂时不用, 每次都在data_index 里随机挑选batch_size 的数据
            for j in range(batch_num):
                # batch_src_index = all_batch_index[j]
                # print(batch_src_index)
                # print(batch_size)
                # batch_src_index = np.random.choice(self.data_index, batch_size, replace=False)
                # batch_src_index = np.random.choice(train_index, size=(BTACH_SIZE,))
                # batch_src_index = np.random.choice(self.data_index, batch_size)
                batch_src_index = all_batch_index[j]
                batch_src_label = torch.from_numpy(train_label[batch_src_index]).long().to(DEVICE)
                batch_sampling_result = multihop_sampling(batch_src_index, NUM_NEIGHBORS_LIST, data.adjacency_dict)
                batch_sampling_x = [torch.from_numpy(x[idx]).float().to(DEVICE) for idx in batch_sampling_result]
                batch_train_logits = model_client(batch_sampling_x)
                loss = criterion(batch_train_logits, batch_src_label)  # 损失函数
                # print(batch_train_logits)
                # print(batch_src_label)
                optimizer.zero_grad()  # 梯度清空
                loss.backward()  # 反向传播计算参数的梯度
                optimizer.step()  # 使用优化方法进行梯度更新
                print("Client: {:03d} Epoch {:03d} Batch {:03d} Loss: {:.4f}".format(client,i + 1, j + 1, loss.item()))

        return model_client.state_dict()



