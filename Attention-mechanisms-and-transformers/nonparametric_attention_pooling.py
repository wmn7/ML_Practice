'''
@Author: WANG Maonan
@Date: 2022-09-26 08:01:51
@Description: 非参数注意力池化的介绍
@LastEditTime: 2022-09-26 11:05:17
'''
import torch
from torch.nn import functional as F
from matplotlib import pyplot as plt

from lib.d2l_torch import DataModule, plot

def plot_kernel_reg(data, y_hat):
    """绘制图像, 将原始数据集和预测的绘制在一起

    Args:
        y_hat (_type_): 预测得到的 y 值
    """
    plot(
        data.x_val, 
        [data.y_val, y_hat.detach().numpy()], 'x', 'y', legend=['Truth', 'Pred'],
        xlim=[0, 5], ylim=[-1, 5]
    )
    plt.plot(data.x_train, data.y_train, 'o', alpha=0.5)
    plt.savefig('./data_des.svg')

    
class NonlinearData(DataModule):
    """生成数据集,
     y = 2*sin(x) + x^0.8 + epsilon
    """
    def __init__(self, n, batch_size):
        self.save_hyperparameters()
        f = lambda x: 2 * torch.sin(x) + x**0.8
        self.x_train, _ = torch.sort(torch.rand(n) * 5)
        self.y_train = f(self.x_train) + torch.randn(n) # 用于训练的 label
        self.x_val = torch.arange(0, 5, 5.0/n)
        self.y_val = f(self.x_val) # 用于测试的 label

    def get_dataloader(self, train):
        arrays = (self.x_train, self.y_train) if train else (self.x_val, self.y_val)
        return self.get_tensorloader(arrays, train)


# #############
# 非参数的注意力汇聚
# #############
def diff(queries, keys):
    """计算 query 和 key 的距离
    """
    return queries.reshape((-1, 1)) - keys.reshape((1, -1))

def attention_pool(query_key_diffs, values):
    attention_weights = F.softmax(- query_key_diffs**2 / 2, dim=1) # 计算权重
    return torch.matmul(attention_weights, values), attention_weights # 权重和 value 相乘



if __name__ == '__main__':
    n = 50
    data = NonlinearData(n, batch_size=10)

    # 使用平均值
    y_hat = data.y_train.mean().repeat(n)
    # 非参数注意力
    y_hat, attention_weights = attention_pool(
        query_key_diffs = diff(data.x_val, data.x_train), 
        values = data.y_train
    )

    plot_kernel_reg(data, y_hat)