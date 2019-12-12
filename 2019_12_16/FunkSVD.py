import numpy as np
import pandas as pd

import os
import matplotlib.pyplot as plt
from datetime import date,datetime
import logging

import torch
from torch import nn
from torch import optim
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

# --------------------
# 下面三种计算loss的方式
# --------------------

def PredictR(P, Q, R):
    """
    最基本的FunkSVD
    """
    R_pred = torch.mm(P, Q.t()) # 矩阵相乘
    N,M = R.shape
    loss = 0
    for i in range(N):
        for j in range(M):
            if R[i][j] > 0:
                loss = loss + (R_pred[i][j]-R[i][j])**2
    return loss

def PredictRegularizationR(P, Q, R):
    """
    FunkSVD+Regularization
    """
    B = 0.02 # 正则话的系数
    R_pred = torch.mm(P, Q.t()) # 矩阵相乘
    N,M = R.shape
    loss = 0
    for i in range(N):
        for j in range(M):
            if R[i][j] > 0:
                loss = loss + (R_pred[i][j]-R[i][j])**2
    
    loss = loss + B*((P**2).sum() + (Q**2).sum()) # 加上正则项
    return loss

def PredictRegularizationConstrainR(P, Q, R):
    """
    FunkSVD+Regularization+矩阵R的约束(取值只能是0-5, P,Q>0)
    """

    B = 0.1 # 正则话的系数
    R_pred = torch.mm(P, Q.t()) # 矩阵相乘
    N,M = R.shape
    loss = 0
    for i in range(N):
        for j in range(M):
            if R[i][j] > 0:
                loss = loss + (R_pred[i][j]-R[i][j])**2
            elif R[i][j] == 0: # 下面是限定R的范围
                if R_pred[i][j] > 5:
                    loss = loss + (R_pred[i][j]-5)**2
                elif R_pred[i][j] < 0:
                    loss = loss + (R_pred[i][j]-0)**2
    
    loss = loss + B*((P**2).sum() + (Q**2).sum()) # 加上正则项

    # 限定P和Q的范围
    loss = loss + ((P[P<0]**2).sum() + (Q[Q<0]**2).sum())
    return loss

if __name__ == "__main__":
    # 原始矩阵R
    R = np.array([[5.0, 3.0, 0.0, 1.0],
    [4.0, 0.0 ,0.0 ,1.0],
    [1.0, 1.0, 0.0, 5.0],
    [1.0, 0.0, 0.0, 4.0],
    [0.0, 1.0, 5.0, 4.0]])
    
    N,M = R.shape
    K = 2
    
    R = torch.from_numpy(R).float()

    # 初始化矩阵P和Q
    P = Variable(torch.randn(N, K), requires_grad=True)
    Q = Variable(torch.randn(M, K), requires_grad=True)

    # -----------
    # 定义优化器
    # -----------
    learning_rate = 0.02
    optimizer = torch.optim.Adam([P,Q], lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9) # 学习率每20轮变为原来的90%
    # ---------
    # 开始训练
    # ---------
    num_epochs = 5000
    for epoch in range(num_epochs):
        lr_scheduler.step()
        # 计算Loss
        loss = PredictRegularizationConstrainR(P, Q, R)
        # 反向传播, 优化矩阵P和Q
        optimizer.zero_grad() # 优化器梯度都要清0
        loss.backward() # 反向传播
        optimizer.step() # 进行优化
        if epoch % 20 ==0:
            print(epoch,loss)
    # 求出最终的矩阵P和Q, 与P*Q
    R_pred = torch.mm(P, Q.t())
    print(R_pred)
    print('-'*10)
    print(P)
    print('-'*10)
    print(Q)