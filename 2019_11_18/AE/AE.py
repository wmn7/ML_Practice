import numpy as np
import pandas as pd

import os
import matplotlib.pyplot as plt
from datetime import date,datetime
import logging

import torch
from torch import nn
from torchvision import datasets, transforms
from torch import optim
from torch.autograd import Variable
from torchvision.utils import make_grid
from torchvision.utils import save_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

# 图像像素还原
def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)


# 定义网络
class DeepAutoEncoder(nn.Module):
    def __init__(self):
        super(DeepAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        ) # encoder可以将图片大小转换为 3*64*64 -> 256*1*1
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=4, bias=False), # 1->4
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, bias=False), # 4 -> 10
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, bias=False), # 10 -> 22
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, bias=False), # 22 -> 46
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=7, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=3, out_channels=3, kernel_size=7, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(in_channels=3, out_channels=3, kernel_size=7, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(True),
            
            # nn.Tanh()
        ) # decoder可以将图片大小转换为 256*1*1 -> 3*64*64
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

if __name__ == "__main__":
    # 将日志保存到文件
    logging.basicConfig(filename='logger.log',level=logging.INFO)
    # ---------
    # 加载数据集
    # ---------
    trans = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = datasets.ImageFolder('./data', transform=trans) # 数据路径

    dataloader = torch.utils.data.DataLoader(dataset,
                                        batch_size=128, # 批量大小
                                        shuffle=True, # 乱序
                                        num_workers=2 # 多进程
                                        )
    # ----------
    # 初始化网络
    # ----------
    AE = DeepAutoEncoder().to(device) # 定义分类器
    # ------------
    # 定义损失函数
    # ------------
    criterion = nn.L1Loss()
    # -----------------------
    # 定义损失函数和优化器
    # -----------------------
    learning_rate = 0.0002
    optimizer = torch.optim.Adam(AE.parameters(), lr=learning_rate)
    # ---------
    # 开始训练
    # ---------
    num_epochs = 200
    total_step = len(dataloader) # 依次epoch的步骤
    sample_dir = './results'
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9) # 学习率每20轮变为原来的70%
    T_images = torch.stack(([dataset[i][0] for i in range(16)]))
    save_image(denorm(T_images), os.path.join(sample_dir, 'real_images.png'), nrow=4) # 保存一下原始的测试图片
    # 开始训练
    for epoch in range(num_epochs):
        lr_scheduler.step()
        for i, (images, _) in enumerate(dataloader):
            batch_size = images.size(0)
            images = images.reshape(batch_size, 3, 64, 64).to(device)
            # ---------------------
            # 开始训练discriminator
            # ---------------------
            AE.train()
            # 首先计算真实的图片
            fake_image = AE(images) # 计算重构之后的内容
            loss = criterion(images, fake_image) # 计算loss
            optimizer.zero_grad() # 优化器梯度都要清0
            loss.backward() # 反向传播
            optimizer.step() # 进行优化

            # ---------
            # 打印结果
            # ---------
            if (i+2) % 20 == 0:
                t = datetime.now() #获取现在的时间
                logging.info('Time {}, Epoch [{}/{}], Step [{}/{}], loss:{:.4f}'
                            .format(t, epoch, num_epochs, i+1, total_step, loss.item()))
        # -------------------------------
        # 结果的保存(每一个epoch保存一次)
        # -------------------------------
        # 每一个epoch显示图片(这里切换为eval模式)
        AE.eval()
        test_images = AE(T_images.to(device))
        save_image(denorm(test_images), os.path.join(sample_dir, 'fake_images-{}.png'.format(epoch+1)), nrow=4)