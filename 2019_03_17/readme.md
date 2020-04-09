# 内容介绍

## PyTorch中维度的理解.

对于Pytorch中dim的理解, 特别是`torch.sum(x, dim=1)`时, 这里的dim=1与x.shape不对应情况的理解.

## PyTorch中交叉熵的理解.

介绍一下在PyTorch中CrossEntropy的背后的计算公式. 给出详细的推导.

## PyTorch实现简单分类

使用Pytorch实现简单的全连接网络。

## Pytorch_CNN_CAT_DOG

使用Pytorch实现简单的CNN网络, 最终实现cat dog的分类

### 注意关于padding的计算

在pytorch中padding不能直接指定option, 如same, 我们可以根据下面公式进行计算.

![](https://github.com/wmn7/ML_Practice/blob/master/2019_03_17/img/snipaste_20190316_174412.png)