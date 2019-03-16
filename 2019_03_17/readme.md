# 内容介绍

## Pytorch_CNN_CAT_DOG

使用Pytorch实现简单的CNN网络。

### 注意关于padding的计算

在pytorch中padding不能直接指定option, 如same, 我们可以根据下面公式进行计算.

![](https://github.com/wmn7/ML_Practice/blob/master/2019_03_17/img/snipaste_20190316_174412.png)