# 内容介绍

## 使用Pytorch训练-验证-测试Resnet50

- 数据集：taobao商品数据（110类） -> 数据集见服务器
- 划分训练集和验证集，每epoch计算一次
- 每epoch保存一次模型

## 论文：Decision-based boundary attack

- 简略翻译见md。


## 可视化

- Visualizer for deep learning and machine learning models.
- 项目地址 : https://github.com/lutzroeder/netron

### 注意

模型保存时， 保存为onnx格式, 一些情况下, 使用.pth画出的图会出错.

```python
import torch.onnx
# 输入测试数据
input_data = Variable(torch.randn(1,1,64,64))
torch.onnx.export(resNet, input_data, "ResNet.onnx", export_params=True, verbose=True, training=False)
```

![](https://github.com/wmn7/ML_Practice/blob/master/2019_03_24/pic/ResNet_test.png)
