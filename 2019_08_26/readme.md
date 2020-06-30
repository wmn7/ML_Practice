# 2019_08_26

## 分类器边界的可视化

![](https://github.com/wmn7/ML_Practice/blob/master/2019_08_26/Snipaste_2019-08-24_15-00-14.jpg)

## 预测结果评价

下面是一个简单的示例:

```python
outputs = linear(X_test.to(device))
_, predicted = torch.max(outputs.data, 1)

# 模型评估, 绘制混淆矩阵
display_model_performance_metrics(true_labels=Y_test.numpy(),
                predicted_labels=predicted.cpu().numpy(),
                classes=[0,1])
```

![](https://github.com/wmn7/ML_Practice/blob/master/2019_08_26/tool/snipaste_20200630_163843.png)
