# 2019_11_18

## 关于AutoEncoder及一些衍生的学习

### AutoEncoder

包含AE的简单例子. 使用Anime dataset作为例子.

原始的图片

![](https://github.com/wmn7/ML_Practice/blob/master/2019_11_18/AE/real_images.png)

第5轮之后, 重构的图片

![](https://github.com/wmn7/ML_Practice/blob/master/2019_11_18/AE/fake_images-5.png)

第200轮之后, 重构的图片

![](https://github.com/wmn7/ML_Practice/blob/master/2019_11_18/AE/fake_images-200.png)


## t-SNE可视化案例

我们首先使用AE对MNIST数据集进行降维, 到24维, 接着使用t-SNE降维到2维度, 并进行可视化. 

![](https://github.com/wmn7/ML_Practice/blob/master/2019_11_18/t-SNE/snipaste_20200716_174504.png)