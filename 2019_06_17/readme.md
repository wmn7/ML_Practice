<!--
 * @Author: WANG Maonan
 * @Date: 2021-06-07 20:50:02
 * @Description: 关于 Seq2Seq 的一些实验
 * @LastEditTime: 2021-06-07 20:50:40
-->
# 2019_06_17

## Seq2Seq - translation

一个简单的Seq2Seq的模型，用来实现英文与法文的翻译。

Loss下降曲线:

![](https://github.com/wmn7/ML_Practice/blob/master/2019_06_17/pic/Snipaste_2019-06-07_17-17-35.jpg)

最终的效果展示:

![](https://github.com/wmn7/ML_Practice/blob/master/2019_06_17/pic/Snipaste_2019-06-07_17-17-26.jpg)

## Seq2Seq with Attention

这里使用 Attention 实现一个 Seq2Seq（英语与法语之间的翻译），简单介绍一下 Attention 的使用。