<!--
 * @Author: WANG Maonan
 * @Date: 2023-01-25 21:30:08
 * @Description: ViT 的介绍
 * @LastEditTime: 2023-01-27 21:08:42
-->
# ViT

该文件夹是对 `ViT` 的简单实现。
`ViT` 其实就是 `Transformer` 中的 `Encoder` 部分。
核心是如何将图片转换为序列输入模型。
作者在这里通过下面的方式来实现：

- Split image into patches，将每个图片分割为 `patch`；
- Vectorization, 将每个 patch 拉伸为向量；
- Position Embedding，最后加上「位置编码」即可；

该文件夹的代码参考自，[Github, mildlyoverfitted-vit](https://github.com/jankrepl/mildlyoverfitted/tree/master/github_adventures/vision_transformer)。他有一个视频的讲解，差不多是对每一行代码进行说明，[Vision Transformer in PyTorch
](https://www.youtube.com/watch?v=ovB0ddFtzzA)。