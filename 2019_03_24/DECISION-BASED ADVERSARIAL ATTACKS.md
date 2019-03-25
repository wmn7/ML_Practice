**DECISION-BASED ADVERSARIAL ATTACKS:
RELIABLE ATTACKS AGAINST BLACK-BOX MACHINE LEARNING MODELS**

---



#### Abstract

提出Decision-based攻击，优势为：(1)可用在现实中的黑盒应用；(2)需要更少信息，比transfer-based攻击更简单；(3)比gradient-和score-更鲁棒。尝试减少扰动的同时保持对抗性。只需要获得模型的决策结果（是不是对抗样本的判断）。在Clarifai分类器（一个网上应用）上攻击成功。foolbox实现：https://github.com/bethgelab/foolbox/blob/master/foolbox/attacks/boundary_attack.py



#### 1 Introduction

当前攻击方式分为三种：

- Gradient-based：基于模型的细节信息，loss的梯度。防御方法是mask the gradients，例如防御蒸馏（defensive distillation）或不可微分分类器。
- Score-based：对模型已知更少，依赖模型的预测得分（置信度/打标），使用预测数值来估计梯度。防御是添加随机元素阻碍梯度估计，在样本周围引入sharp-edged plateau，掩盖数值估计结果。
- Transfer-based：需要训练集的信息，用于训练一个可观察的替代模型来合成扰动，由于对抗样本可以在模型之间迁移，攻击可能达到100%。防御方法，在某替代模型的对抗样本的训练集上进行鲁棒训练。

新：Decision-based：基于模型预测的结果。需要更少的信息。可以抵抗防御蒸馏。



#### 2 Boundary Attack

初始化的对抗样本walk around the boundary，判断(1)是否一直处于对抗区，(2)是否缩短与目标图片的距离。使用proposal分布P逐步发现较小的扰动的对抗样本。算法为：

```
循环步数k，初始样本o(0)
	从分布P中随机获得一个扰动 ita
	如果 o(k-1)+ita 是对抗样本
		o(k) = o(k-1)+ita
	如果不是对抗样本
		o(k) = o(k-1)
	k = k+1
```

##### 2.1 初始化

初始对抗样本。无目标攻击中，从最大熵分布中产生样本，每个pixel从0-255的均匀分布产生。有目标攻击中，从任意被分为此目标中的样本开始。

##### 2.2 Proposal分布

三个公式：扰动样本的值约束。扰动的大小与一个值有关。扰动的减少依赖于扰动图像与原正常样本的距离。
从高斯分部中产生扰动，然后满足距离相等条件（称为正交分布），然后向原始图片靠近。

##### 2.3 对抗样本标准

flexible

##### 2.4 超参调整

调整公式中的步长和扰动值。通过边界的几何情况来调整。



---

edit using Typora