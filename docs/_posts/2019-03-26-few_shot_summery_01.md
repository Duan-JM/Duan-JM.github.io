---
layout: post
title: "Few-shot Learning 总结"
date: 2019-03-26
categories: FewShotLearning
tags: ["Few_Shot_Learning"]
---
## N ways K shot few-shot Learning 问题的描述
最终训练模型的效果需要达到，给模型之前完全没见过的  $N$ 个新类，每个新类中只有 $K$ 个样本。该模型需要能够通过利用这仅有的 $N \times K$  个样本，来对接下来给出的新样本进行分类。在 RelationNet work [^1] 的问题描述中，将这给出的 $N \times K$ 个样本集称为 Support Set ，待分类的图片集称为 Query Set。
## 常用的训练步骤
### 训练集中的类的样本不止 $K$ 个样本
若我们使用数据集 $D$ 来训练模型， 而 $D$  中所有的类中 $a$ 个样本，eg. mini-imagenet 中每个类有 600 个样本，则  $a=600$。整体的训练过程可以分为多个 meta-learning 的过程，在每个 meta-learning 开始的时候，从训练集 $D$ 中随机抽取 $N$ 个类，每个类中抽取 $K$ 个样本做成 Support Set，除此之外，还从已经抽取得到每个类中，除已抽取的样本外，再抽取 $T$ 个样本作为 Query Set。之后，模型将会去学习如何根据 Support Set 的样本，来正确分类 Query Set 的样本。

*注：整个 Meta-Learning 的训练过程，模拟的是当模型真正遇到小样本学习的过程。*

该种方法的模型有：
- Relation Network
- Matching Network
- Prototypical Network
- Siamese Network

### 用生成的方式补充小样本分类的方式
这种方式，主要分为两个组件，一方面是传统的 DCNN 分类器，如 ResNet，另一方面则是用于生成新的“假”样本的模型。其训练过程为，首先将 DCNN 在已有的大样本数据集上进行训练，得到一个在大样本数据集上表现良好的模型。之后，使用生成模型结合大样本数据集中类的样本和新类中的小样本，生成“假”的新类的图片，直到小样本的类中的样本数和大样本数据集中的类样本数目相同。最后，再使用之前训练的 DCNN 分类器在这些含有生成器生成的“假”样本的新类上进行训练，以达到小样本学习的目的。

该种方法的模型有：
- Low-shot Learning via Covariance-Preserving Adversarial Augmentation Networks
- Low-Shot Learning from Imaginary Data
- Low-shot Visual Recognition by Shrinking and Hallucinating Features

### 训练集中的类的样本只有 $K$ 个样本 
在这种方式中，在训练的时候模型就只能使用每个类只含有 $K$ 个样本的数据集。

该种方法的模型有：
- Optimization as a Model for Few-Shot Learning

## Relation Network
Relation Network 是 few-shot learning 中比较直观的模型。正规的来讲，他分为两部分，一部分是特征提取部分 `encoder`，另一部分是计算相似度的`relation network`。其中 `relation network`，部分就是通过两层全连接层学到输入的两个拼接后的样本的相似度。其模型图如下图所示：

![](/assets/images/blog/20190326fewshotsummery/CleanShot%202019-03-23%20at%2017.07.37.png)

*注：源码中，5 ways 5 shot 的训练是取得 $5$ 个 Support 样本和 $10$ 个 Query 样本。*
## Prototypical Network
这个模型简单的来说就是将图片 encoder 成向量之后，再将 Support Set 中的所有的样本求和取平均成一个向量后，再和 Query 的向量求欧式距离，以代表图片和类别的相似度。

![](/assets/images/blog/20190326fewshotsummery/2V5X5ZlMRhSxVt8Zc9QFpw_thumb_f36.jpg)
## Matching Network
Matching Network 则是用 Query 的样本和 Support 的样本做 Attention 操作，最终得到该图片和其他图片的相似度。
![](/assets/images/blog/20190326fewshotsummery/CleanShot%202019-03-26%20at%2009.04.11.png)

<script type="text/x-mathjax-config">MathJax.Hub.Config({tex2jax: {inlineMath:[['$','$']]}});</script>

<script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>

[^1]: Learning to Compare：Relation Network for Few-Shot Learning CVPR 2018