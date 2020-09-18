---
layout: post
title: "Tensorflow入门之Estimator"
date: 2018-12-08
categories: Tensorflow
tags: ["TensorFlow", "Estimator"]
---
## [Estimator](https://tensorflow.google.cn/guide/estimators) 作用简介
Estimator 使用来简化机器学习训练、评估、预测的一个高阶 TensorFlow API。我们可以使用预创建的 Estimator，也可以自己编写自定义的 Estimator，但是所有的 Estimator 都是基于 `tf.estimator.Estimator` 类的类。

## Why Estimator
1. 可以在本地主机上或分布式多服务器环境中运行基于 Estimator 的模型，而无需更改模型。
2. 使用高级直观代码开发先进的模型。简言之，采用 Estimator 创建模型通常比采用低阶 TensorFlow API 更简单。
3. Estimator 本身在 `tf.layers` 之上构建而成，可以简化自定义过程。
4. Estimator 会自动构建图。
5. Estimator 提供安全的分布式训练循环，可以控制如何以及何时：
	- 构建图
	- 初始化变量
	- 开始排队
	- 处理异常
	- 创建检查点文件并从故障中恢复
	- 保存 TensorBoard 的摘要
不过在使用 Estimator 编写应用时，我们必须将数据输入管道从模型中分离出来，而这种分离会方便我们以后在不同数据集上部署实验。

## 使用已经封装好的 Estimator 模型
1. 编写一个或多个数据导入函数。
	官方推荐我们可以创建两个函数来导入数据，一个用来导入训练集，一个用来导入测试集。函数输出要求如下两点：
	- 一个字典，其中 `key` 是特征名称，`value` 是包含相应特征数据的张量
	- 一个包含一个或多个标签的张量

	```bash
	def input_fn(dataset):
		... # manipulate dataset, extracting the feature dict and the label
		return feature_dict, label
	```
2. 定义特征列
	每个 `tf.feature_column` 都标识了特征名称、特征类型和任何输入预处理操作。
	```bash
	# Define three numeric feature columns.
	population = tf.feature_column.numeric_column('population')
	crime_rate = tf.feature_column.numeric_column('crime_rate')
	median_education = tf.feature_column.numeric_column('median_education',
	                	normalizer_fn=lambda x: x - global_education_mean)	
	```
	*注：第三个特征中定义了一个匿名函数，用来调节原始数据*
3. 实例化 Estimator
	```bash
	# Instantiate an estimator, passing the feature columns.
	estimator = tf.estimator.LinearClassifier(
		feature_columns=[population, crime_rate, median_education],
		)
	```
4. 使用模型进行训练、评估或推理方法
	```bash
	# my_training_set is the function created in Step 1
	estimator.train(input_fn=my_training_set, steps=2000)
	```

## 自定义 Estimator 模型
研究用 Tensorflow 大概率上不会有预定义好的 Estimator 所以我们还是需要了解如何写**模型函数**。由于我们需要自己写模型，所以我们可能需要自行实现包含但不仅仅包含多GPU并行计算等功能。

官方推荐的工作流如下：
1. 假设存在合适的预创建的 Estimator，使用它构建第一个模型并使用其结果确定基准。
2. 使用此预创建的 Estimator 构建和测试整体管道，包括数据的完整性和可靠性。
3. 如果存在其他合适的预创建的 Estimator，则运行实验来确定哪个预创建的 Estimator 效果最好。
4. 可以通过构建自定义 Estimator 进一步改进模型。
显然这一部分只有第四个符合我们的需求，当然我们也可以用 1-3 部分来检验读入的数据是否完整等。至于模型部分的构建我们可以通过[直接创建自定义模型](https://tensorflow.google.cn/guide/custom_estimators)和[使用 Keras 构建模型之后转换成 Estimator ](https://vdeamov.github.io/tensorflow/2018/12/08/Tensorflow%E5%85%A5%E9%97%A8%E4%B9%8BKeras/)这两种方式来构建。前者会在之后的进阶Estimator 笔记中总结，而后者已经在 Keras 入门阶段进行了介绍。

## 参考文献
- [Tensorflow官方教程 - Estimator](https://tensorflow.google.cn/guide/estimators)