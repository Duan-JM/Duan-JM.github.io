---
layout: post
title: "Tensorflow 入门之Keras"
date: 2018-12-08
categories: Tensorflow
tags: ["TensorFlow", "Keras"]
---
## [Keras](https://keras.io) 官宣特征
1. 简单快速的圆形部署
2. 支持 CNN 和 RNN，也支持两者结合
3. 同时支持 CPU 和 GPU 计算。

## Keras 设计理念
1. **User friendliness**, Keras is an API designed for human beings, not machines. 
2. **Modularity**, a model is understood as a sequence or a graph of standalone, fully-configurable modules that can be plugged together with as few restrictions as possible.
3. **Easy extensibility**. New modules are simple to add (as new classes and functions), and existing modules provide ample examples. 

## Hello World
1. 导入 Keras
	```python
	import tensorflow as tf
	from tensorflow import keras
	```
2. 模型的导入
	在 Keras 中的模型被看作是一个 Sequential 模型，一个层组成的堆。
	```python
	from keras.models import Sequential
	from keras.layers import Dense
	model = Sequential() # 事例化 equential

	# 构建模型
	model.add(Dense(units=64, activation='relu', inpout_dim=100))
	model.add(Dense(units=10, activation='softmax'))

	# 用 .compile() 设计得分函数和学习方式
	model.compile(loss='categorical_crossentropy',
	        	  optimizer='sgd',
	        	  metrics=['accuracy'])
	model.compile(loss=keras.losses.categorical_crossentropy,
	        	  optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True))

	# train
	model.fit(x_train, y_train, epochs=5, batch_size=32)
	model.train_on_batch(x_batch, y_batch)

	# val
	loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)

	# 直接预测
	classes = model.predict(x_test, batch_size=128)
	```

## 序列模型
简单的来说，Keras 就像搭积木一样以组合层的方式来构成图。最常用的为序列模型 `tf.keras.Sequential`。
一般构建图的流程如下：
1. 事例化一个Sequential：`model = keras.Sequential()`
2. 往里面添加层：`model.add(<layers>)`

## 配置层
简单的层面来说我们可以使用很多预定制的 `keras.layers`。他们有一些统一的参数如下：
- `activation`：设置层的激活函数。此参数由内置函数的名称指定，或指定为可调用对象。
- `kernel_initializer` 和 `bias_initializer`：创建层权重（核和偏差）的初始化方案。此参数是一个名称或可调用对象，默认为 `"Glorot uniform"` 初始化器。
- `kernel_regularizer` 和 `bias_regularizer`：应用层权重（核和偏差）的正则化方案，例如 L1 或 L2 正则化。默认情况下，系统不会应用正则化函数。

## 训练流程
构建了模型之后，通过调用 `compile` 来配置模型的训练流程：
```python
model.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```
`tf.keras.Model.compule` 有是那个重要参数：
- `optimizer`：此对象会指定训练过程。从 `tf.train` 模块向它传递优化器实例如 `AdamOptimizer`、`RMSPropOptimizer` 或 `GradientDescentOptimizer`。
- `loss`：值得在优化期间你是用什么得分函数。常见的有 `mse`、`categorical_crossentropy` 和 `binary_crossentropy`。损失函数可以在 `tf.keras.losses`模块来找。
- `metrics`：用于监控训练。它在 `tf.keras.metrics`模块中找。

```python
# Configure a model for mean-squared error regression.
model.compile(optimizer=tf.train.AdamOptimizer(0.01),
              loss='mse',       # mean squared error
              metrics=['mae'])  # mean absolute error

# Configure a model for categorical classification.
model.compile(optimizer=tf.train.RMSPropOptimizer(0.01),
              loss=keras.losses.categorical_crossentropy,
              metrics=[keras.metrics.categorical_accuracy])
```

## 数据导入
1. NumPy 数据
	```python
	import numpy as np
	data = np.random.random((1000, 32))
	labels = np.random.random((1000, 10))

	model.fit(data, label, epochs=10, batch_size=32)
	```
	`tf.keras.Model.fit` 有三个重要训练参数：
	- `epochs`：以周期为单位进行训练，即一个周期有多少个 epoch。
	- `batch_size`：模型会将输入的数据切分成较小的 batch，并在训练时迭代这些 batch。
	- `validation_data`：传递此参数（输入和标签元组）可以让该模型在每个周期结束时以推理模式显示所传递数据的损失和指标。简单的来说就是**在每个周期结束的时候在验证集测试一遍**。

		```python
		import numpy as np

		data = np.random.random((1000, 32))
		labels = np.random.random((1000, 10))

		val_data = np.random.random((100, 32))
		val_labels = np.random.random((100, 10))

		model.fit(data, labels, epochs=10, batch_size=32,
				  validation_data=(val_data, val_labels))
		```

2. `tf.data` 数据集
	和 NumPy 数据传参数一样，但是不同的地方在于出现了一个 `steps_per_epoch`，这个参数的官方说明如下：
	> `steps_per_epoch`: Integer or None. Total number of steps (batches of samples) before declaring one epoch finished and starting the next epoch. When training with input tensors such as TensorFlow data tensors, the default None is equal to the number of samples in your dataset divided by the batch size, or 1 if that cannot be determined.

	简单的来说就是，dataset 指定一个 epoch 有多少 batch，不指定的话或者 1 的话，就是所有的 batch。同时由于 `Dateset` 会生成批次数据，所以不需要指定 `batch_size`。
	
	```python
	# Instantiates a toy dataset instance:
	dataset = tf.data.Dataset.from_tensor_slices((data, labels))
	dataset = dataset.batch(32)
	dataset = dataset.repeat()

	# Don't forget to specify `steps_per_epoch` when calling `fit` on a dataset.
	model.fit(dataset, epochs=10, steps_per_epoch=30)

	# 用于验证
	dataset = tf.data.Dataset.from_tensor_slices((data, labels))
	dataset = dataset.batch(32).repeat()

	val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_labels))
	val_dataset = val_dataset.batch(32).repeat()

	model.fit(dataset, epochs=10, steps_per_epoch=30,
	    	  validation_data=val_dataset,
	    	  validation_steps=3)
	```

## 评估和预测
评估是采用 `tf.keras.Model.evaluate` 和 `tf.keras.Model.predict` 方法，导入用来评估的数据依旧是 NumPy 和 `tf.data.Dataset` 都可以用。

```python
# 评估
model.evaluate(x, y, batch_size=32)
model.evaluate(dataset, steps=30)

# 预测
model.predict(x, batch_size=32)
model.predict(dataset, steps=30)
```

## 保存和恢复
### 仅限权重
使用 [`tf.keras.Model.save_weights`](https://tensorflow.google.cn/api_docs/python/tf/keras/Model#save_weights) 保存并加载模型的权重：

```python
# Save weights to a TensorFlow Checkpoint file
model.save_weights('./my_model')

# Restore the model's state,
# this requires a model with the same architecture.
model.load_weights('my_model')
```

### 仅限配置
可以保存模型的配置，此操作会对模型架构（不含任何权重）进行序列化。即使没有定义原始模型的代码，保存的配置也可以重新创建并初始化相同的模型。Keras 支持 JSON 和 YAML 序列化格式：

```python
# Serialize a model to JSON format
json_string = model.to_json()

# Recreate the model (freshly initialized)
fresh_model = keras.models.from_json(json_string)

# Serializes a model to YAML format
yaml_string = model.to_yaml()

# Recreate the model
fresh_model = keras.models.from_yaml(yaml_string)
```

*注：子类化模型不可序列化，因为框架由 call 方法正文中的 python 代码定义。*

### 整个模型

```python
# Create a trivial model
model = keras.Sequential([
  keras.layers.Dense(10, activation='softmax', input_shape=(32,)),
  keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(data, targets, batch_size=32, epochs=5)

# Save entire model to a HDF5 file
model.save('my_model.h5')

# Recreate the exact same model, including weights and optimizer.
model = keras.models.load_model('my_model.h5')
```

## 同 Estimator 联动
[Estimator](https://tensorflow.google.cn/guide/estimators) API 用于针对分布式环境训练模型。它适用于一些行业使用场景，例如用大型数据集进行分布式训练并导出模型以用于生产。可以通过 [`tf.keras.estimator.model_to_estimator`](https://tensorflow.google.cn/api_docs/python/tf/keras/estimator/model_to_estimator) 将该模型转换为[`tf.estimator.Estimator`](https://tensorflow.google.cn/api_docs/python/tf/estimator/Estimator) 。
```python
model = keras.Sequential([layers.Dense(10,activation='softmax'),
                          layers.Dense(10,activation='softmax')])

model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

estimator = keras.estimator.model_to_estimator(model)
```

## 多GPU支持
`tf.keras` 模型可以使用 `tf.contrib.distribute.DistributionStrategy` 在多个 GPU 上运行。此 API 在多个 GPU 上提供分布式训练，几乎不需要更改现有代码。目前，`tf.contrib.distribute.MirroredStrategy` 是唯一受支持的分布策略。`MirroredStrategy` 通过在一台机器上使用规约在同步训练中进行图内复制。

```python
# 定义模型
model = keras.Sequential()
model.add(keras.layers.Dense(16, activation='relu', input_shape=(10,)))
model.add(keras.layers.Dense(1, activation='sigmoid'))

optimizer = tf.train.GradientDescentOptimizer(0.2)

model.compile(loss='binary_crossentropy', optimizer=optimizer)
model.summary()

# 定义输入
def input_fn():
  x = np.random.random((1024, 10))
  y = np.random.randint(2, size=(1024, 1))
  x = tf.cast(x, tf.float32)
  dataset = tf.data.Dataset.from_tensor_slices((x, y))
  dataset = dataset.repeat(10)
  dataset = dataset.batch(32)
  return dataset

# 定义策略
strategy = tf.contrib.distribute.MirroredStrategy()
config = tf.estimator.RunConfig(train_distribute=strategy)

# 转换成 estimator
keras_estimator = keras.estimator.model_to_estimator(
  keras_model=model,
  config=config,
  model_dir='/tmp/model_dir')

# 训练
keras_estimator.train(input_fn=input_fn, steps=10)
```
详细[参考这里](https://tensorflow.google.cn/guide/keras)的最后一章节。