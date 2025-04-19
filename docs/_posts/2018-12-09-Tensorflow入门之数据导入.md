---
layout: post
title: "Tensorflow入门之数据导入"
date: 2018-12-09
categories: Tensorflow
tags: ["TensorFlow", "Dataset"]
---

## tf.data API 简介
借助这个 API 可以较为快速的入门数据导入的部分。自定义数据输入可以说是跑任何模型必须要会的部分。学习这部分 API 是入门 Tensorflow跳不过的部分。本部分和之前的 Tensorflow 部分一样，主要是筛选自官方教程，意在跳出自己认为核心的入门内容，抛去复杂的细节，以求快速入门。
<!--more-->

## tf.data 中的两个主要类
### tf.data.Dataset
`tf.data.Dataset` 表示一系列元素，其中每个元素包含一个或多个 Tensor 对象。在图像管道中，这个元素可能是单个训练样本，具有一对表示图像的数据和标签的张量。
- 创建**来源**：通过一个或多个 `tf.Tensor` 对象构成，可以使用 `Dataset.from_tensor_slices()` 来构建。
- **转换**：通过 `tf.data.Dataset` 对象构建数据集，如`Dataset.batch()`

### tf.data.Iterator
上面介绍了如何创建数据集，这里就提供了如何从数据集中提取元素的方法。`Iterator.get_next()` 返回的操作会在执行时生成 `Dataset` 的下一个元素，并且此操作通常充当输入管道代码和模型之间的接口。最简单的迭代器是"单次迭代器"，它与特定的 `Dataset` 相关联，并对其进行一次迭代。要实现更复杂的用途，您可以通过 `Iterator.initializer` 操作使用不同的数据集重新初始化和参数化迭代器，这样一来，这样就可以在同一个程序中对训练和验证数据进行多次迭代。

## 输入管道的流程
1. 构建 Dataset 并处理

	`tf.data.Dataset.from_tensors()` 或 `tf.data.Dataset.from_tensor_slices()`来构建 Dataset，当然如果以特定格式存储的数据，也有对应的读取方式，如 TFRecord 的为`tf.data.TFRecordDataset`。之后我们可以用 `map` 等函数进行对原始数据的二次加工，这部分可以查看 `tf.data.Dataset` 的文档。
2. 创建迭代器
	- `Iterator.initializer` 可以初始化迭代器的状态，可以达到一些复杂的操作。
	- `Iterator.get_next()` 取下一个对象。

## 数据集的结构
首先数据集中的**每个元素的结构需要是相同**的。一个元素可以包含一个或多个 `Tensor` 对象，这些对象为**组件**。每一个组件都有一个 `td.Dtype` 表示张量的类型，和一个 `tf.TensorShape` 来表示元素的形状。我们可以通过 `Dataset.output_types` 和 `Dataset.output_shapes` 来查看数据的类型和形状。
```python
dataset1 = tf.data.Dataset.from_tensor_slices(tf.random_uniform([4, 10]))
print(dataset1.output_types)  # ==> "tf.float32"
print(dataset1.output_shapes)  # ==> "(10,)"

dataset2 = tf.data.Dataset.from_tensor_slices(
   (tf.random_uniform([4]),
    tf.random_uniform([4, 100], maxval=100, dtype=tf.int32)))
print(dataset2.output_types)  # ==> "(tf.float32, tf.int32)"
print(dataset2.output_shapes)  # ==> "((), (100,))"

dataset3 = tf.data.Dataset.zip((dataset1, dataset2))
print(dataset3.output_types)  # ==> (tf.float32, (tf.float32, tf.int32))
print(dataset3.output_shapes)  # ==> "(10, ((), (100,)))"
```
*注：值得注意的是这部分可以看到，Dataset 中的每个样本，在 `output_shapes` 中被写成了列向量的形式。*

我们还能使用字典为每个组件命名，方法如下：
```python
dataset = tf.data.Dataset.from_tensor_slices(
   {"a": tf.random_uniform([4]),
    "b": tf.random_uniform([4, 100], maxval=100, dtype=tf.int32)})
print(dataset.output_types)  # ==> "{'a': tf.float32, 'b': tf.int32}"
print(dataset.output_shapes)  # ==> "{'a': (), 'b': (100,)}"
```

## 使用迭代器
我们有了数据的 `Dataset` 之后，下一步就是使用创建 `Iterator` 的方式来访问数据集的内容。
`Iterator` 有以下四种类型
- 单次：`dataset.make_one_shot_iterator()`，这种迭代器是不能初始化的，仅支持对数据集进行一次迭代，不需要显示初始化。单次迭代器可以处理基于队列的现有输入管道支持的几乎所有情况，但它们不支持参数化。
	```python
	dataset = tf.data.Dataset.range(100)
	iterator = dataset.make_one_shot_iterator()
	next_element = iterator.get_next()

	sess = tf.Session()
	for i in range(100):
	  value = sess.run(next_element)
	  assert i == value
	```
	*注：是唯一易于与 Estimator 搭配使用的类型。*

- 可初始化
	迭代器中含有 Tensor 参数的时候，即数据集中含有 `placeholder` 时，需要显式调用 `iterator.initializer` 操作才能使用该迭代器。
	```python
	max_value = tf.placeholder(tf.int64, shape=[])
	dataset = tf.data.Dataset.range(max_value)
	iterator = dataset.make_initializable_iterator()
	next_element = iterator.get_next()

	# Initialize an iterator over a dataset with 10 elements.
	sess.run(iterator.initializer, feed_dict={max_value: 10})
	for i in range(10):
	  value = sess.run(next_element)
	  assert i == value

	# Initialize the same iterator over a dataset with 100 elements.
	sess.run(iterator.initializer, feed_dict={max_value: 100})
	for i in range(100):
	  value = sess.run(next_element)
	  assert i == value
	```
- 可重新初始化
- 可馈送

### 消耗迭代器中的值
通过上一部分的例子我们就可以看出使用迭代器的值其实是这样一个过程：
1. 通过 `Iterator.get_next()` 的方法返回 `Tensor` 对象，但是在 run 之前是不会运行的。
2. 通过 `Session.run()` 来运行，这时候迭代器才会真正进入下一个状态。
3. 当迭代器到达数据集的尾部的时候，会生成 `tf.errors.OutofRangeError`，之后迭代器将处于不可用状态，余姚重新进行初始化。这一步一般来说会封装在 `try - except` 结构中。

### 保存迭代器状态
[`tf.contrib.data.make_saveable_from_iterator`](https://tensorflow.google.cn/api_docs/python/tf/contrib/data/make_saveable_from_iterator) 函数通过迭代器创建一个 `SaveableObject`，该对象可用于保存和恢复迭代器（实际上是整个输入管道）的当前状态。这样创建的可保存对象可以添加到 [`tf.train.Saver`](https://tensorflow.google.cn/api_docs/python/tf/train/Saver) 变量列表或 [`tf.GraphKeys.SAVEABLE_OBJECTS`](https://tensorflow.google.cn/api_docs/python/tf/GraphKeys#SAVEABLE_OBJECTS)集合中，以便采用与 [`tf.Variable`](https://tensorflow.google.cn/api_docs/python/tf/Variable) 相同的方式进行保存和恢复。请参阅[保存和恢复](https://tensorflow.google.cn/guide/saved_model)，详细了解如何保存和恢复变量。

```python
# Create saveable object from iterator.
saveable = tf.contrib.data.make_saveable_from_iterator(iterator)

# Save the iterator state by adding it to the saveable objects collection.
tf.add_to_collection(tf.GraphKeys.SAVEABLE_OBJECTS, saveable)
saver = tf.train.Saver()

with tf.Session() as sess:

  if should_checkpoint:
    saver.save(path_to_checkpoint)

# Restore the iterator state.
with tf.Session() as sess:
  saver.restore(sess, path_to_checkpoint)
```

## 结语
这部分，主要摘抄自官方教程[导入数据部分](https://tensorflow.google.cn/guide/datasets)，不过由于教程比较全面，所以挑出了最重要的内容，做为自己的入门记忆部分。Dataset 与 Estimator，TFRecord 等文件格式的读取，做为接下来进阶部分的教程来介绍。简单的来说通过这部分的学习主要理解了，迭代器和数据集的工作流程。为接下来学习更加实际细节的操作打下基础。
