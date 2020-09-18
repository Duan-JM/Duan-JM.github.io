---
layout: post
title: "Tensorflow进阶之数据导入"
date: 2018-12-16
categories: Tensorflow
tags: ["TensorFlow", "Dataset"]
---
## 不同格式的数据的导入
### Numpy 数据的导入
这种导入非常直白，就是使用 Numpy 把外部的数据进行导入，然后转换成 `tf.Tensor` ，之后使用 `Dataset.from_tensor_slices()`。就可以成功导入了。简单的案例如下：

```java
# Load the training data into two NumPy arrays, for example using `np.load()`.
with np.load("/var/data/training_data.npy") as data:
  features = data["features"]
  labels = data["labels"]

# Assume that each row of `features` corresponds to the same row as `labels`.
assert features.shape[0] == labels.shape[0]

dataset = tf.data.Dataset.from_tensor_slices((features, labels))
```
上面的简单的实例有一个很大的问题，就是 `features` 和 `labels` 会作为 `tf.constant()` 指令嵌入在 Tensorflow 的图中，会浪费很多内存。所以我们可以根据 `tf.palceholder()` 来定义 `Dataset`，同时在对数据集初始化的时候送入 Numpy 数组。
```java
with np.load("/var/data/training_data.npy") as data:
  features = data["features"]
  labels = data["labels"]

# Assume that each row of `features` corresponds to the same row as `labels`.
assert features.shape[0] == labels.shape[0]

features_placeholder = tf.placeholder(features.dtype, features.shape)
labels_placeholder = tf.placeholder(labels.dtype, labels.shape)

dataset = tf.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))
# [Other transformations on `dataset`...]
dataset = ...
iterator = dataset.make_initializable_iterator()

sess.run(iterator.initializer, feed_dict={features_placeholder: features,
                                          labels_placeholder: labels})
```
### TFRecord 数据的导入
TFRecord 是一种面向记录的简单二进制格式，很多 Tensorflow 应用采用这种方式来训练数据。这个也是推荐的做法。将它做成 Dataset 的方式也非常简单，就是单纯的通过 `tf.data.TFRecordDataset` 类就可以实现。
```java
# Creates a dataset that reads all of the examples from two files.
filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]
dataset = tf.data.TFRecordDataset(filenames)
```
同样我们也能设定成，在初始化迭代器的时候导入数据。其中需要注意的是 `filenames` 需要设置成 `tf.String` 类。
```java
filenames = tf.placeholder(tf.string, shape=[None])
dataset = tf.data.TFRecordDataset(filenames)
dataset = dataset.map(...)  # Parse the record into tensors.
dataset = dataset.repeat()  # Repeat the input indefinitely.
dataset = dataset.batch(32)
iterator = dataset.make_initializable_iterator()

# You can feed the initializer with the appropriate filenames for the current
# phase of execution, e.g. training vs. validation.

# Initialize `iterator` with training data.
training_filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]
sess.run(iterator.initializer, feed_dict={filenames: training_filenames})

# Initialize `iterator` with validation data.
validation_filenames = ["/var/data/validation1.tfrecord", ...]
sess.run(iterator.initializer, feed_dict={filenames: validation_filenames})
```
## Dataset 的预处理
### Dataset.map()
`Dataset.map(f)` 转换通过将指定函数 `f` 应用于输入数据集的每个元素来生成新数据集。
简单的实例（解码图片数据并调整大小）如下：
```java
# Reads an image from a file, decodes it into a dense tensor, and resizes it
# to a fixed shape.
def _parse_function(filename, label):
  image_string = tf.read_file(filename)
  image_decoded = tf.image.decode_jpeg(image_string)
  image_resized = tf.image.resize_images(image_decoded, [28, 28])
  return image_resized, label

# A vector of filenames.
filenames = tf.constant(["/var/data/image1.jpg", "/var/data/image2.jpg", ...])

# `labels[i]` is the label for the image in `filenames[i].
labels = tf.constant([0, 37, ...])

dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
dataset = dataset.map(_parse_function)
```
至此为止，我们对图片的处理还是使用的是 TensorFlow 中的 API，那么我们想用 Python 自带的奇奇怪怪的包应该怎么做呢。TensorFlow 给了我们 `tf.py_func()` 这个选项来使用任意 Python 逻辑。我们只用在 `Dataset.map()` 中调用 `tf.py_func()` 指令就可以了。简单的例子如下：
```java
import cv2

# Use a custom OpenCV function to read the image, instead of the standard
# TensorFlow `tf.read_file()` operation.
def _read_py_function(filename, label):
  image_decoded = cv2.imread(filename.decode(), cv2.IMREAD_GRAYSCALE)
  return image_decoded, label

# Use standard TensorFlow operations to resize the image to a fixed shape.
def _resize_function(image_decoded, label):
  image_decoded.set_shape([None, None, None])
  image_resized = tf.image.resize_images(image_decoded, [28, 28])
  return image_resized, label

filenames = ["/var/data/image1.jpg", "/var/data/image2.jpg", ...]
labels = [0, 37, 29, 1, ...]

dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
dataset = dataset.map(
    lambda filename, label: tuple(tf.py_func(
        _read_py_function, [filename, label], [tf.uint8, label.dtype])))
dataset = dataset.map(_resize_function)
```
### 批处理数据
1. 简单的批处理
	简单的批处理我们直接调用 `Dataset.batch()` 这种 API 即可，但是它有一个限制就是对于每个组件 i，所有元素的**张量形状都必须完全相同**。
	```
	inc_dataset = tf.data.Dataset.range(100)
	dec_dataset = tf.data.Dataset.range(0, -100, -1)
	dataset = tf.data.Dataset.zip((inc_dataset, dec_dataset))
	batched_dataset = dataset.batch(4)

	iterator = batched_dataset.make_one_shot_iterator()
	next_element = iterator.get_next()

	print(sess.run(next_element))  # ==> ([0, 1, 2,   3],   [ 0, -1,  -2,  -3])
	print(sess.run(next_element))  # ==> ([4, 5, 6,   7],   [-4, -5,  -6,  -7])
	print(sess.run(next_element))  # ==> ([8, 9, 10, 11],   [-8, -9, -10, -11])
	```
2. 填充批处理张量
	和简单批处理相比，这种方式可以对具有不同大小的张量进行批处理。这种方法的 API 为 `Dataset.padded_batch()`。简单的实例展示如下：
	```python
	dataset = tf.data.Dataset.range(100)
	dataset = dataset.map(lambda x: tf.fill([tf.cast(x, tf.int32)], x))
	dataset = dataset.padded_batch(4, padded_shapes=[None])

	iterator = dataset.make_one_shot_iterator()
	next_element = iterator.get_next()

	print(sess.run(next_element))  # ==> [[0, 0, 0], [1, 0, 0], [2, 2, 0], [3, 3, 3]]
	print(sess.run(next_element))  # ==> [[4, 4, 4, 4, 0, 0, 0],
	                        	   #      [5, 5, 5, 5, 5, 0, 0],
	                        	   #      [6, 6, 6, 6, 6, 6, 0],
	                        	   #      [7, 7, 7, 7, 7, 7, 7]]
	```
	可以通过 `Dataset.padded_batch()` 转换为每个组件的每个维度设置不同的填充，并且可以采用可变长度（在上面的示例中用 `None` 表示）或恒定长度。也可以替换填充值，默认设置为 0。

## 训练工作流程
### 处理多个周期
有时候我们希望我们的数据集能训练很多个周期，简单的方法是使用`Dataset.repeat()` API。
```python
filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]
dataset = tf.data.TFRecordDataset(filenames)
dataset = dataset.map(...)
dataset = dataset.repeat(10)
dataset = dataset.batch(32)
```
上述例子中，我们将 dataset 重复了 10 个周期，值得注意的是如果 repeat 中没有参数代表中无限次地重复使用，即不会在一个周期结束和下一个周期开始时发出信号。

如果我们想在每个周期结束时收到信号，则可以编写在数据集结束时捕获 [`tf.errors.OutOfRangeError`](https://tensorflow.google.cn/api_docs/python/tf/errors/OutOfRangeError?hl=zh-cn) 的训练循环。此时，就可以收集关于该周期的一些统计信息。
```python
filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]
dataset = tf.data.TFRecordDataset(filenames)
dataset = dataset.map(...)
dataset = dataset.batch(32)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

# Compute for 100 epochs.
for _ in range(100):
    sess.run(iterator.initializer)
    while True:
        try:
            sess.run(next_element)
        except tf.errors.OutOfRangeError:
            break

    # [Perform end-of-epoch calculations here.]
```
### 随机重排数据
有时候我们希望能随机的选取 Dataset 中的元素，则可以使用 `Dataset.shuffle()`。
```python
filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]
dataset = tf.data.TFRecordDataset(filenames)
dataset = dataset.map(...)
dataset = dataset.shuffle(buffer_size=10000)
dataset = dataset.batch(32)
dataset = dataset.repeat()
```
## 参考
- [[官方教程]](https://tensorflow.google.cn/guide/datasets?hl=zh-cn)