---
layout: post
title: "Tensorflow 进阶之 Estimator"
date: 2018-12-10
categories: Tensorflow
tags: ["TensorFlow", "Estimator"]
---
之前的入门部分的 Estimator介绍了如何使用预训练模型，对整体有了一个直观的感受感受。在这部分中着重讲解如何创建自定义 Estimator。

## Estimator 模型的简单说明
所有的 Estimator 的模型的基类为 `tf.estimator.Estimator` ，这意味着即便是预设置的模型其实也是用自定义模型的方式设置的。和之前介绍的使用预创建的 Estimator 的唯一区别在于，我们需要自行编写模型函数（`model_fn`）
<!--more-->

## model\_fn
### 输入格式
虽然是自定义模型我们还是要遵循一定的输入和输出的格式的，输入格式简单介绍如下：
```python
def my_model_fn(
   features, # This is batch_features from input_fn
   labels,   # This is batch_labels from input_fn
   mode,     # An instance of tf.estimator.ModeKeys
   params):  # Additional configuration
```
其中需要注意的是 `params` 是**字典的格式**。

### 模型具备的内容
简单的来看，我们训练模型肯定需要三个操作，即训练，评估和预测。那么理所当然，这个模型中必须具有这三个操作。除此之外，在模型的书写方面也有一定的规范如下：
1. **定义模型结构**
	这一步和正常构建模型结构是相同的。通过从 `input_fn` 传入的 `features` 和 `labels` 进行构建模型就好了
2. 预测、训练和评估
	这一步简单的来说，就是使用通过对 `mode` 进行一个简单的条件判断就可以将三者区分开来，同时对于每一个 `mode` 值都需要返回一个 `tf.estimator.EstimatorSpec` 的一个实例，其中包含调用程序所需的信息。
	```python
	if mode == tf.estimator.ModeKeys.PREDICT:
		# your predict code for mode.predict()
		predictions = {
	    	'class_ids': predicted_classes[:, tf.newaxis],
	    	'probabilities': tf.nn.softmax(logits),
	    	'logits': logits,
		}
		return tf.estimator.EstimatorSpec(mode, predictions=predictions)

	if mode == tf.estimator.ModeKeys.EVAL:
		# your code for mode.evaluate()
		# compute loss and accurarcy
		loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
		accuracy = tf.metrics.accuracy(labels=labels,
	                        	   	   predictions=predicted_classes,
	                        	       name='acc_op')
		# set metrics
		metrics = {'accuracy': accuracy}
		return tf.estimator.EstimatorSpec(
	    	mode, loss=loss, eval_metric_ops=metrics)

	if mode == tf.estimator.ModeKeys.TRAIN:
		# your code for mode.train()
		optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
		train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
		return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
	```
	*注：需要注意的其实是 `EstimatorSpec` 中的三者的参数有略微的不同。*

## 自定义模型的使用
和之前使用封装好的模型的步骤相同，只是在实例化 Estimator 的时候需要传入自定的模型。
```python
classifier = tf.estimator.Estimator(
        model_fn=my_model,
        params={
            'feature_columns': my_feature_columns,
            'your params1': your params1,
            'your params2': 2,
        })
```
## 检查点的使用
### 使用检查点保存模型
我们使用 GPU 进行训练的时间会很长，我们非常不希望因为一些意外原因比如“熊孩子踹掉电源”的时间导致模型完全重新训练。Estimator 有一种简单的方式可以实现实时存盘的功能。值得一提的是，检查点的功能其实是默认开启的，在我们实例化 `Estimator` 的时候，未指定 `model_dir` 的话，会自动存到 Python 的 `tempfile,mkdtmep` 函数选择的临时文件夹中。
这样看，我们直接指定 `model_dir` 就可以使用检查点了，所以检查点的使用代码，就是在实例化 `Estimator`  的时候加一个参数 `model_dir`。

```python
# sample from tensorflow tutorial
classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    hidden_units=[10, 10],
    n_classes=3,
    model_dir='models/iris')
```
模型默认有两个情况是写入检查点的：
- 每 10 分钟写入一个检查点
- 在 `train` 方法开始（第一次迭代）和完成（最后一次迭代）时写入一个检查点
*注：只在目录中保留 5 个最近写入的检查点*

显然我们可以通过 `RunConfig` 对象来定义所需要的存档频率，步骤如下：
1. 创建一个 `RunConfig` 对象来定义所需要的时间安排
2. 实例化的时候传入 `RunConfig`。

```python
my_checkpointing_config = tf.estimator.RunConfig(
    save_checkpoints_secs = 20*60,  # Save checkpoints every 20 minutes.
    keep_checkpoint_max = 10,       # Retain the 10 most recent checkpoints.
)

classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    hidden_units=[10, 10],
    n_classes=3,
    model_dir='models/iris',
    config=my_checkpointing_config)
```
*注：在 Keras 入门中提到的 GPU 并行操作，也是在这里设置的。*

### 检查点的恢复
简单的来说，我们并不需要额外设置检查点的回复，Estimator 在使用 `train`，`evaluate` 或 `predict` 方法的时候，都会以以下方式自动恢复检查点。
1. Estimator 通过运行 `model_fn()` 构建模型图。
2. Estimator 根据最近写入的检查点中存储的数据来初始化新模型的权重。
当然，这种简单的恢复就仅仅支持和当前模型匹配的检查点恢复，如果更改了模型之前的检查点就不能用了。

## 结尾
至此为止，我们学会了如何构建一个自定义的模型，以及如何在训练的时候使用 `Estimator` 保存模型和输入 Tensorboard 可以使用的 log 文件。
