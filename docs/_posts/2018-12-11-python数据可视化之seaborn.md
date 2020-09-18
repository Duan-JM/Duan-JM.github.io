---
layout: post
title: "python数据可视化之 seaborn"
date: 2018-12-11
categories: "Python"
tags: ['Python','技巧']
---
## 简介
Seaborn 是一个数据可视化的库，主要用来生成热力图的，详情查看它的[官网](http://seaborn.pydata.org/tutorial.html)。这个工具一定要混合 `matplotlib` 来使用，我们在做好图之后还是必须要用 `plt.show` 才能展示图片，同时图片的布局也是靠 `matplotlib`。

## 热力图
热力图很多人其实是第一次接触，我在查了百度之后的百度百科结果不太满意，转而投奔[ Wiki 结果](https://en.wikipedia.org/wiki/Heat_map)如下：
> A heat map (or heatmap) is a graphical representation of data where the individual values contained in a matrix are represented as colors. "Heat map" is a newer term but shading matrices have existed for over a century.

除此之外我们还可以看到一些重要的描述如下：
> Heat maps originated in 2D displays of the values in a data matrix. Larger values were represented by small dark gray or black squares (pixels) and smaller values by lighter squares. 

我们可以看到它其实是用深浅来表示大小的。调用 seaborn 的代码实现如下：

```python
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

np.random.seed(0)
sns.set()
uniform_data = np.random.rand(10, 12)
ax = sns.heatmap(uniform_data)
plt.show()
```

![](https://ws1.sinaimg.cn/large/006tNbRwly1fy2vqxl9mtj30zm0tqglu.jpg)
当然这仅仅是热图中的一小部分，它还有其他功能，可以参考[官网 heatmap 的章节](http://seaborn.pydata.org/generated/seaborn.heatmap.html?highlight=heatmap#seaborn.heatmap)。

## 密度图
通过这个图我们能看出数据的密度，能很直观的看出输出的值最集中的区域。简单的代码（二维密度图）如下：

```python
import numpy as np; np.random.seed(10)
import seaborn as sns; sns.set(color_codes=True)
import matplotlib.pyplot as plt
mean, cov = [0, 2], [(1, .5), (.5, 1)]
x, y = np.random.multivariate_normal(mean, cov, size=50).T
ax = sns.kdeplot(x, y, shade=True)
plt.show()
```

![](https://ws4.sinaimg.cn/large/006tNbRwly1fy2vqz5xrkj30zi0ty0t1.jpg)
同理，其他细节可以参考[官网 kdeplot 章节](http://seaborn.pydata.org/generated/seaborn.kdeplot.html?highlight=kdeplot#seaborn.kdeplot)。

## 结尾
对于可视化的工具，个人比较喜欢插它官网的 Gallery，比如 Echart 的 Gallery，在 Gallery 中我们可以很好的把握图标的样式，还能得到样例代码，不失为一件美事。

## 参考
- [4种更快更简单实现 Python 数据可视化的方法](https://mp.weixin.qq.com/s/M7wC0XhDtenvTA_y1jfSjQ "机器之心")
- [seaborn 官网 Gallery](http://seaborn.pydata.org/examples/index.html)