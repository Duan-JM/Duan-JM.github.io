---
layout: post
title: "Style Transfer Loss Summary"
date: 2019-08-12
categories: Losses
tags: [Losses, StyleTransfer]
---

# TransferLoss
## VGGLoss
VGGLoss 是提取 VGG 的不同的层学到的图片的特征，之后通过对比这些不同层的特征来计算两个图片的相似度，计算相似度的功能如下：

$$
l_{vgg}(x, y) = ||f_\phi(x) - f_\phi(y)||^{2}_2
$$
<!--more-->

## StyleLoss
StyleLoss 和 VGGLoss 相似，不同的是计算相似的时候是使用的 [Gram 矩阵](https://zh.wikipedia.org/wiki/%E6%A0%BC%E6%8B%89%E5%A7%86%E7%9F%A9%E9%98%B5)，直观上的差别就是，VGGLoss 是不同的层的结果之间的相减求平方，而 Style Loss 是两个特征要先乘自己的转置之后再做类似 VGGLoss 的操作。

$$
l_{style}(x, y) = ||G(f_\phi(x)) - G(f_\phi(y))||^{2}_2
$$

## SSIMLoss & MSELoss
常用的 MSELoss 针对色彩的敏感度比模糊度高，为了让图片生成更好的清晰的图片，可以用 SSIMLoss 来替代 MSE Loss。具体的公式和解释可以看[这个文章](https://zhuanlan.zhihu.com/p/67199699)。直接能用的源码[在这里](https://github.com/Po-Hsun-Su/pytorch-ssim.git)。

## VGGLoss 和 StyleLoss 的代码实现

```python
# 注意里面对每一层都加了权重，根据个人需要自行选择
import torch
import torch.nn as nn
from torchvision import models
import time

class Vgg19(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

class VGGLoss(nn.Module):
    def __init__(self, layids = None, device=torch.device("cpu")):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19()
        if torch.cuda.is_available():
            #self.vgg.cuda()
            self.vgg = self.vgg.to(device)
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.layids = layids

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        if self.layids is None:
            self.layids = list(range(len(x_vgg)))
        for i in self.layids:
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


class StyleLoss(nn.Module):
    def __init__(self, layids = None):
        super(StyleLoss, self).__init__()
        self.vgg = Vgg19()
        if torch.cuda.is_available():
            self.vgg.cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0, 1.0, 1.0, 1.0, 1.0]
        self.layids = layids

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = []
        if self.layids is None:
            self.layids = list(range(len(x_vgg)))
        for i in self.layids:
            x_gram = self.gram(x_vgg[i].detach())
            y_gram = self.gram(y_vgg[i].detach())
            loss.append(self.weights[i]*self.criterion(x_gram, y_gram).item())
        return loss

    def gram(self, x):
        """
        gram compute
        """
        b, c, h, w = x.size();
        x = x.view(b*c, -1);
        return torch.mm(x, x.t())
```
