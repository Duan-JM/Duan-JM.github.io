---
layout: post
title: "Tensorflow 多GPU并行计算报错"
date: 2018-12-07
categories: "TroubleShoot"
tags: ['Python','TroubleShoot','Tensorflow','多GPU']
---
## 问题分析
在 Keras 中配置多 GPU 并行计算的时候，报错 `libnccl.so.2 not found`。简单的来说就是 Tensorflow 多 GPU 运行的话使用了一个 NCCL 2.x 的库，但是不是默认安装的，同时又由于NCCL 2.x是动态加载的，因此不会影响不调用NCCL原语的程序，也就是说直到你第一次尝试多 GPU 为止都不会发现这个问题。
<!--more-->

## 解决
解决的方式当然是相当简单咯，安装就可以了，你可以选择从[官网](https://developer.nvidia.com/nccl)安装，也可以用如下操作来偷懒。
*注：也有可能直接 `sudo apt-get install libnccl2` 就可以直接安装。*
```bash
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub

wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.1.85-1_amd64.deb

wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64.deb
 
sudo dpkg -i cuda-repo-ubuntu1604_9.1.85-1_amd64.deb
 
sudo dpkg -i nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64.deb
 
sudo apt-get update
sudo apt-get install libnccl2
```
 成功安装后，再次跑程序试试，不出意外就会成功啦。最后祝大家 Happy Coding。
