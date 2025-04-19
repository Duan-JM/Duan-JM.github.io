---
layout: post
title: "Mac OS下 安装Scapy后出现的YCM报错"
date: 2018-09-18
categories: "TroubleShoot"
tags: ['Python','TroubleShoot','Scapy','VIM','YouCompleteMe']
---
## 起因
Mac OS 安装了 Scapy 后，在 YouCompleteMe 中出现如下报错：
```bash
X:ValueError: unknown locale: UTF-8 in Python
```

## 解决方案
添加如下代码到配置环境变量文件（zsh 或者 bash）
```bash
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
```

<!--more-->
