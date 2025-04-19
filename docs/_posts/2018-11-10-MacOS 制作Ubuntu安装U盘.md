---
layout: post
title: "MacOS 制作Ubuntu安装U盘"
date: 2018-11-10
categories: "软件使用"
tags: ['Linux 安装','技巧', 'Mac']
---
# MacOS 制作Ubuntu安装U盘
一般而言我们在 windows 下使用 UtraISO 能很容易做出一个 linux 的安装U盘。但是在 MacOS 下应该怎么做呢？答案是万能的命令行！
<!--more-->

## 操作步骤
1. 使用 hdituil
	```bash
	hdiutil convert -format UDRW -o <output_filename> <your_isoFile>
	```
	我们使用 `hdituil` 对下载的 ISO 文件转换成 dmg 文件。其中 `-format UDRW`，代表着转换成具有读写权限的镜像。
2. 使用 diskutil 卸载U盘
	```bash
	diskutil list # 查看哪个是我们要卸载的U盘
	diskutil unmountDisk <TheDisk> 
	# The Disk 代表着我们要选的U盘 如 /dev/disk2
	```
3. 使用 `dd` 命令
	- `if` 输入的文件路径
	- `of` 输出的文件路径
	- `bs` 块大小，一般使用 `1m`
	```bash
	mv <your dmg> <iso> # 将之前的转换成的dmg文件重命名成iso文件
	sudo dd if=/your/iso/path of=/your/disk/path bs=1m
	# 注意这里的disk path需要前面加r，比如/dev/disk2 -> /dev/rdisk2
	```
4. 重要数据销毁
	```bash
	sudo dd if=/dev/urandom of=/your/disk/path
	```
	用随机数填充U盘，来彻底销毁数据。
