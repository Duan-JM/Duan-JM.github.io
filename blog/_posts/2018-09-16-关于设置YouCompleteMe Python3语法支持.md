---
layout: post
title: "关于设置YouCompleteMe Python3语法支持"
date: 2018-09-16
categories: "TroubleShoot"
tags: ['YouCompleteMe','Vim','TroubleShoot']
---
## 问题描述
YouCompleteMe 只能支持 Python2 的补全，不支持 Python3 库的补全。
## 解决
1. 重新去 YouCompleteMe 用 git pull 一下。
2. 使用 pip 安装 jedi
	```bash
	pip3 install jedi
	```
3. 重新用 python3 编译（**非常重要**）
	```bash
	python3 ./install.py --all
	```
4. 在 `vimrc` 中添加支持
	```bash
	let g:ycm_server_python_interpreter='/usr/bin/python3'
	let g:ycm_python_binary_path = '/usr/local/bin/python3'
	```

## 参考
- [YouCompleteMe Issue 2876](https://github.com/Valloric/YouCompleteMe/issues/2876)
- [ How do I complete Python3 with YouCompleteMe? ](https://vi.stackexchange.com/questions/6692/how-do-i-complete-python3-with-youcompleteme)