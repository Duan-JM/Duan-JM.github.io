---
layout: post
title: "BaiduPCS-Go的使用"
date: 2018-08-17
categories: 软件使用
tags: ['百度网盘','下载','教程']
---
## Why BaiduPCS-Go
BaiduPCS-Go是一个用Go语言编的命令行版的百度网盘，我们可以类比mas和Appstore的关系。那么为什么要用这样一个安装比较麻烦，还要记命令行的百度网盘的替代品，直接用百度网盘客户端不好么？
这还真的是不好，百度网盘在mac下是一个十足的阉割版，最常用的功能中，Mac版缺失了以下几种功能：
1. **没有分享功能**：mac下的客户端的分享功能居然是需要通过浏览器打开，太不优雅了。
2. **没有离线下载任务**：直接导致不能下载磁力链接。
如果你和我一样平时一样习惯终端操作，这个工具的学习成本超级低，同时它还有一定的提升下载速度的功效。

## 使用指南
### 安装
Mac一般是预装了go的，如果没有的话，使用`brew install go`来安装。除了go我们还需要安装git，同样使用`brew install git`。在拥有了git和go以后，执行下面的指令即可。
```bash
go get -u -v github.com/iikira/BaiduPCS-Go
```
注：在安装途中，有提示说其安装到了一个`~/go/bin`的目录，也就是说这个工具的执行文件在`~/go/bin`这个目录。
为了之后我们能够全局使用这个指令，于是我们将`export PATH="/Users/deamov/go/bin:$PATH"`添加到配置环境变量的文件中，如果没有使用zsh的话在`~/.bashrc`中，如果用的是zsh的话在`~/.zshrc`中。
*注：deamov是我的电脑的用户名，至此安装便结束了。*

### 常用操作说明
#### 登陆
```bash
BaiduPCS-Go
```
简单一行指令就可以登录了，如果之前已经登陆过账号的话，现在就已经可以开始进行下载等操作了，如下效果图。
 ![](https://ws3.sinaimg.cn/large/006tNbRwly1fucuk0k8hnj310c0lojst.jpg)
第一次使用需要有登陆的操作，输入`login`即可登陆，尊许提示依次输入账户和密码即可，如果需要验证码，则会输出一个链接，打开就可以看到验证码了。
#### 基本操作
基本的移动目录的方式和linux的操作一样，`ls`是现实当前目录的文件，`rm`是删除命令，`cd`是切换目录，创建目录是`mkdir`，拷贝是`cp`，值得一提的是它支持Tab补全。和平时使用的终端命令不同的有如下几个指令。
- **搜索：**平时我们使用的`grep`在这里是不能使用的，我们用`search`关键词来搜索。
	```bash
	search 关键字 # 搜索当前工作目录的文件
	search -path=/ 关键字 # 搜索根目录的文件
	search -r 关键字	# 递归搜索当前工作目录的文件 
	```
- **下载：**记住是download就好啦
	```bash
	BaiduPCS-Go download <网盘文件或目录的路径1> 
	BaiduPCS-Go d <网盘文件或目录的路径1> <文件或目录2> <文件或目录3> ...
	# 当然支持多文件下载咯，下载目录默认在~/Download文件夹中
	```
- **离线下载: **支持http/https/ftp/电驴/磁力链协议
	```bash
	# 将百度和腾讯主页, 离线下载到根目录 /
	offlinedl add -path=/ http://baidu.com http://qq.com

	# 添加磁力链接任务
	offlinedl add magnet:?xt=urn:btih:xxx

	# 查询任务ID为 12345 的离线下载任务状态
	offlinedl query 12345

	# 取消任务ID为 12345 的离线下载任务
	offlinedl cancel 12345 
	```
#### 分享share
- **查看分享内容**
	```bash
	share list
	share l
	```
- **取消分享**
	```bash
	share cancel <shareid_1>
	share c <shareid_1>
	# 遗憾的是只能支持通过shareid来取消分享
	```
#### 上传：同名文件会被覆盖
*注：需要退出BaiduPCS-Go使用，否则本地文件目录不能自动补全*
```bash
$BaiduPCS-Go upload <本地文件/目录的路径1> <文件/目录2> <文件/目录3> ... <目标目录>
$BaiduPCS-Go u <本地文件/目录的路径1> <文件/目录2> <文件/目录3> ... <目标目录>
# Example
$BaiduPCS-Go upload ~/Downloads/1.mp4 /Video
```
### 其他
这个工具很强大，还可以通过设置下载线程数等等操作来提升下载速度，更多详细的操作请参考它的[官网](https://github.com/iikira/BaiduPCS-Go)。
