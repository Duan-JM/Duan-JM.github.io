---
layout: post
title: "代理 Socks 转 http"
date: 2018-10-13
categories: "Linux 小技巧"
tags: ['privoxy','socks5','http_proxy','教程']
---
## 前言
这两天，学长派的任务中，需要使用 Scrapy 爬去许多国外的网站，需要给 Scrapy 搭梯子，而 Scrapy 只支持 http 的代理，故记录下这次 socks 转 http 的步骤。
## 前置软件
- shadowsocks （就不解释了，大家都知道）
- privoxy

	*注：polipo 的方案已经过时了，privoxy 作为新的方案，更灵活*

## 配置步骤
1. 安装
	```
	sudo apt-get install shadowsocks privoxy
	```

2. 更改 privoxy 配置

	```bash
	vim /etc/privoxy/config

	# 找到 forward-socks5t 取消他的注释，注意后面的端口跟着的是 shadowsocks 的端口
	forward-socks5t / 127.0.0.1:1086 .
	# 注意后面有一个点，以及是 5t

	listen-address localhost:8118
	# 默认是 8118，可以自己改
	# 0.0.0.0 的话，可以给同局域网的设备使用 ss，比如 ps4 等
	```

3. 启动了 shadowsocks 之后，启动 privoxy

	```bash
	sudo systemctl restart privoxy.serivce
	#或者
	sudo privoxy --no-daemon /etc/privoxy/config
	```

4. 测试

	```bash
	export all_proxy=http://127.0.0.1:8118
	curl ip.cn
	#想要 scrapy 生效，这步 export all_proxy, 经过测试表明是必须的
	```
