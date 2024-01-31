---
layout: post
title: "安装配置 Oh my Zsh"
date: 2018-11-15
categories: 软件使用
tags: ['oh_my_zsh','教程']
---

### Why Zsh
1. 好看！
2. 有git实时显示

简单直白粗暴、上述亮点就是我换oh my zsh的初衷。用了它直接免去了很多的重复操作、让终端更美观、让生活更美好。

### 安装
> Talk is Cheap Show me the Code

```bash
sudo apt-get install zsh # linux
brew install zsh # mac

# ===> install oh_my_zsh
git clone git://github.com/robbyrussell/oh-my-zsh.git ~/.oh-my-zsh
cp ~/.oh-my-zsh/templates/zshrc.zsh-template ~/.zshrc

# 设置 zsh 默认启动
chsh -s /bin/zsh

# 从个人的 dotfile 备份恢复设置
git clone https://github.com/Duan-JM/dotfiles.git
cp dotfiles/zsh/zshrc ~/.zshrc
cp -rf dotfiles/zsh/.zsh-config ~/.zsh-config

# 恢复主题
https://github.com/iplaces/astro-zsh-theme.git
cp astro.zsh-theme ~/.oh-my-zsh/themes/

# set ZSH_THEME="astro" in the ~/.zshrc
```
之后，切换默认打开的终端指令。如果想换回 `bash` 可以用 `chsh -s /bin/bash` 换回来。
至于如何切换主题，见下一章。
	
### 更换个人的主题
1. 将自己的主题文件（`file.theme`）放入`~/.oh-my-zsh/themes` 文件夹中
2. 在`~/.zshrc` 中 `ZSH_THEME` 选择自己的主题。

### 更换Powerline主题 (Optional)
1. git下powerline的主题。

	```bash
	git clone git://github.com/jeremyFreeAgent/oh-my-zsh-powerline-theme 
	```

	随后，`./install.sh`就好了。

2. 安装字体。

	```bash
	git clone https://github.com/powerline/fonts.git
	```

	同理，`./install.sh`就可以安装号字体了。

3. 终端下设置字体为`Meslo LG M for powerLine`。
4. 在zshrc里设置主题

	```bash
	ZSH_THEME="powerline" 
	```


## [Plugin] Zsh-syntax-highlighting

可以语法高亮你的 shell 命令，重要的是会标出你输入错误的指令。

![](https://ws2.sinaimg.cn/large/006tNbRwly1fx8kotg7xzg30lh02d3yo.gif)

1. 从 git 上下载到指定位置
	```bash
	git clone https://github.com/zsh-users/zsh-syntax-highlighting ~/.oh-my-zsh/custom/plugins/zsh-syntax-highlighting
	```
2. 添加到 `zshrc` 中的 `plugin list`
	它的名字为 `zsh-syntax-highlighting`，添加到plugins中就好了
	
## [Plugin] Zsh-autosuggestions

它会根据你的指令输入记录来补全你的指令，非常有用。
![](https://ws2.sinaimg.cn/large/006tNbRwly1fx8kotdfzsg30li02amxk.gif)

1. 从 git 上下载
	```bash
	git clone https://github.com/zsh-users/zsh-autosuggestions ~/.oh-my-zsh/custom/plugins/zsh-autosuggestions
	```

2. 添加到 `zshrc` 中的 `plugin list`
	它的名字为 `zsh-autosuggestions`，添加到plugins中就好了

## 参考
本文主要参考为bo\_song的[文章](https://www.jianshu.com/p/563dc1da2199)，里面还有详细的配色可以去关注一下～
