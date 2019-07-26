# 第二课 马尔可夫决策过程 MDP
Markov Decision Process 是强化学习的核心，帅气的 David 说所有的强化学习问题都可以转化为 MDP，即就像 RBM 是深度学习的发源地一样，MDP 是整个强化学习的基础。而和名字一样，我们需要首先理解 Markov 和 Decision（Reward），接下来会从 Markov 过程到 Markov 过程加上 Reward 之后的马尔可夫奖励过程，最后引入 Bellman 方程，通过解 Bellman 方程的方式深入了解到底何为决策。
## 马尔可夫过程  Markov Process
谈到 Markov 过程，就需要提到 Markov 的一个前提条件，即 **Markov Process 认为下一个 t+1 时刻的状态，仅同当前 t 时刻的状态有关**。David 帅大大，对此的解释是认为其当前的状态其实已经包含了之前的状态的信息了。所以就可以用下面的公式表达状态转移概率了：
$$
P_{ss'} = P[S_{t+1} = s' | S_t = s]
$$
接下来我们联想当年求最短路径的算法 Floyd 的那个矩阵，我们只需要用矩阵  $S_t$ 表示当前的状态，再使用矩阵 $P$ 作为代表不同状态之间的转移概率，两者相乘就能得到下一时刻的状态矩阵 $S_{t+1}$ 。而 Markov Process 就是这个当前状态 $S$ 和概率转移矩阵 $P$ 不停相乘的过程，简单的计为 $<S, P>$ 。
$$
P = \left[ \begin{array}{cc}
        P_{11} & P_{12} & P_{13} & ... & P_{1n} \\ 
        P_{21} & P_{22} & P_{23} & ... & P_{2n} \\
		... & ... & ... & ... & ... \\
		P_{n1} & P_{n2} & P_{n3} & ... & P_{nn} 
        \end{array} 
\right]
$$
*注：其中由于 $P$ 中每一行代表着行号表示的状态转移到另一个状态的所有情况，所以每一行的概率和一定等于 1* 

## 马尔可夫奖励过程 Markov Reward Process
上一部分我们讲的 Markov Process 它还是没有奖励的，也就是说没有一个评判的依据，当我们把奖励引入的话就成了 MRP 了。在引入的过程中，我们用 $\gamma$ 作为衰减系数，表达对未来的收益的看重程度。这样我们对 MRP 就记为 $< S,P,R, \gamma >$ 。

接下来，我们进入细节的部分。为了更方便说明价值函数 $v(s)$，我们引入一个叫做收益 Return 的符号，其代表了当前状态，结合一定的未来情况能拿到的奖励，记做 $G_t$。
$$
G_t = R_{t+1} + \gamma R_{t+2} + ... = \sum_{k=0}^{\inf}\gamma^k R_{t+k+1}
$$
  
现在就可以讲述价值函数是什么了，其衡量了当前的状态的长期价值，即马尔科夫奖励过程中，从该状态开始的马尔科夫的收益的期望：
$$
v(s) = E[G_t|S_t = s]
$$
*注：聪明的朋友已经发现了，这里的 $v(s)$ 的值和 $G_t$ 的定义很相似，之后在 Bellman 方程中就会体现。  *

### Bellman 方程
现在们有了上面的价值函数的表达式了，但是似乎还是看不出什么好玩的地方，接下来我们把它展开，进行推导，会得到很有趣的结果：
\begin{split}
v(s) &= E[G_t|S_t = s] \\
&= E[R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + ... | S_t = s] \\
&= E[R_{t+1} + \gamma (R_{t+2} + \gamma R_{t+3} + ...) | S_t =s] \\
&= E[R_{t+1} + \gamma G_{t+1} | S_t = s] \\
&= E[R_{t+1} + \gamma v(S_{t+1}) | S_t = s] \\
&= R_s + \gamma \sum_{s' \in \mathcal{S}} \mathcal{p}_{ss'} v(s') 
\end{split}
接下来换成矩阵的形式为：
$$
v = \mathcal{R} + \gamma \mathcal{P}v  
$$
把矩阵写开为：
$$
\left[ \begin{array}{cc}
        v(1) \\ 
        ... \\
		v(n)
        \end{array} 
\right] =

\left[ \begin{array}{cc}
        R_1 \\ 
        ... \\
		R_n
        \end{array} 
\right] +

\gamma

\left[ \begin{array}{cc}
        P_{11}& ... & P_{1n} \\ 
		... & ... & ... \\
		P_{n1} & ... & P_{nn} 
        \end{array} 
\right]
\left[ \begin{array}{cc}
        v(1) \\ 
        ... \\
		v(n)
        \end{array} 
\right]
$$
这样我们就可以直接求解：
$$
v = (1 - \gamma \mathcal{P})^{-1} \mathcal{R}  
$$
**但是遗憾的是，计算复杂度为 $O(n^3)$ ，该算法的时间复杂度很高**。对此，大规模MRP的求解通常使用迭代法。常用的迭代方法有：
- 动态规划 Dynamic Programming
- 蒙特卡洛评估 Monte-Carlo evaluation
- 时序差分学习 Temporal-Difference
## 马尔可夫决策过程 Markov Decision Process
在 MRP 中我们还没有引入 Action，而我们 MDP 的目的则是找出最佳的 Action，于是 MDP 便引入了**有限行为集合** $A$ ，记做  $< S,A,P,R,\gamma >$ 。接下来，我们就引入了第一课中提到的策略 $\pi$ , 策略是概率的集合或分布，代表着当前状态 $s$ 采取行为 $s$ 的概率，用 $\pi(a|s)$ 表示。

这样从状态 $s$ 到下一个状态 $s'$ 就需要考虑到动作的因素，于是在策略 $\pi$ 下，又 $s$ 转移到 $s'$ 的转移概率为：
$$
\mathcal{P}_{s,s'}^{\pi} = \sum_{a\in\mathcal{A}} \pi(a|s)\mathcal{R}_{ss'}^a  
$$
同时奖励函数变为：
$$
\mathcal{R}_s^{\pi} = \sum_{a\in A} \pi(a|s)\mathcal{R}_s^a  
$$
这样基于策略 $\pi$ 的价值函数为：
$$
\begin{split}
v_{\pi}(s) &= E_{\pi}[G_t | S_t = s] \\ 
&= E_{\pi}[R_{t+1} + \gamma v_{\pi}(S_{t+1}) | S_t = s]
\end{split}
$$
行为价值函数 $q_{\pi}(s, a)$ 则为：
$$
\begin{split}
q_{\pi}(s, a)&=E_{\pi}[G_t | S_t = s， A_t = a] \\
&=E_{\pi}[R_{t+1} + \gamma q_{\pi}(S_{t+1}, A_{t+1}) | S_t = s, A_t = a]
\end{split}
$$
 
接下来我们在看 Bellman 方程，接下来我们做的是看如何把两个价值函数连在一起，他们之前的关系如下：
![](CleanShot%202019-02-22%20at%2020.00.11@2x.png)
![](CleanShot%202019-02-22%20at%2020.03.32@2x.png)
接下来把他们合起来，如下图所示，则公式为：
$$
v_{\pi}(s) = \sum_{a \in A} \pi(a|s) (\mathcal{R}_s^a + \gamma \sum_{s' \in \mathcal{S}} v_{\pi}(s'))  
$$

$$
q_{\pi}(s,a)=\mathcal{R}_s^{a} + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}_{ss'}^a (\sum_{a' \in \mathcal{A}} \pi(a'|s')q_{\pi}(s',a'))
$$
### 决策
公式写完了，我们接下来就是要进入决策过程，分为两个一个是价值函数最优。简单的来说，就是选一个策略让 $v(s)$  和  $q(s,a)$ 最大，公式如下：
$$
v_* = max_{\pi} v_{\pi}(s)
$$

$$
q_* = max_{\pi} q_{\pi}(s,a)
$$

这个解是否存在呢，David 帅哥说存在的，对于任何MDP，下面几点成立：
1. 存在一个最优策略，比任何其他策略更好或至少相等
2. 所有的最优策略有相同的最优价值函数
3. 所有的最优策略具有相同的行为价值函数
以上定理奠定了我们理论上能找到最优策略。那么既然存在我们应该怎么找呢，当然是用 Bellman 最优方程了，遗憾的是，这个同深度学习一样，函数是非线性的所以不能直接求解，需要通过迭代的方式。Bellman 最优方程如下：
$$
v_*(s) = max_a \mathcal{R}_s^a + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}_{ss'}^a v_*(s')  
$$

$$
q_*(s,a) = \mathcal{R}_s^a + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}_{ss'}^a max_{a'}q_*(s', a') 
$$

*注：解的迭代方法有价值迭代、策略迭代、Q学习、Sarsa等*
## 参考链接
- [Joey 老师的教程 02](https://blog.csdn.net/dukuku5038/article/details/84361371)
- [David Silver 的强化学习课系列](https://space.bilibili.com/74997410/video)