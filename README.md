# 2022MCM-C
**2022年数模美赛C题代码** [[paper](https://github.com/dreaming-qin/2022MCM-C/blob/master/paper.pdf)]

- 整体工作

  ![2022MCM-C/our_work.jpg at master · dreaming-qin/2022MCM-C (github.com)](https://github.com/dreaming-qin/2022MCM-C/blob/master/img/our_work.jpg)
  
- 为什么用LSTM，简述结构？

  - 序列预测...

  - 隐状态 [h, c] ，遗忘门 f ，输入门 i ，输出门o。

  - 模型的输出取决于为 上一时刻的隐状态 和 输入x 。

  - 损失函数取均方误差。

- 简述VaR思想？

  - 假设k时刻的数据分布为D(k)，D(k)可以用 **第k天的预测价格** 和 **第[k-30, k-1]的价格** 相减，得到的一个**排序**的**利润分布**来表示。
  - 假设每个利润宽度为 len ，置信度为 c ，则第 len*(1-c) 天的价格就是 VaR 的值，表示有 c 的概率会赚这么多钱。（待补充）
  - 用 VaR/pk 就是**损失率 LR** 。

- 如何买卖？

  - **PI**：LSTM自回归预测未来15天的**最大价格**，rk为当天的**真实价格**，PI相当于一个利润比率。
    $$
    PI(k)=\frac {max(p_1, p_2, ..., p_{15})-r_k}{r_k}
    $$
    且价格最大的这一天就是要卖出的天dm。

  - **RI**：利用 LR ，相当于一个损失比率。
    $$
    RI(k)=(d_m-d_k) · LR
    $$
    
  - **F**：综合考虑PI和RI。rho取决于交易成本。beta是动态变化的，因为持有的资本越多，投入就越多，风险越大，就越要给RI更多的权重，F>0时就可以考虑交易。
    $$
    F=\beta · PI(k) + (1-\beta) · RI(k) - \rho
    $$
  
  - 算出黄金和比特币的 Fg, Fb ，进入有限状态自动机的状态。
  
    - 优先卖比特币。
    - 周末不交易。
    - ...各种状态，比较复杂。（待补充）
  
- 如何评价模型？

  - 根据交易策略，获得一个**总资产 M** 的走势曲线。

    ![2022MCM-C/result_1_new.jpg at master · dreaming-qin/2022MCM-C (github.com)](https://github.com/dreaming-qin/2022MCM-C/blob/master/img/result_1_new.jpg)

  - 对比几个模型：

    1. 三次指数平滑：（待补充）
    2. LSTM-only：只预测，若能盈利就买卖。
    3. VaR-only：只看风险，若能盈利就买卖。
    4. LSTM + VaR：综合。

  - **滑动窗口机制**，窗口大小80天，滑动步数20天。

  - 几个指标：

    1. **FAV** (Final Asset Value) 最终利润，越大越好

    2. **DR** (Down Risk) 下行风险，越小越好

       对于黄金，用利润的平均值 * 天数，得到一个总利润，加上资产总额 M ，得到黄金预计资产 EG

       对于比特币也是，得到 EB

       然后加权，得到**总预计资产E**，然后求得DR
     $$
       E = \lambda · EG + (1-\lambda)·EB \\
       DR = \sqrt{\frac{\sum min(0, E - M)^2}{n}}
       $$
       
    3. **MD** (Max Drawdown) 最大回撤，越小越好
    
       从窗口中间划分，取得左半侧的最大值Mmax，和右半侧的最小值Mmin
       $$
       MD=\frac {M_{min}}{M_{max}}
       $$
  
  ![2022MCM-C/problem_3_MD_FAV.jpg at master · dreaming-qin/2022MCM-C (github.com)](https://github.com/dreaming-qin/2022MCM-C/blob/master/img/problem_3_MD_FAV.jpg)

- 通过实验，探索了交易成本对rho的影响，从而对交易策略和最终利润的影响。

  

- 灵敏性分析：beta对模型的影响，不稳定性先下降后上升，取0.6最合适。

  ![2022MCM-C/sensitivity.jpg at master · dreaming-qin/2022MCM-C (github.com)](https://github.com/dreaming-qin/2022MCM-C/blob/master/img/sensitivity.jpg)
