

# 利用深度强化学习进行加密货币交易

**仅供参考,不构成投资建议**

## 项目介绍

本项目采用深度强化学习（DRL）技术，以加密货币交易为应用场景，基于FinRL框架开发了一个交易系统的一个应用。项目包含多个Jupyter笔记本和笔记，涵盖了强化学习的基础知识、加密货币数据源、技术指标分析、数据清洗和探索性数据分析等多个方面。通过对时间粒度、投资组合、奖励、交易成本和波动性回避策略等因素进行研究，项目展示了DRL技术在多股交易场景中的适应性和效果。该项目希望能够将这些智能体应用于实时交易场景。

## 非常粗糙的笔记

1. 上手强化学习工程：gym环境以及强化学习基础 https://www.wolai.com/davidzjw/iEYvTXN9zz8mnwZFRFt1Rb
2. FinRL上手：多股交易实例 https://www.wolai.com/davidzjw/7Mo7cbAhVjpCZK2PSSgETt
3. 搞点数据和代码：加密货币数据源接入与FinRL代码分析 https://www.wolai.com/davidzjw/8gBjJXajTD8ncfUQhkvP1Z
4. 真的来了：加密货币的模拟交易算法实战 https://www.wolai.com/davidzjw/3TUsZ1hAzvyKNtjwmG9uGH
5. 理论篇：补全交易的理论基础 https://www.wolai.com/davidzjw/aAbJMKREaCwSCPg2dzoqQF


## 工作

1. 获取相关数据
2. 数据处理与可视化
3. 模型训练
4. 实验
   
## 代码结构

``` 
crypto_trading/
├── agent
├── dataset
│   ├── daily-level
│   └── hour-level
├── model_saves
├── notebooks
│   ├── data_retrieve
│   ├── EDA
│   └── training
├── plot
│   ├── dataset
│   ├── fig_save
│   └── test_plot
├── processor
├── results
└── trading_env
```
