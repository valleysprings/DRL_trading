# Crypto DRL trading DEMO

**Nothing in this demo constitutes professional and/or financial advice, nor does any information on this repo constitute a comprehensive or complete statement of the matters discussed or the law relating thereto.**

## Descriptions

This demo aims to create an intelligent and dynamic trading platform prototype by harnessing the power of deep reinforcement learning (DRL). Using the Stable baseline 3 framework, this demo showcases the capabilities of DRL in multi-stock trading scenarios, with a special focus on crypto markets.

The project consists of a series of notebooks that cover various aspects of the DRL-based trading system, from gym environments and basic RL knowledge to data sources for crypto and code analysis of the most-popular DRL trading platform: FinRL. I have also written some notes that incorporate related topics such as mathematical derivations for RL tradings, technical indicator analysis, and data cleaning and exploratory data analysis (EDA).

## Notes (written in Chinese)

1. Hand on RL: gym environment and basic RL knowledge: https://www.wolai.com/davidzjw/iEYvTXN9zz8mnwZFRFt1Rb
2. FinRL demo: multi-stock trading: https://www.wolai.com/davidzjw/7Mo7cbAhVjpCZK2PSSgETt
3. Data source for crypto and some code analysis of FinRL: https://www.wolai.com/davidzjw/8gBjJXajTD8ncfUQhkvP1Z
4. Hand on multi-crypto trading via DRL: https://www.wolai.com/davidzjw/3TUsZ1hAzvyKNtjwmG9uGH
5. Some mathematical derivations for RL tradings: https://www.wolai.com/davidzjw/aAbJMKREaCwSCPg2dzoqQF


## Notebooks

1. primarily fetchs crypto-trading data using yfinance
2. aims to 
   1. add some techinal indicators
   2. does some data-cleaning
   3. EDA
3. aims to get the wheel turning by using popular RL algorithms
   
## code structure

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
