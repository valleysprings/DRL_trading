import gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gym import spaces
from stable_baselines3.common.vec_env import DummyVecEnv
import os
matplotlib.use("Agg")


class CryptoTradingEnv(gym.Env):
    """
    A Crypto trading environment.
    
    Parameters
    ----------
    df: pd.DataFrame
        A pandas dataframe of crypto currency data.
    crypto_dim: int
        The number of crypto currency types to trade.
    cash: int
        The initial amount of money to invest.
    num_crypto_shares: list
        A list of integers representing the number of shares of each crypto currency to buy initially.
    buy_cost_pct: list
        A list of floats representing the transaction cost percentage for buying each crypto currency.
    sell_cost_pct: list
        A list of floats representing the transaction cost percentage for selling each crypto currency.
    state_space_dim: int
        The dimension of the state space.
    action_space_dim: int
        The dimension of the action space.
    tech_indicator_list: list
        A list of technical indicators to use.
    is_debug: bool
        Whether to emit debug information.
    model_name: str
        The name of the model.
    print_verbosity: int
        The interval for printing training logs.
    action_scaling: float
        The scaling factor for the actions.
    alpha: float
        A parameter for reward shaping.
    beta: float
        A parameter for reward shaping.
    gamma: float
        A parameter for reward shaping.
    zeta: float
        A parameter for reward shaping.
        
    Methods
    ----------
    __init__: 
        Initializes the CryptoTradingEnv object
    _initiate_state: 
        Initializes the state for the environment
    _update_state: 
        Updates the state for the environment
    _calculate_portfolio_value: 
        Calculates the portfolio value for the environment
    _sell_share: 
        Sells some share of cryptocurrency
    _buy_share: 
        Buys some share of cryptocurrency
    _make_plot: 
        Makes a plot of the portfolio values memory
    step: 
        Executes one time step within the environment
    reset: 
        Resets the environment for the next episode
    get_sb_env: 
        Returns the stable-baselines environment
    save_portfolio_memory: 
        Saves the portfolio memory
    save_reward_memory: 
        Saves the reward memory
    save_action_memory: 
        Saves the action memory
    save_state_memory: 
        Saves the state memory
    print_verbose: 
        Prints the verbose training logs
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        crypto_dim: int,
        cash: int,
        num_crypto_shares: list,
        buy_cost_pct: list,
        sell_cost_pct: list,
        state_space_dim: int,
        action_space_dim: int,
        tech_indicator_list: list,
        is_debug: bool = False,
        model_name = "",
        print_verbosity = 20,
        action_scaling = 0.1,
        eval_time_interval = 30,
        risk_control: bool = False,
        data_granularity = 1,

    ):
        
        self.timestamp = 0 # current timestamp
        self.reward = 0 # current env reward
        self.cum_buy_cost = 0 # cumulative buy costs
        self.cum_sell_cost = 0 # cumulative sell costs
        self.buy_trades = 0 # aggregated buy trades nums
        self.sell_trades = 0 # aggregated sell trades nums
        self.episode = 0 # current episode
         
        self.df = df # data
        self.max_timestamp = len(self.df.index.unique()) - 1 # max timestamp
        self.crypto_dim = crypto_dim # crypto ticket dimensions
        self.num_crypto_shares = num_crypto_shares # nums of crypto shares for each tic
        self.cash = cash # initial money
        self.action_scaling = action_scaling # action scaling (max allocated proportion)
        self.buy_cost_pct = buy_cost_pct # pct of transaction cost for buying
        self.sell_cost_pct = sell_cost_pct # pct of transaction cost for selling
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(action_space_dim,)) # action space
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(state_space_dim,)) # observation space
        self.tech_indicator_list = tech_indicator_list # technical indicators
        self.data = self.df.loc[self.timestamp] # data for current timestamp
        self.state = self._initiate_state() # initialize state
        self.print_verbosity = print_verbosity # training logs print interval
        self.portfolio_value = self._calculate_portfolio_value(self.cash, np.array(self.state[self.crypto_dim + 1 : self.crypto_dim * 2 + 1]), np.array(self.num_crypto_shares)) # current timestamp portfolio value
        self.model_name = model_name # model name
        self.is_debug = is_debug # emit some debug info for each print_verbosity episode
        self.portfolio_values_memory = [] # current episode portfolio value for each timestamp
        self.states_memory = [] # current episode states list for each timestamp        
        self.rewards_memory = [] # current episode rewards list for each timestamp
        self.actions_memory = [] # current episode actions list for each timestamp
        self.valid_actions_memory = [] # current episode valid actions list for each timestamp 
        
        self.alpha = 0.1 # short-term reward coefficient
        self.beta = 100 # max drawdown coefficient
        self.max_portfolio_value = self.portfolio_value # max portfolio value
        self.max_downturn_threshold = 0.05 # max downturn threshold
        self.immediate_sell = 0 # immediate sell flag
        self.eval_time_interval = eval_time_interval # evaluation time interval
        self.risk_control = risk_control # risk control flag
        self.data_granularity = data_granularity # data granularity 1 for 1day and 24 for 1hour and 24*60 for 1min
    
    
    def _initiate_state(self):
        if self.crypto_dim > 1:
            state = (
                [self.cash] + self.num_crypto_shares + self.data.close.values.tolist() 
                + sum((self.data[tech].values.tolist() for tech in self.tech_indicator_list), [])
            ) 
        else:
            state = (
                [self.cash] + self.num_crypto_shares + [self.data.close] 
                + sum(([self.data[tech]] for tech in self.tech_indicator_list), [])
            )
        return state
   
    
    def _update_state(self):
        if self.crypto_dim > 1:
            state = (
                [self.state[0]] + list(self.state[1 : (self.crypto_dim + 1)]) + self.data.close.values.tolist() 
                + sum((self.data[tech].values.tolist() for tech in self.tech_indicator_list), [])
            ) 
        else:
            state = (
                [self.state[0]] + list(self.state[1 : (self.crypto_dim + 1)]) + [self.data.close] 
                + sum(([self.data[tech]] for tech in self.tech_indicator_list), [])
            ) 
        return state


    def _calculate_portfolio_value(self, cash, crypto_prices, num_crypto_shares):
        return cash + np.sum(num_crypto_shares * crypto_prices)
      
        
    def _sell_share(self, index, action):
        if self.state[index + 1] > 0:
            sell_num_shares = min(abs(action) / self.state[index + self.crypto_dim + 1], self.state[index + 1])
            sell_amount = (self.state[index + self.crypto_dim + 1] * sell_num_shares * (1 - self.sell_cost_pct[index]))
            self.state[0] += sell_amount
            self.state[index + 1] -= sell_num_shares
            self.cum_sell_cost += (self.state[index + self.crypto_dim + 1] * sell_num_shares * self.sell_cost_pct[index])
            self.sell_trades += 1
        else:
            sell_amount = 0
            
        return sell_amount


    def _buy_share(self, index, action):
        if self.state[0] > 0:
            buy_amount = min(self.state[0], action)
            buy_num_shares = buy_amount / (self.state[index + self.crypto_dim + 1] * (1 + self.buy_cost_pct[index]))  
            self.state[0] -= buy_amount
            self.state[index + 1] += buy_num_shares
            self.cum_buy_cost += (self.state[index + self.crypto_dim + 1] * buy_num_shares * self.buy_cost_pct[index])
            self.buy_trades += 1
        else:
            buy_amount = 0
        
        return buy_amount
    
    def _reward_setting(self, terminal):
        # reward shaping
        if terminal: reward = np.log(self.portfolio_values_memory[-1] / self.portfolio_values_memory[0]) * self.timestamp / self.data_granularity
        elif self.timestamp == 0: reward = 0
        else:
            reward = np.log(self.portfolio_values_memory[-1] / self.portfolio_values_memory[-2])
            if self.timestamp > self.eval_time_interval and self.risk_control:
                short_term_reward = np.log(self.portfolio_values_memory[-1] / self.portfolio_values_memory[- self.eval_time_interval]) * self.alpha * self.eval_time_interval / self.data_granularity
                max_downturn_flag = (self.portfolio_values_memory[-1]/self.max_portfolio_value) < (1 - self.max_downturn_threshold)
                reward += short_term_reward
                reward -= max_downturn_flag * self.max_downturn_threshold * self.beta
                if (max_downturn_flag): self.immediate_sell = 1
                if (self.max_portfolio_value < self.portfolio_values_memory[-1]): self.max_portfolio_value = self.portfolio_values_memory[-1]
        return reward
    
    
    def _make_plot(self):
        plt.plot(self.portfolio_values_memory, "r")
        try:
            plt.savefig(f"../../results/value_plot/{self.model_name}/episode_{self.episode}.png")
        except:
            os.makedirs(f"../../results/value_plot/{self.model_name}")
            plt.savefig(f"../../results/value_plot/{self.model_name}/episode_{self.episode}.png")
        plt.close()
    
    def step(self, actions):
        # have to reset here because if we put this code in reset area, we cannot log the last state 
        # and it will cause some error in function like save_portfolio_memory. So I have to put it here.
        
        self.state = self._update_state()
        self.terminal = self.timestamp >= self.max_timestamp 
        if self.timestamp == 0:
            self.rewards_memory = []
            self.actions_memory = []
            self.valid_actions_memory = []
            self.states_memory = []
            self.portfolio_values_memory = []
            
        if self.immediate_sell == 1:
            actions = np.zeros(self.crypto_dim) - self.portfolio_value # sell all crypto at once if max downturn threshold is reached
            self.immediate_sell = 0
            self.max_portfolio_value = self.portfolio_values_memory[-1]
        self.actions_memory.append(actions)
        self.portfolio_value = self._calculate_portfolio_value(self.state[0], np.array(self.state[(self.crypto_dim + 1) : (self.crypto_dim * 2 + 1)]), np.array(self.state[1 : 1 + self.crypto_dim]))
        scaled_actions = actions * self.portfolio_value * self.action_scaling
        argsort_actions = np.argsort(actions)
        sell_index = argsort_actions[: np.where(actions < 0)[0].shape[0]]
        buy_index = argsort_actions[::-1][: np.where(actions > 0)[0].shape[0]]
        for index in sell_index: scaled_actions[index] = self._sell_share(index, scaled_actions[index]) * (-1)
        for index in buy_index: scaled_actions[index] = self._buy_share(index, scaled_actions[index])
        self.valid_actions_memory.append(scaled_actions / (self.portfolio_value * self.action_scaling))
        self.portfolio_value = self._calculate_portfolio_value(self.state[0], np.array(self.state[(self.crypto_dim + 1) : (self.crypto_dim * 2 + 1)]), np.array(self.state[1 : 1 + self.crypto_dim]))
        self.portfolio_values_memory.append(self.portfolio_value)
        self.reward = self._reward_setting(self.terminal)
        self.rewards_memory.append(self.reward)   
        self.states_memory.append(self.state)
        
        if self.terminal:
            if self.episode % self.print_verbosity == 0 and self.model_name != "":
                self.print_verbose()
                if self.is_debug:
                    self._make_plot()
                    df_portfolio_value = self.save_portfolio_memory()
                    df_rewards = self.save_reward_memory()
                    df_actions = self.save_action_memory()
                    df_states = self.save_state_memory()
                    df_valid_actions = self.save_valid_action_memory()
                    try:
                        df_actions.to_csv(f"../../results/actions/{self.model_name}/episode_{self.episode}.csv")
                        df_valid_actions.to_csv(f"../../results/valid_actions/{self.model_name}/episode_{self.episode}.csv")
                        df_rewards.to_csv(f"../../results/rewards/{self.model_name}/episode_{self.episode}.csv")
                        df_portfolio_value.to_csv(f"../../results/portfolio_value/{self.model_name}/episode_{self.episode}.csv")
                        df_states.to_csv(f"../../results/states/{self.model_name}/episode_{self.episode}.csv")                    
                    except:
                        os.makedirs(f"../../results/actions/{self.model_name}")
                        os.makedirs(f"../../results/valid_actions/{self.model_name}")
                        os.makedirs(f"../../results/rewards/{self.model_name}")
                        os.makedirs(f"../../results/portfolio_value/{self.model_name}")
                        os.makedirs(f"../../results/states/{self.model_name}")
                        df_actions.to_csv(f"../../results/actions/{self.model_name}/episode_{self.episode}.csv")
                        df_valid_actions.to_csv(f"../../results/valid_actions/{self.model_name}/episode_{self.episode}.csv")
                        df_rewards.to_csv(f"../../results/rewards/{self.model_name}/episode_{self.episode}.csv")
                        df_portfolio_value.to_csv(f"../../results/portfolio_value/{self.model_name}/episode_{self.episode}.csv")
                        df_states.to_csv(f"../../results/states/{self.model_name}/episode_{self.episode}.csv")
        else:
            self.timestamp += 1
            self.data = self.df.loc[self.timestamp]
                           
        return self.state, self.reward, self.terminal, {}
            
            
    def reset(self):
        # initiate all state dependent variable for next episode
        self.timestamp = 0
        self.data = self.df.loc[self.timestamp]
        self.state = self._initiate_state()
        self.portfolio_value = self._calculate_portfolio_value(self.cash, np.array(self.state[self.crypto_dim + 1 : 2 * self.crypto_dim + 1]), np.array(self.num_crypto_shares))
        self.cum_buy_cost = 0 
        self.cum_sell_cost = 0 
        self.buy_trades = 0 
        self.sell_trades = 0 
        self.terminal = False
        self.immediate_sell = 0
        self.max_portfolio_value = self.portfolio_value
        self.episode += 1
        
        return self.state
    
    
    def get_sb_env(self):
        env = DummyVecEnv([lambda: self])
        obs = env.reset()
        return env, obs
    
    
    def save_portfolio_memory(self):
        timestamps = self.df.timestamp.unique()
        portfolio_values_memory = self.portfolio_values_memory
        df_portfolio_value = pd.DataFrame(
            {"timestamp": timestamps, "portfolio_value": portfolio_values_memory}
        )
        return df_portfolio_value
    
    def save_reward_memory(self):
        date_list = self.df.timestamp.unique()
        timestamps = pd.DataFrame(date_list)
        timestamps.columns = ["timestamp"]
        
        df_rewards = pd.DataFrame(self.rewards_memory)
        df_rewards.columns = ["account_rewards"]
        df_rewards.index = timestamps.timestamp
        return df_rewards
    
    def save_valid_action_memory(self):
        if self.crypto_dim > 1:
            # date and close price length must match actions length
            date_list = self.df.timestamp.unique()
            timestamps = pd.DataFrame(date_list)
            timestamps.columns = ["timestamp"]

            action_list = self.valid_actions_memory
            df_actions = pd.DataFrame(action_list)
            df_actions.columns = self.data.tic.values
            df_actions.index = timestamps.timestamp            
        else:
            date_list = self.df.timestamp.unique()
            action_list = self.valid_actions_memory
            df_actions = pd.DataFrame({"date": date_list, "valid_actions": action_list})
        return df_actions

    def save_action_memory(self):
        if self.crypto_dim > 1:
            # date and close price length must match actions length
            date_list = self.df.timestamp.unique()
            timestamps = pd.DataFrame(date_list)
            timestamps.columns = ["timestamp"]

            action_list = self.actions_memory
            df_actions = pd.DataFrame(action_list)
            df_actions.columns = self.data.tic.values
            df_actions.index = timestamps.timestamp            
        else:
            date_list = self.df.timestamp.unique()
            action_list = self.actions_memory
            df_actions = pd.DataFrame({"date": date_list, "actions": action_list})
        return df_actions
    
    def save_state_memory(self):
        date_list = self.df.timestamp.unique()
        timestamps = pd.DataFrame(date_list)
        timestamps.columns = ["timestamp"]

        states_list = self.states_memory
        df_states = pd.DataFrame(states_list)
        df_states.index = timestamps.timestamp 
        return df_states
    
    def print_verbose(self):
        print("=========================================")        
        print(f"episode:\t\t {self.episode}")
        print(f"begin_portfolio_value:\t {self.portfolio_values_memory[0]:0.2f}")
        print(f"end_portfolio_value:\t {self.portfolio_value:0.2f}")
        print(f"total_profits:\t\t {self.portfolio_values_memory[-1] - self.portfolio_values_memory[0]:0.2f}")
        print(f"total_buy_cost:\t\t {self.cum_buy_cost:0.2f}")
        print(f"total_sell_cost:\t {self.cum_sell_cost:0.2f}")
        print(f"total_buy_trades:\t {self.buy_trades}")
        print(f"total_sell_trades:\t {self.sell_trades}")
        print("=========================================\n") 