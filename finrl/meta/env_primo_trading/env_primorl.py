from typing import List

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from gymnasium.utils import seeding
from stable_baselines3.common.vec_env import DummyVecEnv


class StockTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        stock_dim: int,
        hmax: int,
        initial_amount: int,
        num_stock_shares: list[int],
        buy_cost_pct: list[float],
        sell_cost_pct: list[float],
        reward_scaling: float,
        state_space: int,
        action_space: int,
        tech_indicator_list: list[str],
        fundamental_indicator_list: list[str], 
        turbulence_threshold=None,
        print_verbosity=10,
        risk_indicator_col="turbulence",
        day=0,
        initial=True,
        previous_state=[],
        model_name="",
        mode="",
        iteration="",
        verbose=0,
    ):
        self.day = day
        self.df = df
        self.stock_dim = stock_dim
        self.hmax = hmax
        self.num_stock_shares = num_stock_shares
        self.initial_amount = initial_amount  # get the initial cash
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.action_space = action_space
        self.tech_indicator_list = tech_indicator_list
        self.fundamental_indicator_list = fundamental_indicator_list
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_space,))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_space,)
        )
        self.data = self.df.loc[self.day, :]
        self.terminal = False
        self.turbulence_threshold = turbulence_threshold
        self.print_verbosity = print_verbosity
        self.risk_indicator_col = risk_indicator_col
        self.initial = initial
        self.previous_state = previous_state
        self.model_name = model_name
        self.mode = mode
        self.iteration = iteration
        self.verbose = verbose
        # initalize state
        self.state = self._initiate_state()

        # initialize reward
        self.reward = 0
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.episode = 0
        # memorize all the total balance change
        self.asset_memory = [
            self.initial_amount
            + np.sum(
                np.array(self.num_stock_shares)
                * np.array(self.state[1 : 1 + self.stock_dim])
            )
        ]  # the initial total asset is calculated by cash + sum (num_share_stock_i * price_stock_i)
        self.rewards_memory = []
        self.actions_memory = []
        self.state_memory = (
            []
        )  # we need sometimes to preserve the state in the middle of trading process
        self.date_memory = [self._get_date()]

        self._seed()

        self._print(f"Initialized environment with {self.stock_dim} stocks")
        self._print(f"Technical indicators: {self.tech_indicator_list}")
        self._print(f"Fundamental indicators: {self.fundamental_indicator_list}")
        self._print(f"Initial amount: {self.initial_amount}")
        self._print(f"Initial stock shares: {self.num_stock_shares}")

    def _print(self, msg, level=1):
        if self.verbose >= level:
            print(msg)

    def _sell_stock(self, index, action):
        self._print(f"Attempting to sell stock {index}", level=2)
        self._print(f"  Current cash: {self.state[0]:.2f}", level=2)
        self._print(f"  Current stock holding: {self.state[index + self.stock_dim + 1]}", level=2)
        self._print(f"  Requested action: {action}", level=2)

        sell_num_shares = 0  # Initialize to 0

        if self.state[index + 2 * self.stock_dim + 1] != True:  # check if the stock is able to sell
            if self.state[index + self.stock_dim + 1] > 0:
                sell_num_shares = min(abs(action), self.state[index + self.stock_dim + 1])
                self.state[0] += self.state[index + 1] * sell_num_shares * (1 - self.sell_cost_pct[index])
                self.state[index + self.stock_dim + 1] -= sell_num_shares
                self.cost += self.state[index + 1] * sell_num_shares * self.sell_cost_pct[index]
                self.trades += 1

        self._print(f"  Sold {sell_num_shares} shares", level=2)
        self._print(f"  Remaining cash: {self.state[0]:.2f}", level=2)
        self._print(f"  Updated stock holding: {self.state[index + self.stock_dim + 1]}", level=2)

        return sell_num_shares

    def _buy_stock(self, index, action):
        self._print(f"Attempting to buy stock {index}", level=2)
        self._print(f"  Current cash: {self.state[0]:.2f}", level=2)
        self._print(f"  Current stock holding: {self.state[index + self.stock_dim + 1]}", level=2)
        self._print(f"  Requested action: {action}", level=2)

        buy_num_shares = 0  # Initialize to 0

        if self.state[index + 2 * self.stock_dim + 1] != True:  # check if the stock is able to buy
            if self.state[0] > self.state[index + 1] * action:  # check if cash is enough
                buy_num_shares = action
                self.state[0] -= self.state[index + 1] * buy_num_shares * (1 + self.buy_cost_pct[index])
                self.state[index + self.stock_dim + 1] += buy_num_shares
                self.cost += self.state[index + 1] * buy_num_shares * self.buy_cost_pct[index]
                self.trades += 1
            else:
                buy_num_shares = self.state[0] // (self.state[index + 1] * (1 + self.buy_cost_pct[index]))
                if buy_num_shares > 0:
                    self.state[0] -= self.state[index + 1] * buy_num_shares * (1 + self.buy_cost_pct[index])
                    self.state[index + self.stock_dim + 1] += buy_num_shares
                    self.cost += self.state[index + 1] * buy_num_shares * self.buy_cost_pct[index]
                    self.trades += 1

        self._print(f"  Bought {buy_num_shares} shares", level=2)
        self._print(f"  Remaining cash: {self.state[0]:.2f}", level=2)
        self._print(f"  Updated stock holding: {self.state[index + self.stock_dim + 1]}", level=2)

        return buy_num_shares

    def _calculate_reward(self, begin_total_asset, end_total_asset):
        """
        Calculates the custom reward based on two components:
        1. Return rate (70% weight) - Measures the profit/loss relative to initial investment
        2. Sharpe ratio (30% weight) - Measures risk-adjusted returns
        
        Args:
            begin_total_asset: Total portfolio value at the start of the step
            end_total_asset: Total portfolio value at the end of the step
            
        Returns:
            float: Weighted combination of return rate and Sharpe ratio, scaled by reward_scaling
        """
        # Calculate absolute profit/loss
        profit = end_total_asset - begin_total_asset
        # Calculate return rate (relative profit)
        return_rate = profit / begin_total_asset
        # Get Sharpe ratio from helper function
        sharpe_ratio = self._calculate_sharpe_ratio()
        
        # Combine return rate (70%) and Sharpe ratio (30%) for final reward
        reward = (return_rate * 0.7) + (sharpe_ratio * 0.3)
        return reward * self.reward_scaling

    def _calculate_sharpe_ratio(self):
        """
        Calculates the Sharpe ratio based on historical asset values.
        Sharpe ratio = (Average Return - Risk Free Rate) / Standard Deviation of Returns
        Here we assume risk-free rate = 0 for simplicity.
        
        Returns:
            float: Annualized Sharpe ratio, or 0 if insufficient data/zero standard deviation
        """
        # Need at least 2 data points to calculate returns
        if len(self.asset_memory) < 2:
            return 0
        
        # Calculate daily returns as percentage changes
        daily_returns = np.diff(self.asset_memory) / self.asset_memory[:-1]
        
        # Need at least 2 returns to calculate mean and std
        if len(daily_returns) < 2:
            return 0
        
        # Calculate average daily return and standard deviation
        avg_daily_return = np.mean(daily_returns)
        std_daily_return = np.std(daily_returns)
        
        # Avoid division by zero
        if std_daily_return == 0:
            return 0
        
        annual_factor = np.sqrt(252) # 252 trading days in a year
        
        # Calculate annualized Sharpe ratio
        sharpe_ratio = annual_factor * (avg_daily_return / std_daily_return)
        
        return sharpe_ratio

    def step(self, actions):
        # Na početku metode
        self._print(f"\nDay: {self.day}")
        self._print(f"Current state: {self.state}")
        self._print(f"Actions: {actions}")
        self._print(f"Number of stocks: {self.stock_dim}")

        self.terminal = self.day >= len(self.df.index.unique()) - 1
        if self.terminal:
            # print(f"Episode: {self.episode}")
            end_total_asset = self.state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)])
                * np.array(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
            )
            df_total_value = pd.DataFrame(self.asset_memory)
            tot_reward = (
                self.state[0]
                + sum(
                    np.array(self.state[1 : (self.stock_dim + 1)])
                    * np.array(
                        self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)]
                    )
                )
                - self.asset_memory[0]
            )  # initial_amount is only cash part of our initial asset
            df_total_value.columns = ["account_value"]
            df_total_value["date"] = self.date_memory
            df_total_value["daily_return"] = df_total_value["account_value"].pct_change(
                1
            )
            if df_total_value["daily_return"].std() != 0:
                sharpe = (
                    (252**0.5)
                    * df_total_value["daily_return"].mean()
                    / df_total_value["daily_return"].std()
                )
            df_rewards = pd.DataFrame(self.rewards_memory)
            df_rewards.columns = ["account_rewards"]
            df_rewards["date"] = self.date_memory[:-1]
            if self.verbose == 0:
                if self.episode % self.print_verbosity == 0:
                    print(f"Day: {self.day}, episode: {self.episode}")
                    print(f"Begin total asset: {self.asset_memory[0]:0.2f}")
                    print(f"End total asset: {end_total_asset:0.2f}")
                    print(f"Total reward: {tot_reward:0.2f}")
                    print(f"Total cost: {self.cost:0.2f}")
                    print(f"Total trades: {self.trades}")
                    if df_total_value["daily_return"].std() != 0:
                        print(f"Sharpe: {sharpe:0.3f}")
                    print("=================================")

            return self.state, self.reward, self.terminal, False, {}

        else:
            actions = actions * self.hmax  # actions initially is scaled between 0 to 1
            actions = actions.astype(
                int
            )  # convert into integer because we can't by fraction of shares
            if self.turbulence_threshold is not None:
                if self.turbulence >= self.turbulence_threshold:
                    actions = np.array([-self.hmax] * self.stock_dim)
            begin_total_asset = self.state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)])
                * np.array(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
            )
            # print("begin_total_asset:{}".format(begin_total_asset))

            argsort_actions = np.argsort(actions)
            sell_index = argsort_actions[: np.where(actions < 0)[0].shape[0]]
            buy_index = argsort_actions[::-1][: np.where(actions > 0)[0].shape[0]]

            for index in sell_index:
                # print(f"Num shares before: {self.state[index+self.stock_dim+1]}")
                # print(f'take sell action before : {actions[index]}')
                actions[index] = self._sell_stock(index, actions[index]) * (-1)
                # print(f'take sell action after : {actions[index]}')
                # print(f"Num shares after: {self.state[index+self.stock_dim+1]}")

            for index in buy_index:
                # print('take buy action: {}'.format(actions[index]))
                actions[index] = self._buy_stock(index, actions[index])

            self.actions_memory.append(actions)

            # state: s -> s+1
            self.day += 1
            self.data = self.df.loc[self.day, :]
            if self.turbulence_threshold is not None:
                if len(self.df.tic.unique()) == 1:
                    self.turbulence = self.data[self.risk_indicator_col]
                elif len(self.df.tic.unique()) > 1:
                    self.turbulence = self.data[self.risk_indicator_col].values[0]
            self.state = self._update_state()

            end_total_asset = self.state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)])
                * np.array(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
            )
            self.asset_memory.append(end_total_asset)
            self.date_memory.append(self._get_date())
            self.reward = self._calculate_reward(begin_total_asset, end_total_asset)
            self.rewards_memory.append(self.reward)
            self.state_memory.append(
                self.state
            )  # add current state in state_recorder for each step

            # Nakon izračuna begin_total_asset
            self._print(f"Begin total asset: {begin_total_asset:.2f}")

            # Nakon izvršenja trgovanja
            self._print("Executed trades:")
            for i, action in enumerate(actions):
                if action != 0:
                    self._print(f"  Stock {i}: {'Bought' if action > 0 else 'Sold'} {abs(action)} shares")

            # Nakon ažuriranja stanja
            self._print(f"End total asset: {end_total_asset:.2f}")
            self._print(f"Reward: {self.reward:.5f}")

            # Na kraju metode
            #self._print(f"Updated state: {self.state}")
            self._print(f"Current holdings:")
            for i in range(self.stock_dim):
                stock_price = self.state[i + 1]
                stock_shares = self.state[i + self.stock_dim + 1]
                self._print(f"  Stock {i}: Price: {stock_price:.2f}, Shares: {stock_shares}")
            self._print(f"Cash: {self.state[0]:.2f}")
            self._print(f"Total asset value: {end_total_asset:.2f}")
            self._print(f"Terminal: {self.terminal}")

        return self.state, self.reward, self.terminal, False, {}

    def reset(self, *, seed=None, options=None):
        # Na početku metode
        self._print(f"\nResetting environment. Episode: {self.episode}")

        # initiate state
        self.day = 0
        self.data = self.df.loc[self.day, :]
        self.state = self._initiate_state()

        # Nakon inicijalizacije stanja
        self._print(f"Initial state: {self.state}")
        self._print(f"Number of stocks: {self.stock_dim}")
        self._print(f"Initial holdings:")
        for i in range(self.stock_dim):
            stock_price = self.state[i + 1]
            stock_shares = self.state[i + self.stock_dim + 1]
            self._print(f"  Stock {i}: Price: {stock_price:.2f}, Shares: {stock_shares}")
        self._print(f"Initial cash: {self.state[0]:.2f}")
        self._print(f"Initial total asset: {self.asset_memory[0]:.2f}")

        if self.initial:
            self.asset_memory = [
                self.initial_amount
                + np.sum(
                    np.array(self.num_stock_shares)
                    * np.array(self.state[1 : 1 + self.stock_dim])
                )
            ]
        else:
            previous_total_asset = self.previous_state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)])
                * np.array(
                    self.previous_state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)]
                )
            )
            self.asset_memory = [previous_total_asset]

        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.terminal = False
        # self.iteration=self.iteration
        self.rewards_memory = []
        self.actions_memory = []
        self.date_memory = [self._get_date()]

        self.episode += 1

        return self.state, {}


    def _initiate_state(self):
        if self.initial:
            # For Initial State
            if len(self.df.tic.unique()) > 1:
                # for multiple stock
                state = (
                    [self.initial_amount]
                    + self.data.close.values.tolist()
                    + self.num_stock_shares
                    + sum(
                        (
                            self.data[tech].values.tolist()
                            for tech in self.tech_indicator_list
                        ),
                        [],
                    )
                    + sum(
                        (
                            self.data[fund].values.tolist()
                            for fund in self.fundamental_indicator_list
                        ),
                        [],
                    )
                )  # append initial stocks_share to initial state, instead of all zero
            else:
                # for single stock
                state = (
                    [self.initial_amount]
                    + [self.data.close]
                    + [0] * self.stock_dim
                    + sum(([self.data[tech]] for tech in self.tech_indicator_list), [])
                    + sum(([self.data[fund]] for fund in self.fundamental_indicator_list), [])
                )
        else:
            # Using Previous State
            if len(self.df.tic.unique()) > 1:
                # for multiple stock
                state = (
                    [self.previous_state[0]]
                    + self.data.close.values.tolist()
                    + self.previous_state[
                        (self.stock_dim + 1) : (self.stock_dim * 2 + 1)
                    ]
                    + sum(
                        (
                            self.data[tech].values.tolist()
                            for tech in self.tech_indicator_list
                        ),
                        [],
                    )
                    + sum(
                        (
                            self.data[fund].values.tolist()
                            for fund in self.fundamental_indicator_list
                        ),
                        [],
                    )
                )
            else:
                # for single stock
                state = (
                    [self.previous_state[0]]
                    + [self.data.close]
                    + self.previous_state[
                        (self.stock_dim + 1) : (self.stock_dim * 2 + 1)
                    ]
                    + sum(([self.data[tech]] for tech in self.tech_indicator_list), [])
                    + sum(([self.data[fund]] for fund in self.fundamental_indicator_list), [])
                )
        return state

    def _update_state(self):
        if len(self.df.tic.unique()) > 1:
            # for multiple stock
            state = (
                [self.state[0]]
                + self.data.close.values.tolist()
                + list(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
                + sum(
                    (
                        self.data[tech].values.tolist()
                        for tech in self.tech_indicator_list
                    ),
                    [],
                )
                + sum(
                    (
                        self.data[fund].values.tolist()
                        for fund in self.fundamental_indicator_list
                    ),
                    [],
                )
            )

        else:
            # for single stock
            state = (
                [self.state[0]]
                + [self.data.close]
                + list(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
                + sum(([self.data[tech]] for tech in self.tech_indicator_list), [])
                + sum(([self.data[fund]] for fund in self.fundamental_indicator_list), [])
            )

        return state

    def _get_date(self):
        if len(self.df.tic.unique()) > 1:
            date = self.data.date.unique()[0]
        else:
            date = self.data.date
        return date

    # add save_state_memory to preserve state in the trading process
    def save_state_memory(self):
        if len(self.df.tic.unique()) > 1:
            # date and close price length must match actions length
            date_list = self.date_memory[:-1]
            df_date = pd.DataFrame(date_list)
            df_date.columns = ["date"]

            state_list = self.state_memory
            df_states = pd.DataFrame(
                state_list,
                columns=[
                    "cash",
                    "Bitcoin_price",
                    "Gold_price",
                    "Bitcoin_num",
                    "Gold_num",
                    "Bitcoin_Disable",
                    "Gold_Disable",
                ],
            )
            df_states.index = df_date.date
            # df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
        else:
            date_list = self.date_memory[:-1]
            state_list = self.state_memory
            df_states = pd.DataFrame({"date": date_list, "states": state_list})
        # print(df_states)
        return df_states

    def save_asset_memory(self):
        date_list = self.date_memory
        asset_list = self.asset_memory
        # print(len(date_list))
        # print(len(asset_list))
        df_account_value = pd.DataFrame(
            {"date": date_list, "account_value": asset_list}
        )
        return df_account_value

    def save_action_memory(self):
        if len(self.df.tic.unique()) > 1:
            # date and close price length must match actions length
            date_list = self.date_memory[:-1]
            df_date = pd.DataFrame(date_list)
            df_date.columns = ["date"]

            action_list = self.actions_memory
            df_actions = pd.DataFrame(action_list)
            df_actions.columns = self.data.tic.values
            df_actions.index = df_date.date
            # df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
        else:
            date_list = self.date_memory[:-1]
            action_list = self.actions_memory
            df_actions = pd.DataFrame({"date": date_list, "actions": action_list})
        return df_actions

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs
