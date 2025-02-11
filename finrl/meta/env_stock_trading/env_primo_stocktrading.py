from __future__ import annotations
import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gymnasium import spaces
from gymnasium.utils import seeding
from stable_baselines3.common.vec_env import DummyVecEnv

matplotlib.use("Agg")

class StockTradingEnv(gym.Env):
    """Okruženje za trgovanje dionicama kompatibilno s OpenAI Gymnasium"""

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
        cash_penalty_proportion=float,
        market_trend_window=float,
        stop_loss_pct=float,
        take_profit_pct=float,
        turbulence_threshold=None,
        risk_indicator_col="turbulence",
        make_plots: bool = False,
        print_verbosity=10,
        day=0,
        initial=True,
        previous_state=[],
        model_name="",
        mode="",
        iteration="",
    ):
        """
        Inicijalizacija okruženja za trgovanje dionicama.

        Parametri:
        - df (pd.DataFrame): DataFrame s podacima o dionicama
        - stock_dim (int): Broj različitih dionica kojima se trguje
        - hmax (int): Maksimalni broj dionica koje se mogu trgovati u jednom potezu
        - initial_amount (int): Početni iznos gotovine
        - num_stock_shares (list[int]): Početni broj dionica za svaku dionicu
        - buy_cost_pct (list[float]): Postotak troška kupnje za svaku dionicu
        - sell_cost_pct (list[float]): Postotak troška prodaje za svaku dionicu
        - reward_scaling (float): Faktor skaliranja nagrade
        - state_space (int): Dimenzija prostora stanja
        - action_space (int): Dimenzija prostora akcija
        - tech_indicator_list (list[str]): Lista tehničkih indikatora koji se koriste
        - turbulence_threshold (float, optional): Prag turbulencije tržišta
        - risk_indicator_col (str): Ime stupca koji se koristi kao indikator rizika
        - make_plots (bool): Treba li generirati grafove
        - print_verbosity (int): Učestalost ispisa informacija
        - day (int): Početni dan
        - initial (bool): Je li ovo inicijalno stanje
        - previous_state (list): Prethodno stanje (ako nije inicijalno)
        - model_name (str): Ime modela koji se koristi
        - mode (str): Način rada (trening, testiranje, itd.)
        - iteration (str): Broj iteracije
        """
        # Inicijalizacija parametara okruženja
        self.day = day
        self.df = df
        self.stock_dim = stock_dim
        self.hmax = hmax
        self.num_stock_shares = num_stock_shares
        self.initial_amount = initial_amount
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling
        self.state_space = state_space

        # Definiranje prostora akcija kao kontinuiranog prostora između -1 i 1
        self.action_space = action_space
        self.tech_indicator_list = tech_indicator_list

        # Parametri za dinamičku prilagodbu cash penalty-a
        self.cash_penalty_proportion = cash_penalty_proportion
        self.market_trend_window = market_trend_window   

        # Parametri za stop loss i take profit
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.buy_prices = np.zeros(self.stock_dim)    

        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_space,))

        # Definiranje prostora opservacija kao beskonačnog prostora
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_space,)
        )
        self.data = self.df.loc[self.day, :]
        self.terminal = False
        self.make_plots = make_plots
        self.print_verbosity = print_verbosity
        self.turbulence_threshold = turbulence_threshold
        self.risk_indicator_col = risk_indicator_col
        self.initial = initial
        self.previous_state = previous_state
        self.model_name = model_name
        self.mode = mode
        self.iteration = iteration

        # Inicijalizacija stanja
        self.state = self._initiate_state()

        # Inicijalizacija varijabli za praćenje
        self.reward = 0
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.episode = 0

        # Inicijalizacija memorije za praćenje promjena u ukupnoj imovini
        self.asset_memory = [
            self.initial_amount
            + np.sum(
                np.array(self.num_stock_shares)
                * np.array(self.state[1 : 1 + self.stock_dim])
            )
        ]  # the initial total asset is calculated by cash + sum (num_share_stock_i * price_stock_i)
        
        self.rewards_memory = []
        self.actions_memory = []

        # Ponekad trebamo sačuvati stanje usred procesa trgovanja
        self.state_memory = (
            []
        ) 
        
        self.date_memory = [self._get_date()]
        
        self._seed()

    def _sell_stock(self, index, action):
        """
        Izvršava prodaju dionica.

        Parametri:
        - index (int): Indeks dionice koja se prodaje
        - action (int): Količina dionica za prodaju (negativna vrijednost)

        Vraća:
        - int: Stvarni broj prodanih dionica
        """
        def _do_sell_normal():
            if (self.state[index + 2 * self.stock_dim + 1] != True):  
                # Provjera je li dionica dostupna za prodaju, za jednostavnost smo to dodali u tehnički indeks
                # if self.state[index + 1] > 0: # ako koristimo cijenu < 0 da označimo da se dionicom ne može trgovati taj dan, izračun ukupne imovine može biti pogrešan jer je cijena nerazumna
                # Prodaj samo ako je cijena > 0 (nema nedostajućih podataka na ovaj određeni datum)
                # izvrši akciju prodaje na temelju predznaka akcije
                if self.state[index + self.stock_dim + 1] > 0:
                    # Prodaj samo ako je trenutna imovina > 0
                    sell_num_shares = min(
                        abs(action), self.state[index + self.stock_dim + 1]
                    )
                    sell_amount = (
                        self.state[index + 1]
                        * sell_num_shares
                        * (1 - self.sell_cost_pct[index])
                    )

                    # Ažuriraj stanje
                    self.state[0] += sell_amount

                    self.state[index + self.stock_dim + 1] -= sell_num_shares
                    self.cost += (
                        self.state[index + 1]
                        * sell_num_shares
                        * self.sell_cost_pct[index]
                    )
                    self.trades += 1
                else:
                    sell_num_shares = 0
            else:
                sell_num_shares = 0

            return sell_num_shares

        # Izvrši akciju prodaje na temelju predznaka akcije
        if self.turbulence_threshold is not None:
            if self.turbulence >= self.turbulence_threshold:
                if self.state[index + 1] > 0:
                    # Prodaj samo ako je cijena > 0 (nema nedostajućih podataka na ovaj određeni datum)
                    # Ako turbulencija prijeđe prag, jednostavno očisti sve pozicije
                    if self.state[index + self.stock_dim + 1] > 0:
                        # Prodaj samo ako je trenutna imovina > 0
                        sell_num_shares = self.state[index + self.stock_dim + 1]
                        sell_amount = (
                            self.state[index + 1]
                            * sell_num_shares
                            * (1 - self.sell_cost_pct[index])
                        )
                        # Ažuriraj stanje
                        self.state[0] += sell_amount
                        self.state[index + self.stock_dim + 1] = 0
                        self.cost += (
                            self.state[index + 1]
                            * sell_num_shares
                            * self.sell_cost_pct[index]
                        )
                        self.trades += 1
                    else:
                        sell_num_shares = 0
                else:
                    sell_num_shares = 0
            else:
                sell_num_shares = _do_sell_normal()
        else:
            sell_num_shares = _do_sell_normal()

        return sell_num_shares

    def _buy_stock(self, index, action):
        """
        Izvršava kupnju dionica.

        Parametri:
        - index (int): Indeks dionice koja se kupuje
        - action (int): Količina dionica za kupnju (pozitivna vrijednost)

        Vraća:
        - int: Stvarni broj kupljenih dionica
        """
        def _do_buy():
            if (self.state[index + 2 * self.stock_dim + 1] != True):
                # Provjera je li dionica dostupna za kupnju
                # if self.state[index + 1] >0:
                # Kupi samo ako je cijena > 0 (nema nedostajućih podataka na ovaj određeni datum)
                available_amount = self.state[0] // (
                    self.state[index + 1] * (1 + self.buy_cost_pct[index])
                )
                # Pri kupnji dionica, trebamo uzeti u obzir trošak trgovanja pri izračunu dostupnog iznosa, inače bismo mogli imati gotovinu < 0
                # print('available_amount:{}'.format(available_amount))

                # Ažuriraj stanje
                buy_num_shares = min(available_amount, action)
                buy_amount = (
                    self.state[index + 1]
                    * buy_num_shares
                    * (1 + self.buy_cost_pct[index])
                )
                self.state[0] -= buy_amount

                self.state[index + self.stock_dim + 1] += buy_num_shares

                self.cost += (
                    self.state[index + 1] * buy_num_shares * self.buy_cost_pct[index]
                )
                self.trades += 1
            else:
                buy_num_shares = 0

            return buy_num_shares

        # Izvrši akciju kupnje na temelju predznaka akcije
        if self.turbulence_threshold is None:
            buy_num_shares = _do_buy()
        else:
            if self.turbulence < self.turbulence_threshold:
                buy_num_shares = _do_buy()
            else:
                buy_num_shares = 0
                pass

        if buy_num_shares > 0:
            self.buy_prices[index] = self.state[index + 1]  # Zapamtite cijenu kupnje

        return buy_num_shares

    def _make_plot(self):
        """
        Generira i sprema graf vrijednosti računa tijekom trgovanja.
        """ 
        plt.plot(self.asset_memory, "r")
        plt.savefig(f"results/account_value_trade_{self.episode}.png")
        plt.close()

    def _calculate_dynamic_cash_penalty(self):
        # Izračun tržišnog trenda
        returns = self.df['close'].pct_change().fillna(0)
        market_trend = returns.rolling(self.market_trend_window, min_periods=1).mean().iloc[self.day]

        # Ako je market_trend NaN, koristimo prosječni return zadnjih dostupnih dana
        if np.isnan(market_trend):
            available_returns = returns.iloc[max(0, self.day - self.market_trend_window + 1):self.day + 1]
            market_trend = np.mean(available_returns)

        # Bazna vrijednost cash penalty-a
        base_penalty = 0.05

        # Finija gradacija prilagodbe prema tržišnom trendu
        if market_trend > 0.004:
            trend_adjustment = 0.05
        elif market_trend > 0.003:
            trend_adjustment = 0.04
        elif market_trend > 0.002:
            trend_adjustment = 0.03
        elif market_trend > 0.001:
            trend_adjustment = 0.02
        elif market_trend > 0:
            trend_adjustment = 0.01
        elif market_trend > -0.001:
            trend_adjustment = 0
        elif market_trend > -0.002:
            trend_adjustment = -0.01
        elif market_trend > -0.003:
            trend_adjustment = -0.02
        elif market_trend > -0.004:
            trend_adjustment = -0.03
        else:
            trend_adjustment = -0.04

        # Konačni izračun
        final_penalty = base_penalty + trend_adjustment

        # Ograničavanje konačne vrijednosti
        return np.clip(final_penalty, 0.02, 0.1)

    def _check_stop_loss_take_profit(self):
        for i in range(self.stock_dim):
            current_price = self.state[i + 1]
            buy_price = self.buy_prices[i]
            if buy_price > 0:  # Ako imamo poziciju u ovoj dionici
                if current_price <= buy_price * (1 - self.stop_loss_pct):
                    # Stop-loss aktiviran
                    self._sell_stock(i, self.state[i + self.stock_dim + 1])
                    self.buy_prices[i] = 0
                elif current_price >= buy_price * (1 + self.take_profit_pct):
                    # Take-profit aktiviran
                    self._sell_stock(i, self.state[i + self.stock_dim + 1])
                    self.buy_prices[i] = 0

    def log_step(self, action, reward, done):
        self.episode_history.append({
            'day': self.day,
            'action': action,
            'reward': reward,
            'portfolio_value': self.state[0] + sum(np.array(self.state[1:(self.stock_dim+1)]) * np.array(self.state[(self.stock_dim+1):(self.stock_dim*2+1)])),
            'cash': self.state[0],
            'holdings': self.state[(self.stock_dim+1):(self.stock_dim*2+1)],
            'done': done
        })

    def print_episode_summary(self):
        print(f"\n=== Sažetak epizode {self.episode} ===")
        print(f"Početna vrijednost: {self.episode_history[0]['portfolio_value']:.2f}")
        print(f"Završna vrijednost: {self.episode_history[-1]['portfolio_value']:.2f}")
        print(f"Ukupna nagrada: {sum(step['reward'] for step in self.episode_history):.2f}")
        print(f"Ukupan broj trgovanja: {self.trades}")
        print(f"Ukupni troškovi: {self.cost:.2f}")
        
        # Analiza akcija
        actions = np.array([step['action'] for step in self.episode_history])
        print(f"Prosječna akcija: {np.mean(actions):.4f}")
        
        # Analiza nagrade
        rewards = np.array([step['reward'] for step in self.episode_history])
        print(f"Prosječna nagrada: {np.mean(rewards):.4f}")
        print(f"Standardna devijacija nagrade: {np.std(rewards):.4f}")
        print(f"Min nagrada: {np.min(rewards):.4f}, Max nagrada: {np.max(rewards):.4f}")

        print("========================\n")

    def step(self, actions):
        """
        Izvršava jedan korak u okruženju, što uključuje trgovanje i ažuriranje stanja.

        Parametri:
        - actions (np.array): Akcije koje agent poduzima (kupnja/prodaja dionica)

        Vraća:
        - tuple: (novo stanje, nagrada, je li epizoda završena, dodatne informacije)
        """ 
        self.terminal = self.day >= len(self.df.index.unique()) - 1
        if self.terminal:
            # print(f"Episode: {self.episode}")
            if self.make_plots:
                self._make_plot()
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
            )
            # Početni iznos je samo gotovinski dio naše početne imovine
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
            # if self.episode % self.print_verbosity == 0:
            #     print(f"day: {self.day}, episode: {self.episode}")
            #     print(f"begin_total_asset: {self.asset_memory[0]:0.2f}")
            #     print(f"end_total_asset: {end_total_asset:0.2f}")
            #     print(f"total_reward: {tot_reward:0.2f}")
            #     print(f"total_cost: {self.cost:0.2f}")
            #     print(f"total_trades: {self.trades}")
            #     if df_total_value["daily_return"].std() != 0:
            #         print(f"Sharpe: {sharpe:0.3f}")
            #     print("=================================")

            if (self.model_name != "") and (self.mode != ""):
                df_actions = self.save_action_memory()
                df_actions.to_csv(
                    "results/actions_{}_{}_{}.csv".format(
                        self.mode, self.model_name, self.iteration
                    )
                )
                df_total_value.to_csv(
                    "results/account_value_{}_{}_{}.csv".format(
                        self.mode, self.model_name, self.iteration
                    ),
                    index=False,
                )
                df_rewards.to_csv(
                    "results/account_rewards_{}_{}_{}.csv".format(
                        self.mode, self.model_name, self.iteration
                    ),
                    index=False,
                )
                plt.plot(self.asset_memory, "r")
                plt.savefig(
                    "results/account_value_{}_{}_{}.png".format(
                        self.mode, self.model_name, self.iteration
                    )
                )
                plt.close()

            if self.terminal:
                self.print_episode_summary()
            return self.state, self.reward, self.terminal, False, {}

        else:
            # Akcije su inicijalno skalirane između 0 i 1
            actions = actions * self.hmax
            # Pretvori u cijele brojeve jer ne možemo kupiti ili prodati frakciju dionica
            actions = actions.astype(int)
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

            # Provjerite stop-loss i take-profit uvjete
            self._check_stop_loss_take_profit()

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

            # Ažuriranje cash_penalty_proportion prije izračuna nagrade
            self.cash_penalty_proportion = self._calculate_dynamic_cash_penalty()

            # Izračun cash penalty
            cash = self.state[0]
            cash_penalty = max(0, (end_total_asset * self.cash_penalty_proportion - cash))

            # Primjena cash penalty na ukupnu imovinu
            penalized_total_asset = end_total_asset - cash_penalty     

            self.asset_memory.append(penalized_total_asset)
            self.date_memory.append(self._get_date())

            
            self.reward = penalized_total_asset - begin_total_asset
            self.rewards_memory.append(self.reward)

            self.reward = self.reward * self.reward_scaling
            self.state_memory.append(
                self.state
            )  # add current state in state_recorder for each step

            self.log_step(actions, self.reward, self.terminal)
            return self.state, self.reward, self.terminal, False, {}

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):
        """
        Resetira okruženje na početno stanje.

        Parametri:
        - seed (int, optional): Sjeme za generator slučajnih brojeva
        - options (dict, optional): Dodatne opcije za resetiranje

        Vraća:
        - np.array: Početno stanje okruženja
        - dict: Dodatne informacije (prazan rječnik u ovom slučaju)
        """
        self.day = 0
        self.data = self.df.loc[self.day, :]
        self.state = self._initiate_state()
        self.buy_prices = np.zeros(self.stock_dim)

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

        self.episode_history = []
        return self.state, {}

    def render(self, mode="human", close=False):
        """
        Prikazuje trenutno stanje okruženja.

        Parametri:
        - mode (str): Način prikaza (trenutno nije implementirano)
        - close (bool): Treba li zatvoriti prikaz (trenutno nije implementirano)

        Vraća:
        - str: Opis trenutnog stanja
        """
        return self.state

    def _initiate_state(self):
        """
        Inicijalizira početno stanje okruženja.

        Vraća:
        - list: Početno stanje koje uključuje gotovinu, cijene dionica, broj dionica i tehničke indikatore
        """
        if self.initial:
            # Za početno stanje
            if len(self.df.tic.unique()) > 1:
                # Za više dionica
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
                )  # Dodaj početne dionice_share u početno stanje, umjesto svih nula
            else:
                # Za jednu dionicu
                state = (
                    [self.initial_amount]
                    + [self.data.close]
                    + [0] * self.stock_dim
                    + sum(([self.data[tech]] for tech in self.tech_indicator_list), [])
                )
        else:
            # Koristi prethodno stanje
            if len(self.df.tic.unique()) > 1:
                # Za više dionica
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
                )
            else:
                # Za jednu dionicu
                state = (
                    [self.previous_state[0]]
                    + [self.data.close]
                    + self.previous_state[
                        (self.stock_dim + 1) : (self.stock_dim * 2 + 1)
                    ]
                    + sum(([self.data[tech]] for tech in self.tech_indicator_list), [])
                )
        return state

    def _update_state(self):
        """
        Ažurira trenutno stanje okruženja.

        Vraća:
        - list: Ažurirano stanje koje uključuje gotovinu, cijene dionica, broj dionica i tehničke indikatore
        """
        if len(self.df.tic.unique()) > 1:
            # Za više dionica
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
            )

        else:
            # Za jednu dionicu
            state = (
                [self.state[0]]
                + [self.data.close]
                + list(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
                + sum(([self.data[tech]] for tech in self.tech_indicator_list), [])
            )

        return state

    def _get_date(self):
        """
        Dohvaća trenutni datum iz podataka.

        Vraća:
        - str: Trenutni datum
        """
        if len(self.df.tic.unique()) > 1:
            date = self.data.date.unique()[0]
        else:
            date = self.data.date
        return date

    def save_state_memory(self):
        """
        Sprema stanje u memoriju tijekom trgovanja.
        """
        if len(self.df.tic.unique()) > 1:
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
        """
        Sprema povijest vrijednosti imovine u DataFrame.

        Vraća:
        - pd.DataFrame: DataFrame s povijesti vrijednosti imovine
        """
        date_list = self.date_memory
        asset_list = self.asset_memory
        # print(len(date_list))
        # print(len(asset_list))
        df_account_value = pd.DataFrame(
            {"date": date_list, "account_value": asset_list}
        )
        return df_account_value

    def save_action_memory(self):
        """
        Sprema povijest akcija u DataFrame.

        Vraća:
        - pd.DataFrame: DataFrame s povijesti akcija
        """
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
        """
        Postavlja sjeme za generator slučajnih brojeva.

        Parametri:
        - seed (int, optional): Sjeme za generator slučajnih brojeva

        Vraća:
        - list: Lista s postavljenim sjemenom
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_sb_env(self):
        """
        Vraća okruženje kompatibilno sa Stable Baselines bibliotekom.

        Vraća:
        - tuple: (funkcija za stvaranje okruženja, str: ime okruženja)
        """
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs
