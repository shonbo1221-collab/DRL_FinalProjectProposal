import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class TradingEnv(gym.Env):
    """
    Custom Environment that follows gym interface for 0050 trading.
    """
    metadata = {'render.modes': ['human']}

    def __init__(
        self,
        df,
        initial_balance=1000000,
        fee=0.001425,
        tax=0.001,
        mdd_penalty=0.1,
        turnover_penalty=0.001,
        max_position=0.8
    ):
        super(TradingEnv, self).__init__()
        
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.fee = fee
        self.tax = tax
        self.mdd_penalty = mdd_penalty
        self.turnover_penalty = turnover_penalty
        self.max_position = max_position
        
        # Action space: raw PPO action [-1, 1], mapped to target weight [0, max_position].
        # This makes the initial PPO policy around 0 correspond to half of max_position.
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # State space dimensions:
        # 0: pd_ratio
        # 1: dist_to_bull_fvg
        # 2: dist_to_bear_fvg
        # 3: dist_to_bull_ob
        # 4: return_5d
        # 5: atr_20
        # 6: dev_ma_20
        # 7: dev_ma_60
        # 8: current_weight
        # 9: current_cash_ratio (1 - current_weight)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
        
        self.feature_cols = [
            'pd_ratio', 'dist_to_bull_fvg', 'dist_to_bear_fvg', 'dist_to_bull_ob',
            'return_5d', 'atr_20', 'dev_ma_20', 'dev_ma_60'
        ]
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.current_weight = 0.0
        self.prev_drawdown = 0.0
        
        self.history = []
        
        return self._next_observation(), {}

    def _next_observation(self):
        obs = self.df.loc[self.current_step, self.feature_cols].values
        obs = np.append(obs, [self.current_weight, 1.0 - self.current_weight])
        return obs.astype(np.float32)

    def step(self, action):
        raw_action = np.clip(action[0], -1.0, 1.0)
        target_weight = ((raw_action + 1.0) / 2.0) * self.max_position
        
        current_price = self.df.loc[self.current_step, 'close']
        
        # 1. Update net worth based on current price
        self.net_worth = self.balance + self.shares_held * current_price
        
        # 2. Calculate target value to hold in 0050
        target_value = self.net_worth * target_weight
        current_value = self.shares_held * current_price
        prev_weight = current_value / self.net_worth if self.net_worth > 0 else 0.0
        
        # 3. Execute trade
        value_diff = target_value - current_value
        
        transaction_cost = 0
        if value_diff > 0: # Buy
            max_buy_value = max(0.0, self.balance / (1.0 + self.fee))
            buy_value = min(value_diff, max_buy_value)
            shares_to_buy = buy_value / current_price
            cost = buy_value * self.fee
            transaction_cost = cost
            self.balance -= (buy_value + cost)
            self.shares_held += shares_to_buy
        elif value_diff < 0: # Sell
            shares_to_sell = -value_diff / current_price
            cost = -value_diff * (self.fee + self.tax)
            transaction_cost = cost
            self.balance += (-value_diff - cost)
            self.shares_held -= shares_to_sell
            
        # Update current weight after trade
        new_net_worth = self.balance + self.shares_held * current_price
        self.current_weight = (self.shares_held * current_price) / new_net_worth if new_net_worth > 0 else 0
        
        # Calculate Reward
        # Log return
        if self.current_step == 0:
            log_return = 0
        else:
            prev_net_worth = self.history[-1]['net_worth']
            log_return = np.log(new_net_worth / prev_net_worth) if prev_net_worth > 0 else 0
            
        # MDD Penalty
        self.max_net_worth = max(self.max_net_worth, new_net_worth)
        drawdown = (self.max_net_worth - new_net_worth) / self.max_net_worth if self.max_net_worth > 0 else 0
        drawdown_increase = max(0.0, drawdown - self.prev_drawdown)
        mdd_penalty_val = self.mdd_penalty * drawdown_increase
        self.prev_drawdown = drawdown
        
        # Final Reward
        turnover = abs(target_weight - prev_weight)
        reward = log_return - mdd_penalty_val - (self.turnover_penalty * turnover)
        
        # Move to next step
        self.current_step += 1
        terminated = self.current_step >= len(self.df) - 1
        truncated = False
        
        self.history.append({
            'step': self.current_step,
            'date': self.df.loc[self.current_step-1, 'date'],
            'net_worth': new_net_worth,
            'weight': self.current_weight,
            'action': target_weight,
            'raw_action': raw_action,
            'turnover': turnover,
            'reward': reward,
            'close': current_price
        })
        
        if terminated:
            obs = np.zeros(self.observation_space.shape)
        else:
            obs = self._next_observation()
            
        info = {
            'net_worth': new_net_worth,
            'drawdown': drawdown
        }
        
        return obs, reward, terminated, truncated, info

if __name__ == "__main__":
    from stable_baselines3.common.env_checker import check_env
    # We will run this check once we have data
    print("Environment defined.")
