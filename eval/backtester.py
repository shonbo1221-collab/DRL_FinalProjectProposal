import pandas as pd
import numpy as np
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from env.trading_env import TradingEnv
from features.builder import FeatureBuilder

class Backtester:
    def __init__(
        self,
        model_path,
        df_test,
        initial_balance=1000000,
        mdd_penalty=0.1,
        turnover_penalty=0.001,
        max_position=0.8
    ):
        self.model = PPO.load(model_path)
        self.df_test = df_test
        self.initial_balance = initial_balance
        self.raw_env = TradingEnv(
            self.df_test,
            initial_balance=initial_balance,
            mdd_penalty=mdd_penalty,
            turnover_penalty=turnover_penalty,
            max_position=max_position
        )
        self.vec_env = None

        normalize_path = f"{model_path}_vecnormalize.pkl"
        if os.path.exists(normalize_path):
            vec_env = DummyVecEnv([lambda: self.raw_env])
            self.vec_env = VecNormalize.load(normalize_path, vec_env)
            self.vec_env.training = False
            self.vec_env.norm_reward = False
            self.env = self.raw_env
        else:
            self.env = self.raw_env
        
    def run(self):
        if self.vec_env is not None:
            obs, _ = self.env.reset()
            done = False

            while not done:
                normalized_obs = self.vec_env.normalize_obs(obs.reshape(1, -1))
                action, _states = self.model.predict(normalized_obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.env.step(action[0])
                done = terminated or truncated

            history = pd.DataFrame(self.raw_env.history)
        else:
            obs, _ = self.env.reset()
            done = False

            while not done:
                action, _states = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

            history = pd.DataFrame(self.env.history)
        return history
        
    def calculate_metrics(self, history_df):
        initial_balance = self.initial_balance
        final_balance = history_df['net_worth'].iloc[-1]
        
        # 1. Total Return
        total_return = (final_balance / initial_balance) - 1
        
        # 2. Max Drawdown
        roll_max = history_df['net_worth'].cummax()
        drawdowns = (history_df['net_worth'] - roll_max) / roll_max
        max_drawdown = drawdowns.min()
        
        # 3. Sharpe Ratio (assuming daily data, risk free rate = 0 for simplicity)
        history_df['daily_return'] = history_df['net_worth'].pct_change()
        daily_volatility = history_df['daily_return'].std()
        annualized_return = (1 + total_return) ** (252 / len(history_df)) - 1
        annualized_volatility = daily_volatility * np.sqrt(252)
        
        sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else 0
        
        return {
            'total_return': float(total_return),
            'annualized_return': float(annualized_return),
            'max_drawdown': float(max_drawdown),
            'sharpe_ratio': float(sharpe_ratio)
        }

if __name__ == "__main__":
    from data.downloader import DataDownloader
    import os
    
    model_path = "model/saved/ppo_smc_0050"
    if os.path.exists(model_path + ".zip"):
        print("Running backtest...")
        downloader = DataDownloader(ticker="0050.TW", data_dir="data/raw")
        df_raw = downloader.download(start_date="2021-01-01", end_date="2024-01-01")
        
        builder = FeatureBuilder(df_raw)
        df_features = builder.build()
        
        backtester = Backtester(model_path, df_features)
        history = backtester.run()
        metrics = backtester.calculate_metrics(history)
        
        print(f"Total Return: {metrics['total_return']:.2%}")
        print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    else:
        print("Model not found. Please train first.")
