import os
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import pandas as pd
import yfinance as yf
from gymnasium import spaces


BROKERAGE_FEE = 0.001425 * 0.6
PRICE_LIMIT_UP = 1.10
SHARPE_WINDOW = 20
EPS = 1e-8


@dataclass(frozen=True)
class ModelConfig:
    name: str
    tickers: tuple[str, ...]
    model_path: str
    mode: str


PAIR_CONFIG = ModelConfig(
    name="pair_0050_2330",
    tickers=("0050.TW", "2330.TW"),
    model_path="model/saved/ppo_model_pair.zip",
    mode="pair",
)

BASKET_CONFIG = ModelConfig(
    name="basket_0050_2330_2412",
    tickers=("0050.TW", "2330.TW", "2412.TW"),
    model_path="model/saved/ppo_model_basket.zip",
    mode="basket",
)

SELL_TAX = {
    "0050.TW": 0.001,
    "2330.TW": 0.003,
    "2412.TW": 0.003,
}


def download_ohlcv(tickers, start_date, end_date):
    frames = {}
    for ticker in tickers:
        df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False, progress=False)
        if df.empty:
            raise ValueError(f"No data downloaded for {ticker}")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index()
        df = df.rename(
            columns={
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "adj_close",
                "Volume": "volume",
            }
        )
        required = ["date", "open", "high", "low", "close", "volume"]
        frames[ticker] = df[required].dropna().reset_index(drop=True)
    return frames


def add_smc_features(df):
    out = df.copy()
    high_20 = out["high"].rolling(20).max()
    low_20 = out["low"].rolling(20).min()
    dealing_range = (high_20 - low_20).replace(0, np.nan)
    out["PD_Pos"] = ((out["close"] - low_20) / dealing_range).clip(0.0, 1.0)

    ret_3d = out["close"].pct_change(3)
    bearish_candle = out["close"] < out["open"]
    ob_level = np.nan
    ob_levels = []
    for i in range(len(out)):
        if i >= 3 and ret_3d.iloc[i] > 0.03:
            start = max(0, i - 20)
            for j in range(i - 3, start - 1, -1):
                if bearish_candle.iloc[j]:
                    ob_level = out["high"].iloc[j]
                    break
        ob_levels.append(ob_level)
    out["ob_level"] = pd.Series(ob_levels, index=out.index).ffill()
    out["OB_Dist"] = ((out["close"] - out["ob_level"]) / out["close"]).replace([np.inf, -np.inf], np.nan)

    bullish_fvg = out["low"] > out["high"].shift(2)
    bearish_fvg = out["high"] < out["low"].shift(2)
    bull_low = out["high"].shift(2)
    bull_high = out["low"]
    bear_low = out["high"]
    bear_high = out["low"].shift(2)

    unfilled = np.zeros(len(out), dtype=np.float32)
    active_gaps = []
    for i in range(len(out)):
        still_active = []
        for gap in active_gaps:
            touches_gap = out["low"].iloc[i] <= gap["high"] and out["high"].iloc[i] >= gap["low"]
            if not touches_gap:
                still_active.append(gap)
        active_gaps = still_active

        if bullish_fvg.iloc[i]:
            active_gaps.append({"low": bull_low.iloc[i], "high": bull_high.iloc[i], "direction": "bull"})
        if bearish_fvg.iloc[i]:
            active_gaps.append({"low": bear_low.iloc[i], "high": bear_high.iloc[i], "direction": "bear"})

        unfilled[i] = 1.0 if active_gaps else 0.0
    out["FVG_Signal"] = unfilled

    out[["PD_Pos", "OB_Dist", "FVG_Signal"]] = out[["PD_Pos", "OB_Dist", "FVG_Signal"]].fillna(
        {"PD_Pos": 0.5, "OB_Dist": 0.0, "FVG_Signal": 0.0}
    )
    return out


def build_feature_frame(raw_frames, config):
    feature_frames = []
    for ticker in config.tickers:
        df = add_smc_features(raw_frames[ticker])
        prefix = ticker.replace(".", "_")
        cols = ["date", "open", "high", "low", "close", "volume", "PD_Pos", "OB_Dist", "FVG_Signal"]
        renamed = df[cols].rename(columns={col: f"{prefix}_{col}" for col in cols if col != "date"})
        feature_frames.append(renamed)

    merged = feature_frames[0]
    for frame in feature_frames[1:]:
        merged = pd.merge(merged, frame, on="date", how="inner")

    if config.mode in {"pair", "basket"}:
        spread_cols = []
        pairs = [(config.tickers[0], config.tickers[1])] if config.mode == "pair" else [
            (config.tickers[i], config.tickers[j])
            for i in range(len(config.tickers))
            for j in range(i + 1, len(config.tickers))
        ]
        for left_ticker, right_ticker in pairs:
            left = left_ticker.replace(".", "_")
            right = right_ticker.replace(".", "_")
            col = f"Spread_ZScore_{left}_{right}"
            spread_cols.append(col)
            spread = np.log(merged[f"{left}_close"]) - np.log(merged[f"{right}_close"])
            mean = spread.rolling(20).mean()
            std = spread.rolling(20).std().replace(0, np.nan)
            merged[col] = ((spread - mean) / std).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if config.mode == "pair":
        left = config.tickers[0].replace(".", "_")
        right = config.tickers[1].replace(".", "_")
        pair_col = f"Spread_ZScore_{left}_{right}"
        merged["Spread_ZScore"] = merged[pair_col]
    elif config.mode == "basket":
        merged["Spread_ZScore"] = 0.0
    else:
        merged["Spread_ZScore"] = 0.0

    # Avoid same-bar leakage: decisions at row t can only use market features
    # computed from information available through row t-1. Raw OHLCV columns
    # remain unshifted because they are used by the environment for execution
    # and mark-to-market accounting at row t.
    feature_cols = [
        col
        for col in merged.columns
        if any(token in col for token in ("PD_Pos", "OB_Dist", "FVG_Signal", "Spread_ZScore"))
    ]
    merged[feature_cols] = merged[feature_cols].shift(1)

    merged = merged.dropna().reset_index(drop=True)
    return merged


class TWTradingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, feature_df, config, initial_balance=1_000_000.0, max_position=0.95):
        super().__init__()
        self.df = feature_df.reset_index(drop=True)
        self.config = config
        self.initial_balance = float(initial_balance)
        self.max_position = float(max_position)
        self.asset_prefixes = [ticker.replace(".", "_") for ticker in config.tickers]
        self.feature_cols = self._make_feature_cols()

        obs_dim = len(self.feature_cols) + len(self.config.tickers) + 1
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        action_dim = len(self.config.tickers) if self.config.mode == "basket" else 1
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)

    def _make_feature_cols(self):
        cols = []
        for prefix in self.asset_prefixes:
            cols.extend([f"{prefix}_PD_Pos", f"{prefix}_OB_Dist", f"{prefix}_FVG_Signal"])
        if self.config.mode == "pair":
            cols.append("Spread_ZScore")
        elif self.config.mode == "basket":
            cols.extend([col for col in self.df.columns if col.startswith("Spread_ZScore_")])
        return cols

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.cash = self.initial_balance
        self.shares = np.zeros(len(self.config.tickers), dtype=np.float64)
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.prev_drawdown = 0.0
        self.portfolio_returns = []
        self.history = []
        return self._get_obs(), {}

    def _prices(self, step=None):
        idx = self.current_step if step is None else step
        return np.array([self.df.loc[idx, f"{prefix}_close"] for prefix in self.asset_prefixes], dtype=np.float64)

    def _prev_closes(self):
        prev_step = max(0, self.current_step - 1)
        return np.array([self.df.loc[prev_step, f"{prefix}_close"] for prefix in self.asset_prefixes], dtype=np.float64)

    def _get_weights(self, prices=None, net_worth=None):
        prices = self._prices() if prices is None else prices
        net_worth = self.net_worth if net_worth is None else net_worth
        if net_worth <= 0:
            return np.zeros(len(self.config.tickers), dtype=np.float64)
        return (self.shares * prices) / net_worth

    def _get_obs(self):
        market_features = self.df.loc[self.current_step, self.feature_cols].to_numpy(dtype=np.float32)
        prices = self._prices()
        net_worth = self.cash + float(np.dot(self.shares, prices))
        weights = self._get_weights(prices, net_worth).astype(np.float32)
        cash_ratio = np.array([self.cash / max(net_worth, EPS)], dtype=np.float32)
        return np.concatenate([market_features, weights, cash_ratio]).astype(np.float32)

    def _target_weights(self, raw_action):
        action_array = np.asarray(raw_action, dtype=np.float64).reshape(-1)
        if self.config.mode == "pair":
            action = float(np.clip(action_array[0], -1.0, 1.0))
            weights = np.zeros(2, dtype=np.float64)
            if action >= 0:
                weights[0] = action * self.max_position
            else:
                weights[1] = -action * self.max_position
            return weights

        if self.config.mode == "basket":
            scores = np.clip(action_array[: len(self.config.tickers)], 0.0, 1.0)
            if scores.sum() <= EPS:
                return np.zeros(len(self.config.tickers), dtype=np.float64)
            return (scores / scores.sum()) * self.max_position

        action = float(np.clip(action_array[0], -1.0, 1.0))
        return np.array([max(0.0, action) * self.max_position], dtype=np.float64)

    def _execute_rebalance(self, target_weights, prices):
        prev_prices = self._prev_closes()
        net_worth_before_trade = self.cash + float(np.dot(self.shares, prices))
        current_values = self.shares * prices
        target_values = target_weights * net_worth_before_trade
        value_diffs = target_values - current_values
        transaction_cost = 0.0
        failed_orders = 0

        for idx, value_diff in enumerate(value_diffs):
            ticker = self.config.tickers[idx]
            price = prices[idx]
            limit_up = prev_prices[idx] * PRICE_LIMIT_UP

            if abs(value_diff) > EPS and price >= limit_up:
                failed_orders += 1
                continue

            if value_diff > 0:
                buy_value = min(value_diff, max(0.0, self.cash / (1.0 + BROKERAGE_FEE)))
                fee = buy_value * BROKERAGE_FEE
                self.cash -= buy_value + fee
                self.shares[idx] += buy_value / price
                transaction_cost += fee
            elif value_diff < 0:
                sell_value = min(-value_diff, self.shares[idx] * price)
                fee = sell_value * BROKERAGE_FEE
                tax = sell_value * SELL_TAX[ticker]
                self.cash += sell_value - fee - tax
                self.shares[idx] -= sell_value / price
                transaction_cost += fee + tax

        return transaction_cost, failed_orders

    def step(self, action):
        raw_action_array = np.asarray(action, dtype=np.float64).reshape(-1)
        raw_action = float(raw_action_array[0])
        prices = self._prices()
        prev_net_worth = self.net_worth
        target_weights = self._target_weights(raw_action_array)
        transaction_cost, failed_orders = self._execute_rebalance(target_weights, prices)

        self.net_worth = self.cash + float(np.dot(self.shares, prices))
        pnl = self.net_worth - prev_net_worth
        pnl_rate = pnl / max(prev_net_worth, EPS)
        self.portfolio_returns.append(pnl_rate)

        recent = np.array(self.portfolio_returns[-SHARPE_WINDOW:], dtype=np.float64)
        if len(recent) > 1 and np.std(recent) > EPS:
            sharpe_adjustment = max(0.0, np.mean(recent) / (np.std(recent) + EPS) * np.sqrt(252))
        else:
            sharpe_adjustment = 1.0

        self.max_net_worth = max(self.max_net_worth, self.net_worth)
        drawdown = (self.max_net_worth - self.net_worth) / max(self.max_net_worth, EPS)
        drawdown_penalty = max(0.0, drawdown - self.prev_drawdown)
        self.prev_drawdown = drawdown

        cost_rate = transaction_cost / max(prev_net_worth, EPS)
        reward = (pnl_rate * sharpe_adjustment) - cost_rate - (2.0 * drawdown_penalty)

        weights = self._get_weights(prices, self.net_worth)
        self.history.append(
            {
                "date": self.df.loc[self.current_step, "date"],
                "raw_action": raw_action,
                "net_worth": self.net_worth,
                "pnl": pnl,
                "pnl_rate": pnl_rate,
                "transaction_cost": transaction_cost,
                "drawdown": drawdown,
                "reward": reward,
                "failed_orders": failed_orders,
                **{f"raw_action_{ticker}": raw_action_array[i] for i, ticker in enumerate(self.config.tickers) if i < len(raw_action_array)},
                **{f"weight_{ticker}": weights[i] for i, ticker in enumerate(self.config.tickers)},
            }
        )

        self.current_step += 1
        terminated = self.current_step >= len(self.df) - 1
        truncated = False
        obs = np.zeros(self.observation_space.shape, dtype=np.float32) if terminated else self._get_obs()
        info = {"net_worth": self.net_worth, "drawdown": drawdown, "transaction_cost": transaction_cost}
        return obs, float(reward), terminated, truncated, info


def calculate_metrics(history_df, initial_balance=1_000_000.0):
    if history_df.empty:
        return {"cumulative_return": 0.0, "sharpe_ratio": 0.0, "max_drawdown": 0.0}
    net_worth = history_df["net_worth"].astype(float)
    cumulative_return = (net_worth.iloc[-1] / initial_balance) - 1.0
    returns = net_worth.pct_change().dropna()
    sharpe = 0.0
    if len(returns) > 1 and returns.std() > EPS:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
    roll_max = net_worth.cummax()
    drawdown = (net_worth - roll_max) / roll_max
    return {
        "cumulative_return": float(cumulative_return),
        "sharpe_ratio": float(sharpe),
        "max_drawdown": float(drawdown.min()),
    }


def prepare_dataset(config, start_date, end_date):
    raw_frames = download_ohlcv(config.tickers, start_date, end_date)
    return build_feature_frame(raw_frames, config)


def ensure_model_dir(path):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
