import os
import sys
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Ensure modules can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.downloader import DataDownloader
from features.builder import FeatureBuilder
from env.trading_env import TradingEnv

TOTAL_TIMESTEPS = 200000
MDD_PENALTY = 0.1
TURNOVER_PENALTY = 0.001
MAX_POSITION = 0.8

def train_model():
    print("1. Downloading Data...")
    downloader = DataDownloader(ticker="0050.TW", data_dir="data/raw")
    # Use 2014-2020 as training data, then evaluate on 2021-2024.
    df_raw = downloader.download(start_date="2014-01-01", end_date="2020-12-31")
    
    print("2. Building Features...")
    builder = FeatureBuilder(df_raw)
    df_features = builder.build()
    
    print("3. Setting up Environment...")
    # Wrap and normalize environment for PPO stability.
    env = DummyVecEnv([lambda: TradingEnv(
        df_features,
        mdd_penalty=MDD_PENALTY,
        turnover_penalty=TURNOVER_PENALTY,
        max_position=MAX_POSITION
    )])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    
    print("4. Initializing PPO Model...")
    # Define network architecture [64, 64]
    policy_kwargs = dict(net_arch=[dict(pi=[64, 64], vf=[64, 64])])
    
    model = PPO(
        "MlpPolicy", 
        env, 
        learning_rate=2e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        policy_kwargs=policy_kwargs,
        target_kl=0.03,
        verbose=1
    )
    
    print("5. Starting Training...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    
    print("6. Saving Model...")
    os.makedirs("model/saved", exist_ok=True)
    model.save("model/saved/ppo_smc_0050")
    env.save("model/saved/ppo_smc_0050_vecnormalize.pkl")
    print("Model saved to model/saved/ppo_smc_0050.zip")
    print("VecNormalize stats saved to model/saved/ppo_smc_0050_vecnormalize.pkl")

if __name__ == "__main__":
    train_model()
