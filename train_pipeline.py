import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from twse_pipeline_common import (
    BASKET_CONFIG,
    PAIR_CONFIG,
    TWTradingEnv,
    ensure_model_dir,
    prepare_dataset,
)


TRAIN_START = "2018-01-01"
TRAIN_END = "2024-12-31"
TOTAL_TIMESTEPS = 500_000
INITIAL_BALANCE = 1_000_000.0
DEVICE = "cuda:0"


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is required for train_pipeline.py. "
            "Install a CUDA-enabled PyTorch build or run on a machine with an NVIDIA GPU."
        )
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True


def train_one_model(config):
    print(f"\n=== Training {config.name} ===")
    print(f"Tickers: {', '.join(config.tickers)}")
    print(f"Period: {TRAIN_START} to {TRAIN_END}")

    feature_df = prepare_dataset(config, TRAIN_START, TRAIN_END)
    print(f"Feature rows: {len(feature_df)}")

    env = DummyVecEnv(
        [
            lambda: TWTradingEnv(
                feature_df=feature_df,
                config=config,
                initial_balance=INITIAL_BALANCE,
            )
        ]
    )

    policy_kwargs = dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=2e-4,
        n_steps=2048,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        target_kl=0.03,
        policy_kwargs=policy_kwargs,
        verbose=1,
        device=DEVICE,
    )

    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    ensure_model_dir(config.model_path)
    model.save(config.model_path)
    print(f"Saved model: {config.model_path}")


def main():
    require_cuda()
    print(f"Using fixed training device: {DEVICE}")
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    train_one_model(PAIR_CONFIG)
    train_one_model(BASKET_CONFIG)


if __name__ == "__main__":
    main()
