import pandas as pd
from stable_baselines3 import PPO

from twse_pipeline_common import (
    BASKET_CONFIG,
    PAIR_CONFIG,
    TWTradingEnv,
    calculate_metrics,
    prepare_dataset,
)


PREDICT_START = "2025-01-01"
PREDICT_END = "2026-05-01"
INITIAL_BALANCE = 1_000_000.0


def run_inference(config):
    print(f"\n=== Predicting {config.name} ===")
    print(f"Tickers: {', '.join(config.tickers)}")
    print(f"Period: {PREDICT_START} to {PREDICT_END}")

    feature_df = prepare_dataset(config, PREDICT_START, PREDICT_END)
    env = TWTradingEnv(
        feature_df=feature_df,
        config=config,
        initial_balance=INITIAL_BALANCE,
    )
    model = PPO.load(config.model_path)

    obs, _ = env.reset()
    done = False
    action_logs = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        raw_action_values = action.reshape(-1).astype(float)
        current_date = env.df.loc[env.current_step, "date"]
        log_row = {"date": current_date, "action": float(raw_action_values[0])}
        for idx, ticker in enumerate(config.tickers):
            if idx < len(raw_action_values):
                log_row[f"action_{ticker}"] = float(raw_action_values[idx])
        action_logs.append(log_row)

        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    history = pd.DataFrame(env.history)
    actions = pd.DataFrame(action_logs)
    metrics = calculate_metrics(history, initial_balance=INITIAL_BALANCE)

    print(f"Final net worth: {history['net_worth'].iloc[-1]:,.0f}")
    print(f"Cumulative Return: {metrics['cumulative_return']:.2%}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
    print("Last 5 deterministic actions:")
    print(actions.tail(5).to_string(index=False))

    return {"config": config, "history": history, "actions": actions, "metrics": metrics}


def print_comparison(results):
    rows = []
    for result in results:
        metrics = result["metrics"]
        rows.append(
            {
                "model": result["config"].name,
                "cumulative_return": f"{metrics['cumulative_return']:.2%}",
                "sharpe_ratio": f"{metrics['sharpe_ratio']:.2f}",
                "max_drawdown": f"{metrics['max_drawdown']:.2%}",
            }
        )
    print("\n=== Out-of-sample comparison ===")
    print(pd.DataFrame(rows).to_string(index=False))


def main():
    results = [run_inference(PAIR_CONFIG), run_inference(BASKET_CONFIG)]
    print_comparison(results)


if __name__ == "__main__":
    main()
