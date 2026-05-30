import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from stable_baselines3 import PPO

from twse_pipeline_common import (
    BASKET_CONFIG,
    PAIR_CONFIG,
    TWTradingEnv,
    calculate_metrics,
    prepare_dataset,
)


st.set_page_config(page_title="TWSE PPO + SMC Pipeline", layout="wide")


def run_twse_inference(config, start_date, end_date, initial_capital):
    if not os.path.exists(config.model_path):
        raise FileNotFoundError(f"Model not found: {config.model_path}")

    feature_df = prepare_dataset(config, start_date, end_date)
    env = TWTradingEnv(feature_df=feature_df, config=config, initial_balance=initial_capital)
    model = PPO.load(config.model_path)

    obs, _ = env.reset()
    done = False
    action_logs = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        action_values = action.reshape(-1).astype(float)
        log_row = {"date": env.df.loc[env.current_step, "date"], "action": float(action_values[0])}
        for idx, ticker in enumerate(config.tickers):
            if idx < len(action_values):
                log_row[f"action_{ticker}"] = float(action_values[idx])
        action_logs.append(log_row)

        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    history = pd.DataFrame(env.history)
    actions = pd.DataFrame(action_logs)
    metrics = calculate_metrics(history, initial_balance=initial_capital)
    return feature_df, history, actions, metrics


def plot_result(result):
    config = result["config"]
    history = result["history"]
    actions = result["actions"]
    metrics = result["metrics"]

    st.subheader(config.name)
    col1, col2, col3 = st.columns(3)
    col1.metric("Cumulative Return", f"{metrics['cumulative_return']:.2%}")
    col2.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
    col3.metric("Max Drawdown", f"{metrics['max_drawdown']:.2%}")

    fig_net_worth = go.Figure()
    fig_net_worth.add_trace(
        go.Scatter(
            x=history["date"],
            y=history["net_worth"],
            mode="lines",
            name="Net Worth",
            line=dict(width=3),
        )
    )
    fig_net_worth.update_layout(
        title=f"{config.name} Portfolio Net Worth",
        yaxis_title="NTD",
        height=360,
        template="plotly_dark",
    )
    st.plotly_chart(fig_net_worth, use_container_width=True)

    action_cols = [col for col in actions.columns if col.startswith("action_")]
    fig_action = go.Figure()
    if action_cols:
        for col in action_cols:
            fig_action.add_trace(
                go.Scatter(x=actions["date"], y=actions[col], mode="lines", name=col.replace("action_", ""))
            )
    else:
        fig_action.add_trace(
            go.Scatter(x=actions["date"], y=actions["action"], mode="lines", name="Action")
        )
    fig_action.update_layout(
        title=f"{config.name} Deterministic PPO Actions",
        yaxis_title="Action [-1, 1]",
        height=320,
        template="plotly_dark",
    )
    fig_action.update_yaxes(range=[-1.05, 1.05])
    st.plotly_chart(fig_action, use_container_width=True)

    weight_cols = [col for col in history.columns if col.startswith("weight_")]
    if weight_cols:
        fig_weights = go.Figure()
        for col in weight_cols:
            fig_weights.add_trace(
                go.Scatter(x=history["date"], y=history[col], mode="lines", name=col.replace("weight_", ""))
            )
        fig_weights.update_layout(
            title=f"{config.name} Executed Portfolio Weights",
            yaxis_title="Weight",
            height=320,
            template="plotly_dark",
        )
        st.plotly_chart(fig_weights, use_container_width=True)

    with st.expander(f"{config.name} latest inference rows"):
        display_cols = [
            col
            for col in ["date", "raw_action", "net_worth", "transaction_cost", "drawdown", "failed_orders"]
            if col in history.columns
        ]
        st.dataframe(history[display_cols].tail(20), use_container_width=True)


def main():
    st.title("TWSE PPO + SMC Training / Inference Verification")
    st.caption(
        "This app validates the separated TWSE training and inference pipeline."
    )

    col_a, col_b, col_c = st.columns(3)
    start_date = col_a.date_input("Inference Start", pd.to_datetime("2025-01-01"))
    end_date = col_b.date_input("Inference End", pd.to_datetime("2026-05-01"))
    initial_capital = col_c.number_input("Initial Capital", value=1_000_000, step=100_000)

    configs = (PAIR_CONFIG, BASKET_CONFIG)
    model_status = pd.DataFrame(
        [
            {
                "model": config.name,
                "tickers": ", ".join(config.tickers),
                "action_shape": len(config.tickers) if config.mode == "basket" else 1,
                "path": config.model_path,
                "exists": os.path.exists(config.model_path),
            }
            for config in configs
        ]
    )
    st.dataframe(model_status, use_container_width=True)

    if st.button("Run Out-of-Sample Inference", type="primary"):
        results = []
        for config in configs:
            try:
                with st.spinner(f"Running inference for {config.name}..."):
                    feature_df, history, actions, metrics = run_twse_inference(
                        config,
                        start_date.strftime("%Y-%m-%d"),
                        end_date.strftime("%Y-%m-%d"),
                        initial_capital,
                    )
                results.append(
                    {
                        "config": config,
                        "feature_df": feature_df,
                        "history": history,
                        "actions": actions,
                        "metrics": metrics,
                    }
                )
            except FileNotFoundError as exc:
                st.error(str(exc))
                st.info("Run `python train_pipeline.py` first, then return to this page.")
                return
            except Exception as exc:
                st.exception(exc)
                return

        comparison = pd.DataFrame(
            [
                {
                    "model": result["config"].name,
                    "tickers": ", ".join(result["config"].tickers),
                    "cumulative_return": result["metrics"]["cumulative_return"],
                    "sharpe_ratio": result["metrics"]["sharpe_ratio"],
                    "max_drawdown": result["metrics"]["max_drawdown"],
                }
                for result in results
            ]
        )
        st.subheader("Out-of-Sample Comparison")
        st.dataframe(
            comparison.style.format(
                {
                    "cumulative_return": "{:.2%}",
                    "sharpe_ratio": "{:.2f}",
                    "max_drawdown": "{:.2%}",
                }
            ),
            use_container_width=True,
        )

        for result in results:
            plot_result(result)
    else:
        st.info("Run `python train_pipeline.py` first, then click the inference button.")


if __name__ == "__main__":
    main()
