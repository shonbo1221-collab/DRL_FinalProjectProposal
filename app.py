import os
from dataclasses import replace

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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

COLAB_MODEL_DIR = "model/save_colab"
COLAB_CONFIGS = (
    replace(PAIR_CONFIG, model_path=os.path.join(COLAB_MODEL_DIR, "ppo_model_pair.zip")),
    replace(BASKET_CONFIG, model_path=os.path.join(COLAB_MODEL_DIR, "ppo_model_basket.zip")),
)


@st.cache_data(show_spinner=False)
def load_feature_data(config, start_date, end_date):
    return prepare_dataset(config, start_date, end_date)


@st.cache_resource(show_spinner=False)
def load_ppo_model(model_path):
    return PPO.load(model_path)


def run_twse_inference(
    config,
    start_date,
    end_date,
    initial_capital,
    max_position,
    deterministic,
):
    if not os.path.exists(config.model_path):
        raise FileNotFoundError(f"Model not found: {config.model_path}")

    feature_df = load_feature_data(config, start_date, end_date)
    env = TWTradingEnv(
        feature_df=feature_df,
        config=config,
        initial_balance=initial_capital,
        max_position=max_position,
    )
    model = load_ppo_model(config.model_path)

    obs, _ = env.reset()
    done = False
    action_logs = []

    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
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


def ticker_prefix(ticker):
    return ticker.replace(".", "_")


def pct(value):
    return f"{value:.2%}"


def get_signal_label(weight, action_value):
    if weight >= 0.65:
        return "Strong Buy / High Allocation"
    if weight >= 0.35:
        return "Buy / Medium Allocation"
    if weight >= 0.10:
        return "Watch / Low Allocation"
    if action_value < -0.25:
        return "Reduce / Avoid"
    return "Hold Cash / Neutral"


def latest_signal_table(config, feature_df, history, actions):
    latest_history = history.iloc[-1]
    latest_actions = actions.iloc[-1]
    latest_features = feature_df.iloc[-1]

    rows = []
    for ticker in config.tickers:
        prefix = ticker_prefix(ticker)
        weight = float(latest_history.get(f"weight_{ticker}", 0.0))
        action_value = float(latest_actions.get(f"action_{ticker}", latest_actions.get("action", 0.0)))
        pd_pos = float(latest_features.get(f"{prefix}_PD_Pos", np.nan))
        ob_dist = float(latest_features.get(f"{prefix}_OB_Dist", np.nan))
        fvg = int(latest_features.get(f"{prefix}_FVG_Signal", 0))
        close = float(latest_features.get(f"{prefix}_close", np.nan))

        rows.append(
            {
                "ticker": ticker,
                "last_close": close,
                "model_action": action_value,
                "portfolio_weight": weight,
                "signal": get_signal_label(weight, action_value),
                "PD_Pos": pd_pos,
                "OB_Dist": ob_dist,
                "FVG": "Unfilled" if fvg else "None",
            }
        )
    return pd.DataFrame(rows)


def risk_summary(history):
    net_worth = history["net_worth"].astype(float)
    returns = net_worth.pct_change().dropna()
    weight_cols = [col for col in history.columns if col.startswith("weight_")]
    invested = history[weight_cols].sum(axis=1) if weight_cols else pd.Series(0.0, index=history.index)

    return {
        "daily_volatility": float(returns.std() * np.sqrt(252)) if len(returns) > 1 else 0.0,
        "win_rate": float((returns > 0).mean()) if len(returns) else 0.0,
        "avg_exposure": float(invested.mean()) if len(invested) else 0.0,
        "max_exposure": float(invested.max()) if len(invested) else 0.0,
        "total_cost": float(history["transaction_cost"].sum()) if "transaction_cost" in history else 0.0,
        "failed_orders": int(history["failed_orders"].sum()) if "failed_orders" in history else 0,
    }


def trade_events(history, ticker, threshold=0.03):
    col = f"weight_{ticker}"
    if col not in history:
        return pd.DataFrame(columns=["date", "event", "delta_weight"])
    diff = history[col].diff().fillna(history[col])
    events = history.loc[diff.abs() >= threshold, ["date"]].copy()
    events["event"] = np.where(diff.loc[events.index] > 0, "Buy / Add", "Sell / Reduce")
    events["delta_weight"] = diff.loc[events.index].values
    return events


def plot_market_chart(config, feature_df, history):
    fig = make_subplots(
        rows=len(config.tickers),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=[f"{ticker} Price with PPO Trade Points" for ticker in config.tickers],
    )

    for row, ticker in enumerate(config.tickers, start=1):
        prefix = ticker_prefix(ticker)
        fig.add_trace(
            go.Candlestick(
                x=feature_df["date"],
                open=feature_df[f"{prefix}_open"],
                high=feature_df[f"{prefix}_high"],
                low=feature_df[f"{prefix}_low"],
                close=feature_df[f"{prefix}_close"],
                name=ticker,
                showlegend=False,
            ),
            row=row,
            col=1,
        )

        price_lookup = feature_df.set_index("date")[f"{prefix}_close"]
        events = trade_events(history, ticker)
        if not events.empty:
            events = events[events["date"].isin(price_lookup.index)].copy()
            events["price"] = events["date"].map(price_lookup)
            buys = events[events["event"].eq("Buy / Add")]
            sells = events[events["event"].eq("Sell / Reduce")]

            fig.add_trace(
                go.Scatter(
                    x=buys["date"],
                    y=buys["price"],
                    mode="markers",
                    marker=dict(symbol="triangle-up", size=11, color="#22C55E"),
                    name=f"{ticker} Buy/Add",
                ),
                row=row,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=sells["date"],
                    y=sells["price"],
                    mode="markers",
                    marker=dict(symbol="triangle-down", size=11, color="#EF4444"),
                    name=f"{ticker} Sell/Reduce",
                ),
                row=row,
                col=1,
            )

    fig.update_layout(
        title="Market Chart and Model Trade Points",
        height=max(420, 330 * len(config.tickers)),
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
    )
    for idx in range(1, len(config.tickers) + 1):
        fig.update_xaxes(rangeslider_visible=False, row=idx, col=1)
    return fig


def plot_strategy_dashboard(config, history, actions):
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=("Portfolio Net Worth", "Drawdown", "PPO Actions / Allocation Signals"),
    )
    fig.add_trace(
        go.Scatter(x=history["date"], y=history["net_worth"], mode="lines", name="Net Worth", line=dict(width=3)),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=history["date"], y=-history["drawdown"], mode="lines", name="Drawdown", fill="tozeroy"),
        row=2,
        col=1,
    )

    action_cols = [col for col in actions.columns if col.startswith("action_")]
    if action_cols:
        for col in action_cols:
            fig.add_trace(
                go.Scatter(x=actions["date"], y=actions[col], mode="lines", name=col.replace("action_", "")),
                row=3,
                col=1,
            )
    else:
        fig.add_trace(
            go.Scatter(x=actions["date"], y=actions["action"], mode="lines", name="Action"),
            row=3,
            col=1,
        )
    fig.update_layout(title=f"{config.name} Strategy Dashboard", height=780, template="plotly_dark")
    fig.update_yaxes(title_text="NTD", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown", tickformat=".1%", row=2, col=1)
    fig.update_yaxes(title_text="Action", range=[-1.05, 1.05], row=3, col=1)
    return fig


def plot_weights(history):
    weight_cols = [col for col in history.columns if col.startswith("weight_")]
    fig = go.Figure()
    for col in weight_cols:
        fig.add_trace(go.Scatter(x=history["date"], y=history[col], mode="lines", stackgroup="one", name=col.replace("weight_", "")))
    fig.update_layout(
        title="Executed Portfolio Weights",
        yaxis_title="Portfolio Weight",
        height=360,
        template="plotly_dark",
    )
    return fig


def plot_result(result):
    config = result["config"]
    feature_df = result["feature_df"]
    history = result["history"]
    actions = result["actions"]
    metrics = result["metrics"]

    signal_df = latest_signal_table(config, feature_df, history, actions)
    risk = risk_summary(history)

    st.header(config.name)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Cumulative Return", f"{metrics['cumulative_return']:.2%}")
    col2.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
    col3.metric("Max Drawdown", f"{metrics['max_drawdown']:.2%}")
    col4.metric("Avg Exposure", pct(risk["avg_exposure"]))

    st.subheader("Model Signal Board")
    st.dataframe(
        signal_df.style.format(
            {
                "last_close": "{:,.2f}",
                "model_action": "{:.3f}",
                "portfolio_weight": "{:.2%}",
                "PD_Pos": "{:.2f}",
                "OB_Dist": "{:.2%}",
            }
        ),
        use_container_width=True,
    )

    latest = signal_df.sort_values("portfolio_weight", ascending=False).iloc[0]
    st.info(
        f"Latest model preference: **{latest['ticker']}** | "
        f"Signal: **{latest['signal']}** | "
        f"Weight: **{latest['portfolio_weight']:.2%}** | "
        f"PD_Pos: **{latest['PD_Pos']:.2f}** | FVG: **{latest['FVG']}**"
    )

    st.plotly_chart(plot_market_chart(config, feature_df, history), use_container_width=True)
    st.plotly_chart(plot_strategy_dashboard(config, history, actions), use_container_width=True)
    st.plotly_chart(plot_weights(history), use_container_width=True)

    risk_cols = st.columns(4)
    risk_cols[0].metric("Annualized Volatility", pct(risk["daily_volatility"]))
    risk_cols[1].metric("Daily Win Rate", pct(risk["win_rate"]))
    risk_cols[2].metric("Total Costs", f"NTD {risk['total_cost']:,.0f}")
    risk_cols[3].metric("Failed Orders", f"{risk['failed_orders']}")

    with st.expander(f"{config.name} Professional Tables"):
        st.markdown("**Recent inference log**")
        display_cols = [col for col in history.columns if col in ["date", "raw_action", "net_worth", "transaction_cost", "drawdown", "failed_orders"] or col.startswith("weight_")]
        st.dataframe(history[display_cols].tail(30), use_container_width=True)

        st.markdown("**Detected buy / sell allocation changes**")
        event_frames = []
        for ticker in config.tickers:
            events = trade_events(history, ticker)
            if not events.empty:
                events.insert(1, "ticker", ticker)
                event_frames.append(events)
        if event_frames:
            st.dataframe(pd.concat(event_frames).sort_values("date").tail(50), use_container_width=True)
        else:
            st.write("No allocation changes above threshold.")


def main():
    st.title("TWSE PPO + SMC Professional Analysis")
    st.caption(
        "Load trained PPO models, run out-of-sample inference, and review professional signal, risk, and trade-point analysis."
    )

    config_by_name = {config.name: config for config in COLAB_CONFIGS}
    with st.sidebar:
        st.header("Inference Settings")
        selected_names = st.multiselect(
            "Models",
            options=list(config_by_name),
            default=list(config_by_name),
        )
        start_date = st.date_input("Validation Start", pd.to_datetime("2025-01-01"))
        end_date = st.date_input("Validation End", pd.to_datetime("2026-05-01"))
        initial_capital = st.number_input(
            "Initial Capital (NTD)",
            min_value=100_000,
            value=1_000_000,
            step=100_000,
        )
        max_position = st.slider(
            "Maximum Invested Position Ratio",
            min_value=0.05,
            max_value=1.00,
            value=0.95,
            step=0.05,
            format="%.2f",
        )
        deterministic = st.toggle("Deterministic Actions", value=True)
        run_inference = st.button("Run Validation Inference", type="primary", use_container_width=True)

    configs = tuple(config_by_name[name] for name in selected_names)
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
    st.subheader("Colab Model Status")
    st.dataframe(model_status, use_container_width=True)

    if run_inference:
        if not configs:
            st.warning("Select at least one model.")
            return
        if start_date >= end_date:
            st.error("Validation End must be later than Validation Start.")
            return

        results = []
        for config in configs:
            try:
                with st.spinner(f"Running inference for {config.name}..."):
                    feature_df, history, actions, metrics = run_twse_inference(
                        config,
                        start_date.strftime("%Y-%m-%d"),
                        end_date.strftime("%Y-%m-%d"),
                        initial_capital,
                        max_position,
                        deterministic,
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
                st.info("Upload or copy the trained Colab model files into `model/save_colab`.")
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
        st.caption(
            f"Validation period: {start_date} to {end_date} | "
            f"Initial capital: NTD {initial_capital:,.0f} | "
            f"Maximum invested position ratio: {max_position:.2f} | "
            f"Deterministic actions: {deterministic}"
        )
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
        st.info("Choose a validation period and inference settings, then run validation inference.")


if __name__ == "__main__":
    main()
