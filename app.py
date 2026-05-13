import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os

from data.downloader import DataDownloader
from features.builder import FeatureBuilder
from eval.backtester import Backtester

st.set_page_config(page_title="0050 PPO+SMC Dynamic Allocation", layout="wide")

st.title("結合 PPO 與 SMC 特徵之動態資金配置 (0050)")

# Sidebar
st.sidebar.header("設定 (Settings)")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2021-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-01-01"))
initial_capital = st.sidebar.number_input("Initial Capital", value=1000000, step=100000)

if st.sidebar.button("Run Backtest"):
    with st.spinner("Downloading Data..."):
        downloader = DataDownloader(ticker="0050.TW", data_dir="data/raw")
        df_raw = downloader.download(start_date=start_date.strftime("%Y-%m-%d"), end_date=end_date.strftime("%Y-%m-%d"))
        
    with st.spinner("Extracting SMC Features..."):
        builder = FeatureBuilder(df_raw)
        df_features = builder.build()
        
    with st.spinner("Running PPO Agent..."):
        model_path = "model/saved/ppo_smc_0050.zip"
        if not os.path.exists(model_path):
            st.error(f"Model not found at {model_path}. Please run train.py first.")
            st.stop()
            
        backtester = Backtester("model/saved/ppo_smc_0050", df_features, initial_balance=initial_capital)
        history = backtester.run()
        metrics = backtester.calculate_metrics(history)
        
    # --- UI Layout ---
    
    # Metrics
    st.header("回測績效 (Performance Metrics)")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Return", f"{metrics['total_return']:.2%}")
    col2.metric("Max Drawdown", f"{metrics['max_drawdown']:.2%}")
    col3.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
    
    # 1. Price & SMC Chart
    st.header("1. 市場特徵視圖 (Price & SMC)")
    fig_price = go.Figure()
    
    # Candlestick
    fig_price.add_trace(go.Candlestick(
        x=df_features['date'],
        open=df_features['open'], high=df_features['high'],
        low=df_features['low'], close=df_features['close'],
        name="0050"
    ))
    
    fig_price.update_layout(xaxis_rangeslider_visible=False, height=500, template="plotly_dark")
    st.plotly_chart(fig_price, use_container_width=True)
    
    # 2. PPO Action
    st.header("2. AI 決策視圖 (PPO Position Weight)")
    fig_weight = go.Figure()
    fig_weight.add_trace(go.Scatter(
        x=history['date'], y=history['action'],
        mode='lines', name='Target Weight',
        line=dict(color='cyan', width=2)
    ))
    fig_weight.update_layout(yaxis_title="Weight (0 to 1)", height=300, template="plotly_dark")
    st.plotly_chart(fig_weight, use_container_width=True)
    
    # 3. Portfolio Value
    st.header("3. 累積收益曲線 (Portfolio Net Worth)")
    fig_port = go.Figure()
    
    # Agent Portfolio
    fig_port.add_trace(go.Scatter(
        x=history['date'], y=history['net_worth'],
        mode='lines', name='PPO+SMC Strategy',
        line=dict(color='yellow', width=2)
    ))
    
    # Buy and Hold Baseline
    buy_hold_shares = initial_capital / df_features.iloc[0]['close']
    df_features['buy_hold_value'] = df_features['close'] * buy_hold_shares
    # Align dates (history might be shorter due to feature extraction windows)
    aligned_bh = df_features[df_features['date'].isin(history['date'])]
    
    fig_port.add_trace(go.Scatter(
        x=aligned_bh['date'], y=aligned_bh['buy_hold_value'],
        mode='lines', name='Buy & Hold 0050',
        line=dict(color='gray', width=2, dash='dash')
    ))
    
    fig_port.update_layout(yaxis_title="Net Worth (NTD)", height=400, template="plotly_dark")
    st.plotly_chart(fig_port, use_container_width=True)

else:
    st.info("👈 請在左側設定區間並點擊 'Run Backtest' 以開始分析。")
    st.write("注意事項：執行前回測請先確保已於終端機執行過 `python train.py` 產出模型。")
