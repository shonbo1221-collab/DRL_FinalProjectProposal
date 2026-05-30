# TWSE PPO + SMC Trading Pipeline

本專案是一個台股量化交易研究平台，核心是把 SMC 價格行為特徵轉成數值資料，再用 Stable-Baselines3 PPO 訓練資金配置策略。

目前專案已移除舊版單一 0050 demo，只保留正式的「訓練 / 推論分離」架構。

## 目前模型

Model 1：Pair model

```text
0050.TW vs 2330.TW
model/saved/ppo_model_pair.zip
```

Model 2：Basket model

```text
0050.TW vs 2330.TW vs 2412.TW
model/saved/ppo_model_basket.zip
```

## 資料區間

訓練資料：

```text
2018-01-01 to 2024-12-31
```

推論資料：

```text
2025-01-01 to 2026-05-01
```

## 主要檔案

```text
app.py                    # Streamlit 推論驗證網頁
train_pipeline.py         # 只負責訓練與儲存 PPO 權重
predict_pipeline.py       # 只負責載入模型並做 out-of-sample 推論
twse_pipeline_common.py   # 共用資料、特徵、交易環境、績效計算
environment.yml           # Conda 環境設定
model/saved/              # 模型輸出位置
docs/                     # 報告與圖片
```

## SMC 特徵

每個標的都會產生：

```text
PD_Pos      # 價格在 20 日 dealing range 的位置，約 0 到 1
OB_Dist     # 距離最近 Order Block 的百分比距離
FVG_Signal  # 是否存在尚未回補的 Fair Value Gap
```

Pair / Basket 模型還會加入 rolling spread z-score：

```text
Spread_ZScore_0050_TW_2330_TW
Spread_ZScore_0050_TW_2412_TW
Spread_ZScore_2330_TW_2412_TW
```

Pair model 只會使用 `0050.TW` 與 `2330.TW` 的 spread z-score。Basket model 會使用三組 pairwise spread z-score。

## Action 設計

Pair model action shape：

```text
[-1, 1]
```

解讀：

```text
action > 0  -> 配置 0050.TW
action < 0  -> 配置 2330.TW
action ~= 0 -> 偏現金
```

Basket model action shape：

```text
[action_0050, action_2330, action_2412]
```

解讀：

```text
正值越大，該標的分到越多權重
全部小於等於 0，則保留現金
三個正值會被正規化成總持倉權重
```

## TWSE 交易限制

環境內建：

```text
手續費：0.1425% * 0.6
0050.TW 賣出交易稅：0.1%
2330.TW / 2412.TW 賣出交易稅：0.3%
漲停限制：若價格 >= 昨日收盤 * 1.10，該標的 order 失敗
```

## Reward

```text
Reward = PnL * Sharpe_Adjustment - Transaction_Cost - 2.0 * Max_Drawdown_Penalty
```

模型不只追求損益，也會被交易成本與回撤懲罰約束。

## 安裝環境

```bash
conda env create -f environment.yml
conda activate ppo_smc_0050
```

如果要指定環境位置：

```bash
conda env create -f environment.yml -p D:\Env\conda_envs\ppo_smc_0050
conda activate D:\Env\conda_envs\ppo_smc_0050
```

## GPU 訓練

`train_pipeline.py` 現在是 CUDA-only 設定：

```python
DEVICE = "cuda:0"
```

如果沒有 CUDA GPU，程式會直接停止，不會自動退回 CPU。

訓練時會固定把 PPO policy/value network 放在 GPU 上：

```python
model = PPO(..., device="cuda:0")
```

注意：Gymnasium 環境模擬、yfinance 下載與 pandas 特徵工程仍然會在 CPU 執行。這是 Stable-Baselines3 + Gym 的正常架構。此修改能避免 PPO 權重訓練默默退回 CPU，但不代表整個資料處理流程都會變成 GPU kernel。

如果 `torch.cuda.is_available()` 是 `False`，請安裝 CUDA 版 PyTorch，例如：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

檢查 GPU：

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

## 訓練

```bash
python train_pipeline.py
```

訓練會依序產生：

```text
model/saved/ppo_model_pair.zip
model/saved/ppo_model_basket.zip
```

## 推論

```bash
python predict_pipeline.py
```

推論只會載入模型並執行：

```python
model.predict(obs, deterministic=True)
```

推論階段不會呼叫 `model.learn()`，也不會更新 policy weights。

## 網頁驗證

```bash
streamlit run app.py
```

網頁會顯示：

```text
模型檔案是否存在
2025-2026 out-of-sample inference
Pair vs Basket 績效比較
Cumulative Return
Sharpe Ratio
Max Drawdown
PPO actions
實際持倉權重
Portfolio net worth
Inference log
```

## 專案定位

一句話：

```text
這是一個用 PPO 強化學習搭配 SMC 特徵，測試台股 pair/basket 資金配置策略的研究平台。
```
