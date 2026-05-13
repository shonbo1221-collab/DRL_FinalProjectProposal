import pandas as pd
import numpy as np

class DataPreprocessor:
    def __init__(self):
        pass
        
    def process(self, df):
        """
        Calculates basic technical indicators for the environment.
        """
        df = df.copy()
        
        # Calculate daily returns
        df['return'] = df['close'].pct_change()
        
        # Calculate 5-day cumulative return
        df['return_5d'] = df['close'] / df['close'].shift(5) - 1
        
        # Calculate MAs
        df['ma_20'] = df['close'].rolling(window=20).mean()
        df['ma_60'] = df['close'].rolling(window=60).mean()
        
        # Calculate deviations from MA
        df['dev_ma_20'] = (df['close'] - df['ma_20']) / df['ma_20']
        df['dev_ma_60'] = (df['close'] - df['ma_60']) / df['ma_60']
        
        # Calculate ATR (Average True Range)
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = abs(df['high'] - df['close'].shift(1))
        df['tr3'] = abs(df['low'] - df['close'].shift(1))
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        df['atr_20'] = df['tr'].rolling(window=20).mean()
        
        # Drop intermediate TR columns
        df.drop(['tr1', 'tr2', 'tr3', 'tr'], axis=1, inplace=True)
        
        # Forward fill and drop NaNs
        df.ffill(inplace=True)
        df.dropna(inplace=True)
        
        df.reset_index(drop=True, inplace=True)
        return df

if __name__ == "__main__":
    # Test with dummy data
    from downloader import DataDownloader
    import os
    
    if os.path.exists("data/raw/0050_TW_2014-01-01_2024-01-01.csv"):
        df = pd.read_csv("data/raw/0050_TW_2014-01-01_2024-01-01.csv")
        preprocessor = DataPreprocessor()
        processed_df = preprocessor.process(df)
        print(processed_df.tail())
