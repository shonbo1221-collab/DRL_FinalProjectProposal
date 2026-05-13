import pandas as pd
import numpy as np

class SMCExtractor:
    def __init__(self, df):
        """
        df needs to have 'open', 'high', 'low', 'close' columns.
        """
        self.df = df.copy()

    def extract_fvg(self):
        """
        Identify Fair Value Gaps (FVG).
        Bullish FVG: Low_t > High_{t-2}
        Bearish FVG: High_t < Low_{t-2}
        """
        self.df['fvg_bullish_gap'] = self.df['low'] - self.df['high'].shift(2)
        self.df['is_bullish_fvg'] = self.df['fvg_bullish_gap'] > 0
        
        self.df['fvg_bearish_gap'] = self.df['low'].shift(2) - self.df['high']
        self.df['is_bearish_fvg'] = self.df['fvg_bearish_gap'] > 0
        
        # Calculate midpoints of FVGs to use as reference levels
        self.df['bullish_fvg_mid'] = np.where(
            self.df['is_bullish_fvg'], 
            (self.df['low'] + self.df['high'].shift(2)) / 2, 
            np.nan
        )
        self.df['bearish_fvg_mid'] = np.where(
            self.df['is_bearish_fvg'], 
            (self.df['high'] + self.df['low'].shift(2)) / 2, 
            np.nan
        )
        
        # Forward fill the last known FVG midpoints
        self.df['last_bullish_fvg_mid'] = self.df['bullish_fvg_mid'].ffill()
        self.df['last_bearish_fvg_mid'] = self.df['bearish_fvg_mid'].ffill()
        
        # Feature: Distance to nearest FVG mid (percentage)
        self.df['dist_to_bull_fvg'] = (self.df['close'] - self.df['last_bullish_fvg_mid']) / self.df['close']
        self.df['dist_to_bear_fvg'] = (self.df['close'] - self.df['last_bearish_fvg_mid']) / self.df['close']
        
        # Fill remaining NaNs (at the beginning of the series) with 0
        self.df['dist_to_bull_fvg'] = self.df['dist_to_bull_fvg'].fillna(0)
        self.df['dist_to_bear_fvg'] = self.df['dist_to_bear_fvg'].fillna(0)
        
        return self.df

    def extract_pd_array(self, window=60):
        """
        Calculate Premium/Discount Array ratio.
        Ratio = (Close - Min_Low) / (Max_High - Min_Low)
        """
        min_low = self.df['low'].rolling(window=window).min()
        max_high = self.df['high'].rolling(window=window).max()
        
        # Avoid division by zero
        range_high_low = max_high - min_low
        range_high_low = range_high_low.replace(0, 1e-5)
        
        self.df['pd_ratio'] = (self.df['close'] - min_low) / range_high_low
        self.df['pd_ratio'] = self.df['pd_ratio'].fillna(0.5) # default neutral
        
        return self.df

    def extract_order_blocks(self, window=20, impulse_threshold=0.03):
        """
        Simplified Order Block logic:
        A Bullish OB is the last bearish candle before a strong upward impulse.
        Impulse defined as > impulse_threshold return over 3 days.
        """
        # Calculate 3-day return
        self.df['ret_3d'] = self.df['close'] / self.df['close'].shift(3) - 1
        
        # Identify bearish candles (close < open)
        self.df['is_bearish_candle'] = self.df['close'] < self.df['open']
        
        # Condition: strong bullish impulse
        impulse_condition = self.df['ret_3d'] > impulse_threshold
        
        # We need to find the last bearish candle *before* the impulse.
        # This is a bit tricky to vectorize cleanly without a loop. We will use rolling windows.
        
        ob_tops = []
        last_ob_top = np.nan
        
        # Get values as numpy arrays for speed
        closes = self.df['close'].values
        opens = self.df['open'].values
        highs = self.df['high'].values
        ret_3ds = self.df['ret_3d'].values
        is_bearish = self.df['is_bearish_candle'].values
        
        for i in range(len(self.df)):
            if i < 3:
                ob_tops.append(np.nan)
                continue
                
            if ret_3ds[i] > impulse_threshold:
                # Look back over the window (e.g., last 10 days) to find the last bearish candle
                start_idx = max(0, i - window)
                found_ob = False
                for j in range(i-3, start_idx-1, -1): # search backwards from right before the impulse started
                    if is_bearish[j]:
                        last_ob_top = highs[j] # Top of OB is the high of the bearish candle
                        found_ob = True
                        break
                        
            ob_tops.append(last_ob_top)
            
        self.df['last_bull_ob_top'] = ob_tops
        self.df['last_bull_ob_top'] = self.df['last_bull_ob_top'].ffill()
        
        # Feature: Distance to nearest Bullish OB
        self.df['dist_to_bull_ob'] = (self.df['close'] - self.df['last_bull_ob_top']) / self.df['close']
        self.df['dist_to_bull_ob'] = self.df['dist_to_bull_ob'].fillna(0)
        
        return self.df

    def get_all_features(self):
        self.extract_fvg()
        self.extract_pd_array()
        self.extract_order_blocks()
        return self.df

if __name__ == "__main__":
    import os
    if os.path.exists("data/raw/0050_TW_2014-01-01_2024-01-01.csv"):
        df = pd.read_csv("data/raw/0050_TW_2014-01-01_2024-01-01.csv")
        extractor = SMCExtractor(df)
        smc_df = extractor.get_all_features()
        print(smc_df[['date', 'close', 'pd_ratio', 'dist_to_bull_fvg', 'dist_to_bull_ob']].tail(10))
