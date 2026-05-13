import pandas as pd
import numpy as np

# Adjust imports to handle both direct execution and module import
try:
    from .smc_extractor import SMCExtractor
except ImportError:
    from smc_extractor import SMCExtractor

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from data.preprocessor import DataPreprocessor
except ImportError:
    pass # If running from root, this might fail or succeed depending on path

class FeatureBuilder:
    def __init__(self, raw_df):
        self.raw_df = raw_df
        
    def build(self):
        """
        Combines traditional indicators and SMC features into a single DataFrame.
        """
        # 1. Technical Indicators
        preprocessor = DataPreprocessor()
        processed_df = preprocessor.process(self.raw_df)
        
        # 2. SMC Features
        extractor = SMCExtractor(processed_df)
        smc_df = extractor.get_all_features()
        
        # 3. Select final feature columns used for RL State (Observation Space)
        # 10 dims expected: PD_Ratio, dist_bull_fvg, dist_bear_fvg, dist_bull_ob, ret_5d, atr_20, dev_ma_20, dev_ma_60
        # The other 2 dims are account state (weight, cash) which will be added inside the Env.
        
        feature_cols = [
            'date', 'open', 'high', 'low', 'close', 'volume', # Keep raw prices for environment step simulation
            'pd_ratio', 
            'dist_to_bull_fvg', 
            'dist_to_bear_fvg', 
            'dist_to_bull_ob',
            'return_5d',
            'atr_20',
            'dev_ma_20',
            'dev_ma_60'
        ]
        
        final_df = smc_df[feature_cols].copy()
        
        # Drop rows with NaNs caused by rolling windows or feature extraction
        final_df.dropna(inplace=True)
        final_df.reset_index(drop=True, inplace=True)
        
        return final_df

if __name__ == "__main__":
    if os.path.exists("data/raw/0050_TW_2014-01-01_2024-01-01.csv"):
        df = pd.read_csv("data/raw/0050_TW_2014-01-01_2024-01-01.csv")
        builder = FeatureBuilder(df)
        features_df = builder.build()
        print(f"Built features dataframe shape: {features_df.shape}")
        print(features_df.head())
