import yfinance as yf
import pandas as pd
import os
from datetime import datetime

class DataDownloader:
    def __init__(self, ticker="0050.TW", data_dir="data/raw"):
        self.ticker = ticker
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        
    def download(self, start_date="2014-01-01", end_date="2024-01-01"):
        print(f"Downloading data for {self.ticker} from {start_date} to {end_date}...")
        df = yf.download(self.ticker, start=start_date, end=end_date)
        
        if df.empty:
            raise ValueError(f"No data downloaded for {self.ticker}")
            
        # yfinance might return MultiIndex columns if not careful, ensure it's flat
        if isinstance(df.columns, pd.MultiIndex):
            # Drop the ticker level
            df.columns = df.columns.droplevel(1)
            
        df.reset_index(inplace=True)
        # Rename columns to standard format
        df.rename(columns={'Date': 'date', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Adj Close': 'adj_close', 'Volume': 'volume'}, inplace=True)
        
        # Save raw data
        file_path = os.path.join(self.data_dir, f"{self.ticker.replace('.', '_')}_{start_date}_{end_date}.csv")
        df.to_csv(file_path, index=False)
        print(f"Data saved to {file_path}")
        
        return df

if __name__ == "__main__":
    downloader = DataDownloader()
    df = downloader.download()
    print(df.head())
