import pandas as pd
import numpy as np
import yfinance as yf
import akshare as ak
import tushare as ts
import os
from datetime import datetime, timedelta
from tqdm import tqdm

class MarketDataCollector:
    """
    Collects market data from various sources for FinLLM
    """
    def __init__(self, data_dir="./data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
    def get_us_stocks(self, tickers, start_date, end_date=None):
        """
        Get US stock data using Yahoo Finance
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD), defaults to today
            
        Returns:
            Dictionary mapping tickers to DataFrames with OHLCV data
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
            
        stock_data = {}
        
        for ticker in tqdm(tickers, desc="Downloading US stocks"):
            try:
                # Download data from Yahoo Finance
                data = yf.download(ticker, start=start_date, end=end_date)
                
                if len(data) > 0:
                    # Save to file
                    file_path = os.path.join(self.data_dir, f"{ticker}_ohlcv.csv")
                    data.to_csv(file_path)
                    
                    stock_data[ticker] = data
                    print(f"Downloaded {ticker}: {len(data)} rows")
                else:
                    print(f"No data for {ticker}")
                    
            except Exception as e:
                print(f"Error downloading {ticker}: {str(e)}")
                
        return stock_data
    
    def get_china_stocks(self, symbols, start_date, end_date=None, token=None):
        """
        Get Chinese A-share data using AKShare or TuShare
        
        Args:
            symbols: List of stock symbols (e.g., '000001' for Ping An)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD), defaults to today
            token: TuShare API token (optional)
            
        Returns:
            Dictionary mapping symbols to DataFrames with OHLCV data
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
            
        stock_data = {}
        
        # Try using AKShare first
        for symbol in tqdm(symbols, desc="Downloading China A-shares"):
            try:
                # Format symbol for AKShare (needs market prefix)
                if symbol.startswith('6'):
                    ak_symbol = f"{symbol}.SH"  # Shanghai
                else:
                    ak_symbol = f"{symbol}.SZ"  # Shenzhen
                
                # Get daily data
                data = ak.stock_zh_a_hist(symbol=ak_symbol, period="daily", 
                                         start_date=start_date.replace("-", ""), 
                                         end_date=end_date.replace("-", ""))
                
                # Rename columns to standard OHLCV
                data = data.rename(columns={
                    "日期": "Date",
                    "开盘": "Open",
                    "收盘": "Close",
                    "最高": "High",
                    "最低": "Low",
                    "成交量": "Volume"
                })
                
                # Set Date as index
                data["Date"] = pd.to_datetime(data["Date"])
                data.set_index("Date", inplace=True)
                
                # Save to file
                file_path = os.path.join(self.data_dir, f"{symbol}_ohlcv.csv")
                data.to_csv(file_path)
                
                stock_data[symbol] = data
                print(f"Downloaded {symbol}: {len(data)} rows")
                
            except Exception as e:
                print(f"AKShare error for {symbol}: {str(e)}")
                
                # Fall back to TuShare if token is provided
                if token:
                    try:
                        # Initialize TuShare
                        ts.set_token(token)
                        pro = ts.pro_api()
                        
                        # Get daily data
                        data = pro.daily(ts_code=f"{symbol}.{'SH' if symbol.startswith('6') else 'SZ'}", 
                                        start_date=start_date.replace("-", ""), 
                                        end_date=end_date.replace("-", ""))
                        
                        # Process TuShare data
                        data["Date"] = pd.to_datetime(data["trade_date"])
                        data = data.rename(columns={
                            "open": "Open",
                            "close": "Close", 
                            "high": "High",
                            "low": "Low",
                            "vol": "Volume"
                        })
                        data.set_index("Date", inplace=True)
                        
                        # Save to file
                        file_path = os.path.join(self.data_dir, f"{symbol}_ohlcv.csv")
                        data.to_csv(file_path)
                        
                        stock_data[symbol] = data
                        print(f"Downloaded {symbol} with TuShare: {len(data)} rows")
                        
                    except Exception as e2:
                        print(f"TuShare error for {symbol}: {str(e2)}")
        
        return stock_data
    
    def get_crypto_data(self, symbols, start_date, end_date=None):
        """
        Get cryptocurrency data using Yahoo Finance
        
        Args:
            symbols: List of cryptocurrency symbols (e.g., 'BTC-USD')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD), defaults to today
            
        Returns:
            Dictionary mapping symbols to DataFrames with OHLCV data
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
            
        crypto_data = {}
        
        for symbol in tqdm(symbols, desc="Downloading crypto data"):
            try:
                # Download data from Yahoo Finance
                data = yf.download(symbol, start=start_date, end=end_date)
                
                if len(data) > 0:
                    # Save to file
                    file_path = os.path.join(self.data_dir, f"{symbol.replace('-', '_')}_ohlcv.csv")
                    data.to_csv(file_path)
                    
                    crypto_data[symbol] = data
                    print(f"Downloaded {symbol}: {len(data)} rows")
                else:
                    print(f"No data for {symbol}")
                    
            except Exception as e:
                print(f"Error downloading {symbol}: {str(e)}")
                
        return crypto_data


# Example usage
if __name__ == "__main__":
    collector = MarketDataCollector(data_dir="./market_data")
    
    # Get US stock data
    us_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    us_data = collector.get_us_stocks(us_tickers, start_date="2020-01-01")
    
    # Get China stock data (requires TuShare token for fallback)
    # china_symbols = ['000001', '600036', '601318']
    # china_data = collector.get_china_stocks(china_symbols, start_date="2020-01-01", token="YOUR_TUSHARE_TOKEN")
    
    # Get crypto data
    crypto_symbols = ['BTC-USD', 'ETH-USD']
    crypto_data = collector.get_crypto_data(crypto_symbols, start_date="2020-01-01")