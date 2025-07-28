import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import torch
from tqdm import tqdm
import talib
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, AutoModel

class FinLLMDataProcessor:
    """
    Processes raw financial data for the FinLLM model
    """
    def __init__(self, market_data_dir="./market_data", news_data_dir="./news_data", output_dir="./processed_data"):
        self.market_data_dir = market_data_dir
        self.news_data_dir = news_data_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize FinBERT
        self.tokenizer = None
        self.model = None
        
    def load_finbert(self, model_name="yiyanghkust/finbert-tone", device=None):
        """
        Load FinBERT model for text embedding
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
        print(f"Loading FinBERT model on {device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.device = device
        self.model.eval()
        print("FinBERT model loaded")
        
    def compute_technical_features(self, stock_data):
        """
        Compute technical indicators from OHLCV data
        
        Args:
            stock_data: DataFrame with OHLCV data
        
        Returns:
            DataFrame with technical indicators
        """
        df = stock_data.copy()
        
        # Extract price and volume data
        close = df['Close'].values
        high = df['High'].values
        low = df['Low'].values
        open_price = df['Open'].values
        volume = df['Volume'].values
        
        # Moving Averages
        df['MA5'] = talib.SMA(close, timeperiod=5)
        df['MA10'] = talib.SMA(close, timeperiod=10)
        df['MA20'] = talib.SMA(close, timeperiod=20)
        df['MA50'] = talib.SMA(close, timeperiod=50)
        df['MA200'] = talib.SMA(close, timeperiod=200)
        
        # Price Gaps
        df['Price_Gap'] = df['Open'] - df['Close'].shift(1)
        
        # Volatility
        df['Volatility_5'] = df['Close'].pct_change().rolling(5).std()
        df['Volatility_20'] = df['Close'].pct_change().rolling(20).std()
        
        # Relative Strength Index (RSI)
        df['RSI'] = talib.RSI(close, timeperiod=14)
        
        # MACD
        macd, macd_signal, macd_hist = talib.MACD(
            close, fastperiod=12, slowperiod=26, signalperiod=9
        )
        df['MACD'] = macd
        df['MACD_Signal'] = macd_signal
        df['MACD_Hist'] = macd_hist
        
        # Bollinger Bands
        upper, middle, lower = talib.BBANDS(
            close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
        )
        df['BB_Upper'] = upper
        df['BB_Middle'] = middle
        df['BB_Lower'] = lower
        df['BB_Width'] = (upper - lower) / middle
        
        # Average True Range (ATR)
        df['ATR'] = talib.ATR(high, low, close, timeperiod=14)
        
        # On-Balance Volume (OBV)
        df['OBV'] = talib.OBV(close, volume)
        
        # Money Flow Index (MFI)
        df['MFI'] = talib.MFI(high, low, close, volume, timeperiod=14)
        
        # Rate of Change (ROC)
        df['ROC_5'] = talib.ROC(close, timeperiod=5)
        df['ROC_10'] = talib.ROC(close, timeperiod=10)
        df['ROC_20'] = talib.ROC(close, timeperiod=20)
        
        # Commodity Channel Index (CCI)
        df['CCI'] = talib.CCI(high, low, close, timeperiod=14)
        
        # Williams %R
        df['Williams_R'] = talib.WILLR(high, low, close, timeperiod=14)
        
        # Stochastic Oscillator
        slowk, slowd = talib.STOCH(
            high, low, close, fastk_period=5, slowk_period=3,
            slowk_matype=0, slowd_period=3, slowd_matype=0
        )
        df['SlowK'] = slowk
        df['SlowD'] = slowd
        
        # Average Directional Index (ADX)
        df['ADX'] = talib.ADX(high, low, close, timeperiod=14)
        
        # Ichimoku Cloud
        # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
        nine_period_high = df['High'].rolling(window=9).max()
        nine_period_low = df['Low'].rolling(window=9).min()
        df['Tenkan_Sen'] = (nine_period_high + nine_period_low) / 2
        
        # Kijun-sen (Base Line): (26-period high + 26-period low)/2
        period26_high = df['High'].rolling(window=26).max()
        period26_low = df['Low'].rolling(window=26).min()
        df['Kijun_Sen'] = (period26_high + period26_low) / 2
        
        # Target variables - future returns
        df['Return_1d'] = df['Close'].pct_change(1).shift(-1)
        df['Return_5d'] = df['Close'].pct_change(5).shift(-5)
        df['Return_10d'] = df['Close'].pct_change(10).shift(-10)
        df['Return_20d'] = df['Close'].pct_change(20).shift(-20)
        
        # Fill missing values
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        return df
        
    def generate_text_embeddings(self, news_df, batch_size=16):
        """
        Generate text embeddings using FinBERT
        
        Args:
            news_df: DataFrame with news data
            batch_size: Batch size for processing
            
        Returns:
            DataFrame with text embeddings
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("FinBERT model not loaded. Call load_finbert() first.")
            
        # Group by ticker and date
        ticker_date_groups = news_df.groupby(['ticker', 'date'])
        
        embeddings_data = []
        
        for (ticker, date), group in tqdm(ticker_date_groups, desc="Generating embeddings"):
            # Get headlines for this ticker and date
            headlines = group['headline'].tolist()
            
            # Process headlines in batches
            all_embeddings = []
            
            for i in range(0, len(headlines), batch_size):
                batch = headlines[i:i+batch_size]
                
                # Tokenize
                inputs = self.tokenizer(
                    batch, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True,
                    max_length=128
                ).to(self.device)
                
                # Generate embeddings
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    
                    # Use CLS token as the sentence embedding
                    batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                    
                all_embeddings.append(batch_embeddings)
            
            if all_embeddings:
                # Concatenate batch results
                embeddings = np.vstack(all_embeddings)
                
                # Average all headlines for this ticker and date
                avg_embedding = np.mean(embeddings, axis=0)
                
                # Store embedding
                embeddings_data.append({
                    'ticker': ticker,
                    'date': date,
                    'embedding': avg_embedding,
                    'num_headlines': len(headlines)
                })
        
        # Convert embeddings to a dictionary format for easy storage
        embeddings_dict = {}
        for item in embeddings_data:
            ticker = item['ticker']
            date = item['date']
            if ticker not in embeddings_dict:
                embeddings_dict[ticker] = {}
            embeddings_dict[ticker][date] = item['embedding']
        
        # Save embeddings
        np.save(os.path.join(self.output_dir, 'news_embeddings.npy'), embeddings_dict)
        
        return embeddings_dict
    
    def prepare_model_inputs(self, ticker, embeddings_dict, technical_df, window_size=30):
        """
        Prepare inputs for the FinLLM model
        
        Args:
            ticker: Stock ticker symbol
            embeddings_dict: Dictionary of news embeddings by ticker and date
            technical_df: DataFrame with technical indicators
            window_size: Size of the sliding window
            
        Returns:
            Dictionary of model inputs
        """
        # Get dates from technical data
        dates = technical_df.index.tolist()
        
        # Create windows of dates
        windows = []
        targets = []
        
        for i in range(len(dates) - window_size):
            window_dates = dates[i:i+window_size]
            target_date = dates[i+window_size]
            
            windows.append(window_dates)
            targets.append(target_date)
        
        # Prepare inputs
        ts_inputs = []  # Technical data inputs
        text_inputs = []  # Text data inputs
        target_values = []  # Target values
        
        for window_dates, target_date in zip(windows, targets):
            # Get technical data for this window
            window_technical = technical_df.loc[window_dates]
            
            # Select features (exclude OHLCV and target variables)
            feature_cols = [col for col in window_technical.columns 
                           if col not in ['Open', 'High', 'Low', 'Close', 'Volume',
                                         'Return_1d', 'Return_5d', 'Return_10d', 'Return_20d']]
            
            ts_input = window_technical[feature_cols].values
            
            # Get text embeddings for this window
            window_embeddings = []
            
            for date in window_dates:
                date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else date
                
                # Check if we have embeddings for this date
                if ticker in embeddings_dict and date_str in embeddings_dict[ticker]:
                    window_embeddings.append(embeddings_dict[ticker][date_str])
                else:
                    # No news for this date, use zeros
                    if embeddings_dict:
                        # Get embedding dimension from first available embedding
                        first_ticker = list(embeddings_dict.keys())[0]
                        first_date = list(embeddings_dict[first_ticker].keys())[0]
                        emb_dim = len(embeddings_dict[first_ticker][first_date])
                        window_embeddings.append(np.zeros(emb_dim))
                    else:
                        # Default to FinBERT's 768 dimensions
                        window_embeddings.append(np.zeros(768))
            
            # Get target value
            target = technical_df.loc[target_date, 'Return_1d']
            
            # Append to lists
            ts_inputs.append(ts_input)
            text_inputs.append(np.array(window_embeddings))
            target_values.append(target)
        
        return {
            'ts_inputs': np.array(ts_inputs),
            'text_inputs': np.array(text_inputs),
            'targets': np.array(target_values),
            'dates': targets
        }
    
    def process_all_data(self, tickers):
        """
        Process all data for a list of tickers
        
        Args:
            tickers: List of ticker symbols
            
        Returns:
            Dictionary of processed data for each ticker
        """
        # Load news data
        news_files = [f for f in os.listdir(self.news_data_dir) if f.endswith('.csv')]
        
        if news_files:
            news_df = pd.read_csv(os.path.join(self.news_data_dir, news_files[0]))
        else:
            raise FileNotFoundError("No news data files found")
        
        # Filter news for our tickers
        ticker_news = news_df[news_df['ticker'].isin(tickers)]
        
        # Generate text embeddings
        print("Generating text embeddings...")
        embeddings_dict = self.generate_text_embeddings(ticker_news)
        
        # Process each ticker
        processed_data = {}
        
        for ticker in tickers:
            print(f"Processing {ticker}...")
            
            # Find market data file
            market_files = [f for f in os.listdir(self.market_data_dir) if f.startswith(f"{ticker}_") and f.endswith('.csv')]
            
            if not market_files:
                print(f"No market data file found for {ticker}, skipping")
                continue
                
            # Load market data
            market_df = pd.read_csv(os.path.join(self.market_data_dir, market_files[0]), index_col=0, parse_dates=True)
            
            # Compute technical features
            print(f"Computing technical features for {ticker}...")
            tech_df = self.compute_technical_features(market_df)
            
            # Prepare model inputs
            print(f"Preparing model inputs for {ticker}...")
            inputs = self.prepare_model_inputs(ticker, embeddings_dict, tech_df)
            
            # Store processed data
            processed_data[ticker] = inputs
            
            # Save processed data
            np.savez(
                os.path.join(self.output_dir, f"{ticker}_processed.npz"),
                ts_inputs=inputs['ts_inputs'],
                text_inputs=inputs['text_inputs'],
                targets=inputs['targets'],
                dates=[d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else d for d in inputs['dates']]
            )
            
        return processed_data


# Example usage
if __name__ == "__main__":
    # Create processor
    processor = FinLLMDataProcessor(
        market_data_dir="./market_data",
        news_data_dir="./news_data",
        output_dir="./processed_data"
    )
    
    # Load FinBERT model
    processor.load_finbert()
    
    # Process all data
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    processed_data = processor.process_all_data(tickers)
    
    print("Data processing complete!")
    for ticker, data in processed_data.items():
        print(f"{ticker}: {data['ts_inputs'].shape[0]} samples")