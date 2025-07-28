import pandas as pd
import numpy as np
import talib
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, AutoModel
import torch


def compute_technical_indicators(df):
    """
    Compute technical indicators from OHLCV data
    
    Args:
        df: DataFrame with columns 'Open', 'High', 'Low', 'Close', 'Volume'
    
    Returns:
        DataFrame with technical indicators
    """
    # Ensure we have OHLCV data
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Copy input data
    result = df.copy()
    close = df['Close'].values
    high = df['High'].values
    low = df['Low'].values
    volume = df['Volume'].values
    
    # Moving Averages
    result['MA5'] = talib.SMA(close, timeperiod=5)
    result['MA10'] = talib.SMA(close, timeperiod=10)
    result['MA20'] = talib.SMA(close, timeperiod=20)
    result['MA50'] = talib.SMA(close, timeperiod=50)
    result['MA200'] = talib.SMA(close, timeperiod=200)
    
    # Exponential Moving Averages
    result['EMA12'] = talib.EMA(close, timeperiod=12)
    result['EMA26'] = talib.EMA(close, timeperiod=26)
    
    # MACD
    macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    result['MACD'] = macd
    result['MACD_signal'] = macd_signal
    result['MACD_hist'] = macd_hist
    
    # RSI
    result['RSI'] = talib.RSI(close, timeperiod=14)
    
    # Bollinger Bands
    upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
    result['BB_upper'] = upper
    result['BB_middle'] = middle
    result['BB_lower'] = lower
    
    # Stochastic Oscillator
    slowk, slowd = talib.STOCH(high, low, close, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    result['SlowK'] = slowk
    result['SlowD'] = slowd
    
    # Average Directional Index
    result['ADX'] = talib.ADX(high, low, close, timeperiod=14)
    
    # Commodity Channel Index
    result['CCI'] = talib.CCI(high, low, close, timeperiod=14)
    
    # Volume indicators
    result['OBV'] = talib.OBV(close, volume)
    
    # Calculate returns
    result['return_1d'] = result['Close'].pct_change(1)
    result['return_5d'] = result['Close'].pct_change(5)
    result['return_10d'] = result['Close'].pct_change(10)
    
    # Log returns for better statistical properties
    result['log_return_1d'] = np.log(result['Close']).diff(1)
    
    # Volatility
    result['volatility_10d'] = result['log_return_1d'].rolling(window=10).std()
    
    # Forward returns (target variable)
    result['target_1d'] = result['return_1d'].shift(-1)
    
    # Fill missing values with forward and backward filling
    result = result.fillna(method='ffill').fillna(method='bfill')
    
    return result


def prepare_time_series_data(df, window_size=30, features=None, target_col='target_1d'):
    """
    Prepare time series data with sliding window for model input
    
    Args:
        df: DataFrame with features
        window_size: Size of the sliding window
        features: List of feature columns to use, if None use all except target
        target_col: Name of the target column
    
    Returns:
        X: Numpy array of shape [n_samples, window_size, n_features]
        y: Numpy array of shape [n_samples]
    """
    if features is None:
        # Exclude date column, target column, and 'Open', 'High', 'Low', 'Close', 'Volume'
        features = [col for col in df.columns if col not in 
                   ['Date', target_col, 'Open', 'High', 'Low', 'Close', 'Volume']]
    
    # Scale features
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df[features]),
        columns=features,
        index=df.index
    )
    
    X = []
    y = []
    
    for i in range(len(df) - window_size):
        X.append(df_scaled.iloc[i:i+window_size][features].values)
        y.append(df.iloc[i+window_size][target_col])
    
    return np.array(X), np.array(y)


class FinBERTEmbedder:
    """
    Generate FinBERT embeddings for financial text
    """
    def __init__(self, model_name="yiyanghkust/finbert-tone"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
    
    def get_embeddings(self, texts, max_length=512):
        """
        Generate FinBERT embeddings for a list of texts
        
        Args:
            texts: List of strings
            max_length: Maximum token length
        
        Returns:
            Tensor of embeddings [batch_size, embedding_dim]
        """
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
            # Use CLS token embedding as the sentence embedding
            embeddings = outputs.last_hidden_state[:, 0, :]
        
        return embeddings


def process_sentiment_data(news_df, embedder, window_size=14):
    """
    Process financial news and generate sentiment embeddings
    
    Args:
        news_df: DataFrame with columns 'Date', 'Headline'
        embedder: FinBERTEmbedder instance
        window_size: Number of days to consider for each window
    
    Returns:
        Dict mapping dates to tensors of shape [window_size, embedding_dim]
    """
    # Group news by date
    news_by_date = news_df.groupby('Date')['Headline'].apply(list).to_dict()
    
    # Get all unique dates
    all_dates = sorted(news_df['Date'].unique())
    date_to_idx = {date: i for i, date in enumerate(all_dates)}
    
    # Generate embeddings for each date
    date_embeddings = {}
    
    for date, headlines in news_by_date.items():
        # For each date, get up to 5 headlines
        day_headlines = headlines[:5]
        
        # Generate embeddings
        if len(day_headlines) > 0:
            embeddings = embedder.get_embeddings(day_headlines)
            
            # Average the embeddings if there are multiple headlines
            avg_embedding = embeddings.mean(dim=0)
            date_embeddings[date] = avg_embedding
    
    # Create sliding windows of embeddings
    windowed_embeddings = {}
    
    for i, date in enumerate(all_dates[window_size:]):
        # Get the previous window_size dates
        window_dates = all_dates[i:i+window_size]
        
        # Collect embeddings for these dates
        window_embeds = []
        for d in window_dates:
            if d in date_embeddings:
                window_embeds.append(date_embeddings[d])
            else:
                # If no news for this date, use zeros
                window_embeds.append(torch.zeros_like(next(iter(date_embeddings.values()))))
        
        # Stack into a single tensor [window_size, embedding_dim]
        windowed_embeddings[date] = torch.stack(window_embeds)
    
    return windowed_embeddings