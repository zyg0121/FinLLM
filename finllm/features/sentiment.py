import pandas as pd
import numpy as np
import torch
import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, pipeline
from sklearn.metrics.pairwise import cosine_similarity
import re

class SentimentFeatureProcessor:
    """
    Process financial news for sentiment features
    """
    def __init__(self, model_name="yiyanghkust/finbert-tone", device=None, cache_dir=None):
        """
        Initialize the sentiment feature processor
        
        Args:
            model_name: Name of the FinBERT model to use
            device: Torch device
            cache_dir: Directory to cache model files
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        print(f"Loading FinBERT model '{model_name}' on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir).to(self.device)
        self.model.eval()
        
        # Load sentiment analyzer pipeline
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model=model_name,
            tokenizer=self.tokenizer,
            device=0 if self.device.type == "cuda" else -1,
        )
        
        print("FinBERT model loaded successfully")
    
    def get_embeddings(self, texts, batch_size=8, max_length=128):
        """
        Generate FinBERT embeddings for texts
        
        Args:
            texts: List of text strings
            batch_size: Batch size for processing
            max_length: Maximum token length
            
        Returns:
            numpy array of embeddings [num_texts, embedding_dim]
        """
        embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            ).to(self.device)
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                
                # Use CLS token as the sentence embedding
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            embeddings.append(batch_embeddings)
        
        if not embeddings:
            # Return empty array with correct embedding dimension
            return np.array([]).reshape(0, self.model.config.hidden_size)
        
        # Concatenate batch results
        return np.vstack(embeddings)
    
    def get_sentiment_scores(self, texts, batch_size=8):
        """
        Get sentiment scores (positive, negative, neutral) for texts
        
        Args:
            texts: List of text strings
            batch_size: Batch size for processing
            
        Returns:
            DataFrame with sentiment scores
        """
        results = []
        
        # Process in batches
        for i in tqdm(range(0, len(texts), batch_size), desc="Analyzing sentiment"):
            batch = texts[i:i+batch_size]
            batch = [text[:512] if text else "" for text in batch]  # Truncate to avoid errors
            
            try:
                # Get sentiment predictions
                sentiments = self.sentiment_analyzer(batch, truncation=True)
                
                for j, sentiment in enumerate(sentiments):
                    score = sentiment['score']
                    label = sentiment['label']
                    
                    # Create dict with zeros for all labels
                    result = {
                        'positive': 0.0,
                        'negative': 0.0,
                        'neutral': 0.0,
                        'text_index': i + j
                    }
                    
                    # Set score for predicted label
                    result[label.lower()] = score
                    
                    results.append(result)
            except Exception as e:
                print(f"Error processing batch {i}-{i+batch_size}: {e}")
                # Add placeholder results
                for j in range(len(batch)):
                    results.append({
                        'positive': 0.33,
                        'negative': 0.33,
                        'neutral': 0.34,
                        'text_index': i + j
                    })
        
        return pd.DataFrame(results)
    
    def process_news_dataframe(self, news_df, ticker_col='ticker', date_col='date', headline_col='headline', batch_size=32):
        """
        Process a DataFrame of news articles
        
        Args:
            news_df: DataFrame with news articles
            ticker_col: Name of ticker column
            date_col: Name of date column
            headline_col: Name of headline column
            batch_size: Batch size for processing
            
        Returns:
            Dictionary of processed features by ticker and date
        """
        # Group by ticker and date
        grouped = news_df.groupby([ticker_col, date_col])
        
        print(f"Processing {len(grouped)} ticker-date groups...")
        
        results = {}
        
        for (ticker, date), group in tqdm(grouped, desc="Processing news groups"):
            if ticker not in results:
                results[ticker] = {}
            
            # Get headlines
            headlines = group[headline_col].tolist()
            
            # Analyze sentiment
            sentiment_scores = self.get_sentiment_scores(headlines, batch_size=min(batch_size, len(headlines)))
            
            # Calculate summary statistics for sentiment scores
            sentiment_summary = {}
            for col in ['positive', 'negative', 'neutral']:
                sentiment_summary[f'{col}_mean'] = sentiment_scores[col].mean()
                sentiment_summary[f'{col}_max'] = sentiment_scores[col].max()
            
            # Get embeddings
            embeddings = self.get_embeddings(headlines, batch_size=min(batch_size, len(headlines)))
            
            # Calculate average embedding
            if len(embeddings) > 0:
                avg_embedding = np.mean(embeddings, axis=0)
            else:
                avg_embedding = np.zeros(self.model.config.hidden_size)
            
            # Store results
            results[ticker][date] = {
                'sentiment': sentiment_summary,
                'embedding': avg_embedding,
                'num_headlines': len(headlines)
            }
        
        return results
    
    def create_daily_features(self, news_results, ticker, market_dates, window_size=14):
        """
        Create daily features from news results for a ticker
        
        Args:
            news_results: Dictionary of news results from process_news_dataframe
            ticker: Stock ticker
            market_dates: List of market dates (pandas DatetimeIndex)
            window_size: Window size for feature calculation
            
        Returns:
            DataFrame of daily features
        """
        if ticker not in news_results:
            print(f"No news data for ticker {ticker}")
            return None
        
        ticker_results = news_results[ticker]
        date_strs = [date.strftime('%Y-%m-%d') for date in market_dates]
        
        features = {}
        
        # Create daily sentiment features
        for i, date_str in enumerate(date_strs):
            # Initialize with zeros
            daily_features = {
                'positive_mean': 0.0,
                'negative_mean': 0.0,
                'neutral_mean': 0.0,
                'positive_max': 0.0,
                'negative_max': 0.0,
                'neutral_max': 0.0,
                'news_count': 0,
                'has_news': 0,
            }
            
            # If date has news, use sentiment values
            if date_str in ticker_results:
                sentiment = ticker_results[date_str]['sentiment']
                daily_features.update(sentiment)
                daily_features['news_count'] = ticker_results[date_str]['num_headlines']
                daily_features['has_news'] = 1
            
            # Calculate window features if we have enough history
            if i >= window_size:
                window_dates = date_strs[i-window_size:i]
                window_sentiments = [
                    ticker_results[d]['sentiment'] 
                    for d in window_dates 
                    if d in ticker_results
                ]
                
                # Sentiment momentum (change in sentiment)
                if window_sentiments:
                    pos_values = [s['positive_mean'] for s in window_sentiments]
                    neg_values = [s['negative_mean'] for s in window_sentiments]
                    
                    daily_features['sent_momentum'] = pos_values[-1] - pos_values[0] if len(pos_values) > 1 else 0
                    daily_features['sent_volatility'] = np.std(pos_values) if len(pos_values) > 1 else 0
                    daily_features['neg_momentum'] = neg_values[-1] - neg_values[0] if len(neg_values) > 1 else 0
                    
                    # Exponentially weighted sentiment
                    weights = np.exp(np.linspace(0, 1, len(pos_values)))
                    daily_features['sent_ewm'] = np.average(pos_values, weights=weights) if pos_values else 0
                
                # News volume features
                news_counts = [
                    ticker_results[d]['num_headlines'] 
                    for d in window_dates 
                    if d in ticker_results
                ]
                
                daily_features['news_volume'] = np.sum(news_counts) if news_counts else 0
                daily_features['news_spike'] = (daily_features['news_count'] > 2 * np.mean(news_counts)) if news_counts else 0
            
            # Add features to result
            features[date_str] = daily_features
        
        # Convert to DataFrame
        df = pd.DataFrame.from_dict(features, orient='index')
        df.index = pd.DatetimeIndex(df.index)
        
        return df
    
    def create_embedding_similarity_features(self, news_results, ticker, market_dates, window_size=14):
        """
        Create embedding similarity features for a ticker
        
        Args:
            news_results: Dictionary of news results
            ticker: Stock ticker
            market_dates: List of market dates
            window_size: Window size for similarity calculation
            
        Returns:
            DataFrame of similarity features
        """
        if ticker not in news_results:
            return None
        
        ticker_results = news_results[ticker]
        date_strs = [date.strftime('%Y-%m-%d') for date in market_dates]
        
        similarity_features = {}
        
        # Create daily similarity features
        for i, date_str in enumerate(date_strs):
            features = {
                'embedding_sim_1d': 0.0,
                'embedding_sim_3d': 0.0,
                'embedding_sim_7d': 0.0,
                'embedding_sim_trend': 0.0,
            }
            
            # Get current date embedding
            current_embedding = ticker_results.get(date_str, {}).get('embedding')
            
            if current_embedding is not None and i > 0:
                # Previous day similarity
                prev_date = date_strs[i-1]
                prev_embedding = ticker_results.get(prev_date, {}).get('embedding')
                
                if prev_embedding is not None:
                    sim = cosine_similarity([current_embedding], [prev_embedding])[0][0]
                    features['embedding_sim_1d'] = sim
                
                # 3-day similarity
                if i >= 3:
                    prev3_date = date_strs[i-3]
                    prev3_embedding = ticker_results.get(prev3_date, {}).get('embedding')
                    
                    if prev3_embedding is not None:
                        sim = cosine_similarity([current_embedding], [prev3_embedding])[0][0]
                        features['embedding_sim_3d'] = sim
                
                # 7-day similarity
                if i >= 7:
                    prev7_date = date_strs[i-7]
                    prev7_embedding = ticker_results.get(prev7_date, {}).get('embedding')
                    
                    if prev7_embedding is not None:
                        sim = cosine_similarity([current_embedding], [prev7_embedding])[0][0]
                        features['embedding_sim_7d'] = sim
                
                # Similarity trend (increasing or decreasing similarity)
                if i >= 3:
                    sim_values = []
                    for j in range(3):
                        prev_date = date_strs[i-j-1]
                        prev_embedding = ticker_results.get(prev_date, {}).get('embedding')
                        
                        if prev_embedding is not None and current_embedding is not None:
                            sim = cosine_similarity([current_embedding], [prev_embedding])[0][0]
                            sim_values.append(sim)
                    
                    if len(sim_values) >= 2:
                        features['embedding_sim_trend'] = sim_values[0] - sim_values[-1]
            
            similarity_features[date_str] = features
        
        # Convert to DataFrame
        df = pd.DataFrame.from_dict(similarity_features, orient='index')
        df.index = pd.DatetimeIndex(df.index)
        
        return df