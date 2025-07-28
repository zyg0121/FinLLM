import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import torch
from sklearn.preprocessing import StandardScaler, RobustScaler
import pickle

class FeatureFusion:
    """
    Fuse features from different sources for FinLLM
    """
    def __init__(self, output_dir="./processed_features"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Scalers for different feature sets
        self.technical_scaler = None
        self.sentiment_scaler = None
        self.macro_scaler = None
        self.graph_scaler = None
    
    def align_features(self, technical_df, sentiment_df=None, macro_df=None, graph_df=None):
        """
        Align features from different sources to the same date index
        
        Args:
            technical_df: DataFrame with technical features
            sentiment_df: DataFrame with sentiment features
            macro_df: DataFrame with macro features
            graph_df: DataFrame with graph features
            
        Returns:
            Dictionary of aligned DataFrames
        """
        # Get dates from technical data
        dates = technical_df.index
        
        # Initialize result with technical features
        result = {
            'technical': technical_df
        }
        
        # Align sentiment features
        if sentiment_df is not None:
            aligned_sentiment = sentiment_df.reindex(dates, method='ffill')
            result['sentiment'] = aligned_sentiment
        
        # Align macro features
        if macro_df is not None:
            aligned_macro = macro_df.reindex(dates, method='ffill')
            result['macro'] = aligned_macro
        
        # Align graph features
        if graph_df is not None:
            aligned_graph = graph_df.reindex(dates, method='ffill')
            result['graph'] = aligned_graph
        
        return result
    
    def fit_scalers(self, aligned_features):
        """
        Fit scalers to different feature sets
        
        Args:
            aligned_features: Dictionary of aligned feature DataFrames
            
        Returns:
            Self with fitted scalers
        """
        # Fit technical scaler
        if 'technical' in aligned_features:
            self.technical_scaler = RobustScaler()
            self.technical_scaler.fit(aligned_features['technical'])
        
        # Fit sentiment scaler
        if 'sentiment' in aligned_features:
            self.sentiment_scaler = StandardScaler()
            self.sentiment_scaler.fit(aligned_features['sentiment'])
        
        # Fit macro scaler
        if 'macro' in aligned_features:
            self.macro_scaler = RobustScaler()
            self.macro_scaler.fit(aligned_features['macro'])
        
        # Fit graph scaler
        if 'graph' in aligned_features:
            self.graph_scaler = StandardScaler()
            self.graph_scaler.fit(aligned_features['graph'])
        
        return self
    
    def transform_features(self, aligned_features):
        """
        Scale features using fitted scalers
        
        Args:
            aligned_features: Dictionary of aligned feature DataFrames
            
        Returns:
            Dictionary of scaled feature DataFrames
        """
        scaled_features = {}
        
        # Scale technical features
        if 'technical' in aligned_features and self.technical_scaler is not None:
            technical_df = aligned_features['technical'].copy()
            scaled_technical = pd.DataFrame(
                self.technical_scaler.transform(technical_df),
                index=technical_df.index,
                columns=technical_df.columns
            )
            scaled_features['technical'] = scaled_technical
        
        # Scale sentiment features
        if 'sentiment' in aligned_features and self.sentiment_scaler is not None:
            sentiment_df = aligned_features['sentiment'].copy()
            scaled_sentiment = pd.DataFrame(
                self.sentiment_scaler.transform(sentiment_df),
                index=sentiment_df.index,
                columns=sentiment_df.columns
            )
            scaled_features['sentiment'] = scaled_sentiment
        
        # Scale macro features
        if 'macro' in aligned_features and self.macro_scaler is not None:
            macro_df = aligned_features['macro'].copy()
            scaled_macro = pd.DataFrame(
                self.macro_scaler.transform(macro_df),
                index=macro_df.index,
                columns=macro_df.columns
            )
            scaled_features['macro'] = scaled_macro
        
        # Scale graph features
        if 'graph' in aligned_features and self.graph_scaler is not None:
            graph_df = aligned_features['graph'].copy()
            scaled_graph = pd.DataFrame(
                self.graph_scaler.transform(graph_df),
                index=graph_df.index,
                columns=graph_df.columns
            )
            scaled_features['graph'] = scaled_graph
        
        return scaled_features
    
    def create_windows(self, scaled_features, target_df, window_size=30):
        """
        Create sliding windows of features for model input
        
        Args:
            scaled_features: Dictionary of scaled feature DataFrames
            target_df: DataFrame with target variables
            window_size: Window size
            
        Returns:
            Dictionary with windowed features and targets
        """
        # Get common dates
        dates = scaled_features['technical'].index if 'technical' in scaled_features else None
        
        if dates is None:
            raise ValueError("No technical features provided")
        
        # Create windows
        windows = []
        targets = []
        window_dates = []
        
        for i in range(len(dates) - window_size):
            window_idx = dates[i:i+window_size]
            target_date = dates[i+window_size]
            
            # Skip if target not available
            if target_date not in target_df.index:
                continue
            
            window_data = {}
            
            # Get technical features
            if 'technical' in scaled_features:
                window_data['technical'] = scaled_features['technical'].loc[window_idx].values
            
            # Get sentiment features
            if 'sentiment' in scaled_features:
                window_data['sentiment'] = scaled_features['sentiment'].loc[window_idx].values
            
            # Get macro features
            if 'macro' in scaled_features:
                window_data['macro'] = scaled_features['macro'].loc[window_idx].values
            
            # Get graph features
            if 'graph' in scaled_features:
                window_data['graph'] = scaled_features['graph'].loc[window_idx].values
            
            # Get target
            target = target_df.loc[target_date].values
            
            windows.append(window_data)
            targets.append(target)
            window_dates.append((window_idx[0], target_date))
        
        return {
            'windows': windows,
            'targets': np.array(targets),
            'dates': window_dates
        }
    
    def save_processed_data(self, ticker, windowed_data):
        """
        Save processed data for a ticker
        
        Args:
            ticker: Stock ticker
            windowed_data: Dictionary with windowed features and targets
            
        Returns:
            Path to saved file
        """
        # Create file path
        file_path = os.path.join(self.output_dir, f"{ticker}_processed.pkl")
        
        # Save to pickle
        with open(file_path, 'wb') as f:
            pickle.dump(windowed_data, f)
        
        print(f"Saved processed data to {file_path}")
        
        return file_path
    
    def save_scalers(self):
        """
        Save fitted scalers
        
        Returns:
            Path to saved file
        """
        # Create file path
        file_path = os.path.join(self.output_dir, "feature_scalers.pkl")
        
        # Save to pickle
        scalers = {
            'technical': self.technical_scaler,
            'sentiment': self.sentiment_scaler,
            'macro': self.macro_scaler,
            'graph': self.graph_scaler
        }
        
        with open(file_path, 'wb') as f:
            pickle.dump(scalers, f)
        
        print(f"Saved scalers to {file_path}")
        
        return file_path
    
    def load_scalers(self, file_path):
        """
        Load saved scalers
        
        Args:
            file_path: Path to saved scalers
            
        Returns:
            Self with loaded scalers
        """
        with open(file_path, 'rb') as f:
            scalers = pickle.load(f)
        
        self.technical_scaler = scalers.get('technical')
        self.sentiment_scaler = scalers.get('sentiment')
        self.macro_scaler = scalers.get('macro')
        self.graph_scaler = scalers.get('graph')
        
        print(f"Loaded scalers from {file_path}")
        
        return self