import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import pickle

from finllm.features.technical import TechnicalFeatureProcessor
from finllm.features.sentiment import SentimentFeatureProcessor
from finllm.features.macro import MacroFeatureProcessor
from finllm.features.graph import GraphFeatureProcessor
from finllm.features.feature_fusion import FeatureFusion
from finllm.features.auto_feature_selection import FeatureSelector

class FeaturePipeline:
    """
    Complete feature pipeline for FinLLM
    """
    def __init__(self, output_dir="./processed_features"):
        """
        Initialize feature pipeline
        
        Args:
            output_dir: Directory to save processed features
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize feature processors
        self.technical_processor = TechnicalFeatureProcessor()
        self.sentiment_processor = None
        self.macro_processor = None
        self.graph_processor = None
        
        # Initialize feature fusion
        self.feature_fusion = FeatureFusion(output_dir=output_dir)
        
        # Initialize feature selector
        self.feature_selector = FeatureSelector(cut_threshold=0.15)
    
    def initialize_sentiment_processor(self, model_name="yiyanghkust/finbert-tone", device=None):
        """
        Initialize sentiment processor
        
        Args:
            model_name: Name of FinBERT model
            device: PyTorch device
            
        Returns:
            Self with initialized processor
        """
        self.sentiment_processor = SentimentFeatureProcessor(
            model_name=model_name, 
            device=device
        )
        return self
    
    def initialize_macro_processor(self, fred_api_key=None):
        """
        Initialize macro processor
        
        Args:
            fred_api_key: FRED API key
            
        Returns:
            Self with initialized processor
        """
        self.macro_processor = MacroFeatureProcessor(fred_api_key=fred_api_key)
        return self
    
    def initialize_graph_processor(self):
        """
        Initialize graph processor
        
        Returns:
            Self with initialized processor
        """
        self.graph_processor = GraphFeatureProcessor()
        return self
    
    def process_technical_features(self, price_data, include_targets=True):
        """
        Process technical features
        
        Args:
            price_data: DataFrame with OHLCV data
            include_targets: Whether to include target variables
            
        Returns:
            DataFrame with technical features
        """
        return self.technical_processor.compute_features(
            price_data, 
            include_targets=include_targets
        )
    
    def process_sentiment_features(self, news_data, ticker_col='ticker', date_col='date', headline_col='headline'):
        """
        Process sentiment features
        
        Args:
            news_data: DataFrame with news data
            ticker_col: Name of ticker column
            date_col: Name of date column
            headline_col: Name of headline column
            
        Returns:
            Dictionary with sentiment features by ticker and date
        """
        if self.sentiment_processor is None:
            raise ValueError("Sentiment processor not initialized. Call initialize_sentiment_processor first.")
            
        return self.sentiment_processor.process_news_dataframe(
            news_data,
            ticker_col=ticker_col,
            date_col=date_col,
            headline_col=headline_col
        )
    
    def process_macro_features(self, series_list=None, start_date=None, end_date=None):
        """
        Process macro features
        
        Args:
            series_list: List of FRED series IDs
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with macro features
        """
        if self.macro_processor is None:
            raise ValueError("Macro processor not initialized. Call initialize_macro_processor first.")
        
        if series_list is None:
            # Use default series if none provided
            series_list = [
                'UNRATE',  # Unemployment Rate
                'CPIAUCSL',  # Consumer Price Index
                'FEDFUNDS',  # Federal Funds Rate
                'T10Y2Y',  # 10-Year Treasury Constant Maturity Minus 2-Year Treasury
                'INDPRO',  # Industrial Production Index
                'HOUST',  # Housing Starts
                'RSXFS',  # Retail Sales
                'UMCSENT',  # Consumer Sentiment Index
                'DTWEXBGS',  # Trade Weighted U.S. Dollar Index
                'VIXCLS'  # CBOE Volatility Index
            ]
        
        # Get macro data
        if self.macro_processor.fred_api_key:
            macro_df = self.macro_processor.get_fred_data(
                series_list, 
                start_date, 
                end_date
            )
        else:
            macro_df = self.macro_processor.get_dummy_macro_data(
                start_date, 
                end_date,
                num_series=len(series_list)
            )
        
        # Compute derivative features
        return self.macro_processor.compute_derivative_features(macro_df)
    
    def process_graph_features(self, returns_df, news_df, stock_info_df, ticker):
        """
        Process graph features
        
        Args:
            returns_df: DataFrame with stock returns
            news_df: DataFrame with news data
            stock_info_df: DataFrame with stock information
            ticker: Stock ticker to extract features for
            
        Returns:
            DataFrame with graph features
        """
        if self.graph_processor is None:
            raise ValueError("Graph processor not initialized. Call initialize_graph_processor first.")
            
        # Build daily graphs
        daily_graphs = self.graph_processor.build_daily_graphs(
            returns_df, 
            news_df, 
            stock_info_df
        )
        
        # Extract features for ticker
        return self.graph_processor.extract_daily_graph_features(
            daily_graphs, 
            [ticker]
        )
    
    def process_ticker_data(self, ticker, price_data, news_data=None, macro_data=None, 
                           stock_info=None, returns_data=None, window_size=30):
        """
        Process all data for a single ticker
        
        Args:
            ticker: Stock ticker
            price_data: DataFrame with OHLCV data
            news_data: DataFrame with news data
            macro_data: DataFrame with macro data
            stock_info: DataFrame with stock information
            returns_data: DataFrame with stock returns
            window_size: Window size for feature creation
            
        Returns:
            Dictionary with processed data
        """
        print(f"Processing data for {ticker}...")
        
        # Process technical features
        technical_df = self.process_technical_features(price_data)
        
        # Initialize feature dictionary
        feature_dfs = {
            'technical': technical_df
        }
        
        # Process sentiment features if available
        if news_data is not None and self.sentiment_processor is not None:
            ticker_news = news_data[news_data['ticker'] == ticker]
            if len(ticker_news) > 0:
                sentiment_results = self.process_sentiment_features(ticker_news)
                sentiment_df = self.sentiment_processor.create_daily_features(
                    sentiment_results, 
                    ticker, 
                    price_data.index,
                    window_size=window_size
                )
                
                if sentiment_df is not None:
                    feature_dfs['sentiment'] = sentiment_df
                    
                # Add embedding similarity features
                sim_df = self.sentiment_processor.create_embedding_similarity_features(
                    sentiment_results,
                    ticker,
                    price_data.index,
                    window_size=window_size
                )
                
                if sim_df is not None:
                    feature_dfs['embedding_sim'] = sim_df
        
        # Process macro features if available
        if macro_data is not None:
            feature_dfs['macro'] = macro_data
        
        # Process graph features if available
        if returns_data is not None and news_data is not None and stock_info is not None and self.graph_processor is not None:
            graph_df = self.process_graph_features(
                returns_data,
                news_data,
                stock_info,
                ticker
            )
            
            if graph_df is not None:
                feature_dfs['graph'] = graph_df
        
        # Align all features to the same date index
        aligned_features = self.feature_fusion.align_features(
            feature_dfs['technical'],
            feature_dfs.get('sentiment'),
            feature_dfs.get('macro'),
            feature_dfs.get('graph')
        )
        
        # Fit scalers if not already fitted
        if self.feature_fusion.technical_scaler is None:
            self.feature_fusion.fit_scalers(aligned_features)
        
        # Scale features
        scaled_features = self.feature_fusion.transform_features(aligned_features)
        
        # Extract target variables
        target_cols = [col for col in technical_df.columns if col.startswith('Target_')]
        if not target_cols:
            raise ValueError("No target columns found in technical data")
            
        target_df = technical_df[target_cols]
        
        # Create windows
        windowed_data = self.feature_fusion.create_windows(
            scaled_features, 
            target_df,
            window_size=window_size
        )
        
        # Save processed data
        self.feature_fusion.save_processed_data(ticker, windowed_data)
        
        return windowed_data
    
    def run_feature_selection(self, processed_data_dict, method='lgb_shap', plot=True):
        """
        Run feature selection on processed data
        
        Args:
            processed_data_dict: Dictionary of processed data for multiple tickers
            method: Feature selection method ('lgb_shap', 'permutation', 'f_regression', 'ensemble')
            plot: Whether to plot feature importances
            
        Returns:
            List of selected feature names
        """
        print("Running feature selection...")
        
        # Combine data from all tickers
        all_X = []
        all_y = []
        all_features = []
        
        for ticker, data in processed_data_dict.items():
            windows = data['windows']
            targets = data['targets']
            
            for i, window in enumerate(windows):
                # Flatten window data
                flattened = []
                
                for feature_type, features in window.items():
                    feature_names = [f"{feature_type}_{i}_{j}" for i in range(features.shape[0]) for j in range(features.shape[1])]
                    flattened.append(features.flatten())
                    
                    if len(all_features) == 0:
                        all_features.extend(feature_names)
                
                X = np.concatenate(flattened)
                y = targets[i][0]  # Assuming single target value
                
                all_X.append(X)
                all_y.append(y)
        
        # Convert to arrays
        X = np.array(all_X)
        y = np.array(all_y)
        
        # Split into train and validation sets
        n_samples = len(X)
        split_idx = int(n_samples * 0.8)
        
        X_train = X[:split_idx]
        y_train = y[:split_idx]
        X_val = X[split_idx:]
        y_val = y[split_idx:]
        
        print(f"Training feature selector on {len(X_train)} samples with {len(all_features)} features")
        
        # Train a simple model for permutation importance
        if method == 'permutation' or method == 'ensemble':
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
        else:
            model = None
        
        # Run feature selection
        if method == 'lgb_shap':
            selected_features = self.feature_selector.select_features_lgb_shap(
                X_train, y_train, all_features
            )
        elif method == 'permutation':
            selected_features = self.feature_selector.select_features_permutation(
                model, X_val, y_val, all_features
            )
        elif method == 'f_regression':
            selected_features = self.feature_selector.select_features_f_regression(
                X_train, y_train, all_features
            )
        elif method == 'ensemble':
            selected_features = self.feature_selector.select_features_ensemble(
                X_train, y_train, X_val, y_val, model, all_features
            )
        else:
            raise ValueError(f"Unknown feature selection method: {method}")
        
        # Plot feature importances
        if plot:
            self.feature_selector.plot_feature_importance(method)
        
        # Save selected features
        with open(os.path.join(self.output_dir, f"selected_features_{method}.pkl"), 'wb') as f:
            pickle.dump(selected_features, f)
        
        print(f"Feature selection complete. Selected {len(selected_features)} features.")
        
        return selected_features
    
    def save_pipeline(self, filename="feature_pipeline.pkl"):
        """
        Save pipeline to file
        
        Args:
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        file_path = os.path.join(self.output_dir, filename)
        
        # Save feature fusion scalers
        self.feature_fusion.save_scalers()
        
        # Create state dict with everything except the models
        state = {
            'technical_processor': self.technical_processor,
            'feature_selector': self.feature_selector,
            'sentiment_processor_initialized': self.sentiment_processor is not None,
            'macro_processor_initialized': self.macro_processor is not None,
            'graph_processor_initialized': self.graph_processor is not None
        }
        
        with open(file_path, 'wb') as f:
            pickle.dump(state, f)
        
        print(f"Saved pipeline to {file_path}")
        
        return file_path
    
    @classmethod
    def load_pipeline(cls, file_path, load_sentiment=False, model_name=None, device=None, fred_api_key=None):
        """
        Load pipeline from file
        
        Args:
            file_path: Path to saved pipeline
            load_sentiment: Whether to initialize sentiment processor
            model_name: Name of FinBERT model (if loading sentiment)
            device: PyTorch device (if loading sentiment)
            fred_api_key: FRED API key
            
        Returns:
            Loaded pipeline
        """
        # Load state dict
        with open(file_path, 'rb') as f:
            state = pickle.load(f)
        
        # Create new pipeline
        output_dir = os.path.dirname(file_path)
        pipeline = cls(output_dir=output_dir)
        
        # Restore state
        pipeline.technical_processor = state['technical_processor']
        pipeline.feature_selector = state['feature_selector']
        
        # Load feature fusion scalers
        scalers_path = os.path.join(output_dir, "feature_scalers.pkl")
        if os.path.exists(scalers_path):
            pipeline.feature_fusion.load_scalers(scalers_path)
        
        # Initialize processors if needed
        if load_sentiment and state['sentiment_processor_initialized']:
            if model_name is None:
                model_name = "yiyanghkust/finbert-tone"
            pipeline.initialize_sentiment_processor(model_name=model_name, device=device)
        
        if state['macro_processor_initialized']:
            pipeline.initialize_macro_processor(fred_api_key=fred_api_key)
        
        if state['graph_processor_initialized']:
            pipeline.initialize_graph_processor()
        
        print(f"Loaded pipeline from {file_path}")
        
        return pipeline