import torch
import os
import json
import numpy as np
from flask import Flask, request, jsonify
import logging
import time
from datetime import datetime
from threading import Lock
import pandas as pd

# Import FinLLM components
from finllm.models.core import FinLLM
from finllm.features.processing import compute_technical_indicators
from finllm.features.sentiment import FinBERTEmbedder


class ModelServer:
    """
    Server for FinLLM model inference
    """
    def __init__(self, model_path, config_path=None, device=None):
        self.logger = logging.getLogger('ModelServer')
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.logger.info(f"Using device: {self.device}")
        
        # Load configuration
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            # Default configuration
            self.config = {
                "ts_input_dim": 30,
                "text_embedding_dim": 768,
                "hidden_dim": 64,
                "num_heads": 4,
                "num_layers": 2,
                "dropout": 0.2,
                "window_size": 30,
                "finbert_model": "yiyanghkust/finbert-tone"
            }
        
        # Initialize model
        self.model = FinLLM(
            ts_input_dim=self.config["ts_input_dim"],
            text_embedding_dim=self.config["text_embedding_dim"],
            hidden_dim=self.config["hidden_dim"],
            num_heads=self.config["num_heads"],
            num_layers=self.config["num_layers"],
            dropout=self.config["dropout"]
        )
        
        # Load model weights
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.logger.info(f"Loaded model from {model_path}")
        else:
            self.logger.warning(f"Model path {model_path} not found, using untrained model")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize FinBERT embedder
        self.embedder = FinBERTEmbedder(
            model_name=self.config["finbert_model"],
            device=self.device
        )
        
        # Lock for thread safety
        self.lock = Lock()
        
        # Warm up the model
        self._warm_up()
    
    def _warm_up(self):
        """Warm up the model with dummy data"""
        self.logger.info("Warming up the model...")
        
        # Create dummy time series data
        dummy_ts = torch.zeros(
            1, 
            self.config["window_size"], 
            self.config["ts_input_dim"]
        ).to(self.device)
        
        # Create dummy text data
        dummy_text = torch.zeros(
            self.config["window_size"],
            1,
            self.config["text_embedding_dim"]
        ).to(self.device)
        
        # Run inference
        with torch.no_grad():
            _ = self.model(dummy_ts, dummy_text)
        
        self.logger.info("Model warm-up complete")
    
    def preprocess_time_series(self, ohlcv_data):
        """
        Preprocess time series data for model input
        
        Args:
            ohlcv_data: DataFrame with OHLCV data
        
        Returns:
            Tensor ready for model input
        """
        # Compute technical indicators
        processed_data = compute_technical_indicators(ohlcv_data)
        
        # Extract features (excluding OHLCV, Date, and target)
        features = processed_data.columns.difference(['Open', 'High', 'Low', 'Close', 'Volume', 'target_1d', 'Date'])
        feature_data = processed_data[features].values
        
        # Standardize features
        mean = np.nanmean(feature_data, axis=0)
        std = np.nanstd(feature_data, axis=0)
        std[std == 0] = 1  # Avoid division by zero
        
        standardized = (feature_data - mean) / std
        
        # Get the last window_size rows
        window = standardized[-self.config["window_size"]:]
        
        # Convert to tensor
        ts_tensor = torch.FloatTensor(window).unsqueeze(0).to(self.device)
        
        return ts_tensor
    
    def preprocess_news(self, news_data):
        """
        Preprocess news data for model input
        
        Args:
            news_data: List of news headlines for the last window_size days
            
        Returns:
            Tensor ready for model input
        """
        # Ensure we have window_size days of news
        if len(news_data) < self.config["window_size"]:
            # Pad with empty strings
            news_data = [''] * (self.config["window_size"] - len(news_data)) + news_data
        elif len(news_data) > self.config["window_size"]:
            # Take only the most recent window_size days
            news_data = news_data[-self.config["window_size"]:]
        
        # Get embeddings for each day
        embeddings = []
        for day_news in news_data:
            if day_news:
                # If multiple headlines for a day, split by semicolon
                headlines = day_news.split(';') if isinstance(day_news, str) else day_news
                
                # Get embeddings
                day_emb = self.embedder.get_embeddings(headlines)
                
                # Average if multiple headlines
                day_emb = day_emb.mean(dim=0)
            else:
                # No news for this day
                day_emb = torch.zeros(self.config["text_embedding_dim"])
            
            embeddings.append(day_emb)
        
        # Stack into tensor [window_size, embedding_dim]
        text_tensor = torch.stack(embeddings).unsqueeze(1).to(self.device)  # [window_size, 1, embedding_dim]
        
        return text_tensor
    
    def predict(self, ts_data, news_data):
        """
        Make prediction for a single ticker
        
        Args:
            ts_data: DataFrame with OHLCV data
            news_data: List of news headlines
            
        Returns:
            Dictionary with prediction results
        """
        start_time = time.time()
        
        # Acquire lock to ensure thread safety
        with self.lock:
            try:
                # Preprocess data
                ts_tensor = self.preprocess_time_series(ts_data)
                text_tensor = self.preprocess_news(news_data)
                
                # Run inference
                with torch.no_grad():
                    outputs = self.model(ts_tensor, text_tensor)
                
                # Extract results
                prediction = outputs['mean'].item()
                uncertainty = outputs['scale'].item()
                
                # Get attention weights for interpretability
                ts_attn = None
                text_attn = outputs['text_attn_weights'].cpu().numpy()
                cross_attn = outputs['cross_attn_weights'].cpu().numpy()
                
                # Calculate latency
                latency = time.time() - start_time
                
                return {
                    'prediction': prediction,
                    'uncertainty': uncertainty,
                    'text_attention': text_attn.tolist(),
                    'cross_attention': cross_attn.tolist(),
                    'latency': latency
                }
                
            except Exception as e:
                self.logger.error(f"Prediction error: {str(e)}")
                return {
                    'error': str(e),
                    'latency': time.time() - start_time
                }


# Flask application
app = Flask(__name__)

# Global model server instance
model_server = None

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint for making predictions
    
    Expected JSON input:
    {
        "ticker": "AAPL",
        "ohlcv": [...],  # List of OHLCV dictionaries
        "news": [...]    # List of news headlines by day
    }
    """
    global model_server
    
    if not model_server:
        return jsonify({
            'error': 'Model server not initialized'
        }), 500
    
    # Get request data
    data = request.get_json()
    
    if not data or 'ohlcv' not in data or 'ticker' not in data:
        return jsonify({
            'error': 'Missing required data (ticker, ohlcv)'
        }), 400
    
    try:
        # Convert OHLCV data to DataFrame
        ohlcv_df = pd.DataFrame(data['ohlcv'])
        
        # Get news data
        news_data = data.get('news', [])
        
        # Make prediction
        result = model_server.predict(ohlcv_df, news_data)
        
        # Add timestamp and ticker
        result['timestamp'] = datetime.now().isoformat()
        result['ticker'] = data['ticker']
        
        return jsonify(result)
        
    except Exception as e:
        app.logger.error(f"API error: {str(e)}")
        return jsonify({
            'error': str(e)
        }), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    global model_server
    
    if model_server:
        return jsonify({
            'status': 'healthy',
            'model': 'loaded'
        })
    else:
        return jsonify({
            'status': 'unhealthy',
            'model': 'not_loaded'
        }), 503


def start_server(model_path, config_path=None, host='0.0.0.0', port=5000):
    """
    Start the Flask server
    
    Args:
        model_path: Path to saved model
        config_path: Path to model config
        host: Server host
        port: Server port
    """
    global model_server
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize model server
    model_server = ModelServer(model_path, config_path)
    
    # Start Flask app
    app.run(host=host, port=port)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='FinLLM Model Server')
    parser.add_argument('--model', type=str, required=True, help='Path to saved model')
    parser.add_argument('--config', type=str, help='Path to model config')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Server host')
    parser.add_argument('--port', type=int, default=5000, help='Server port')
    
    args = parser.parse_args()
    
    start_server(args.model, args.config, args.host, args.port)