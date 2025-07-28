import os
import time
import json
import logging
import numpy as np
import pandas as pd
import torch
from flask import Flask, request, jsonify
from datetime import datetime
from threading import Lock
import traceback
import gc

# Import FinLLM components
from finllm.models.core_model import FinLLM
from finllm.features.technical import TechnicalFeatureProcessor
from finllm.features.sentiment import SentimentFeatureProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("finllm_api.log")
    ]
)

logger = logging.getLogger(__name__)

class ModelServer:
    """
    Server for FinLLM model inference
    """
    def __init__(self, model_path, config_path=None, device=None, cache_size=100):
        """
        Initialize model server
        
        Args:
            model_path: Path to model weights
            config_path: Path to model config
            device: Device to run model on
            cache_size: Size of prediction cache
        """
        self.model_path = model_path
        self.config_path = config_path
        self.cache_size = cache_size
        self.prediction_cache = {}
        self.cache_keys = []  # For LRU cache implementation
        self.lock = Lock()
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        logger.info(f"Using device: {self.device}")
        
        # Load config
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            # Default config
            self.config = {
                "ts_input_dim": 30,
                "text_embedding_dim": 768,
                "hidden_dim": 64,
                "output_dim": 64,
                "num_heads": 4,
                "num_layers": 2,
                "dropout": 0.0,  # Use 0 for inference
                "window_size": 30,
                "alpha": 0.95,
                "finbert_model": "yiyanghkust/finbert-tone"
            }
        
        # Initialize model
        self.load_model()
        
        # Initialize feature processors
        self.technical_processor = TechnicalFeatureProcessor()
        self.sentiment_processor = None  # Lazy loading
        
        # Warm up the model
        self._warm_up()
        
        logger.info("Model server initialized successfully")
    
    def load_model(self):
        """Load the FinLLM model"""
        try:
            self.model = FinLLM(
                ts_input_dim=self.config.get("ts_input_dim", 30),
                text_embedding_dim=self.config.get("text_embedding_dim", 768),
                hidden_dim=self.config.get("hidden_dim", 64),
                output_dim=self.config.get("output_dim", 64),
                num_heads=self.config.get("num_heads", 4),
                num_layers=self.config.get("num_layers", 2),
                dropout=self.config.get("dropout", 0.0),
                alpha=self.config.get("alpha", 0.95)
            )
            
            # Load weights
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Loaded model from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def load_sentiment_processor(self):
        """Lazy-load the sentiment processor"""
        if self.sentiment_processor is None:
            try:
                logger.info("Loading sentiment processor...")
                self.sentiment_processor = SentimentFeatureProcessor(
                    model_name=self.config.get("finbert_model", "yiyanghkust/finbert-tone"),
                    device=self.device
                )
                logger.info("Sentiment processor loaded")
            except Exception as e:
                logger.error(f"Failed to load sentiment processor: {str(e)}")
                raise
    
    def _warm_up(self):
        """Warm up the model with a dummy batch"""
        logger.info("Warming up the model...")
        try:
            # Create dummy inputs
            window_size = self.config.get("window_size", 30)
            ts_input_dim = self.config.get("ts_input_dim", 30)
            text_embedding_dim = self.config.get("text_embedding_dim", 768)
            
            dummy_ts = torch.zeros(1, window_size, ts_input_dim).to(self.device)
            dummy_text = torch.zeros(window_size, 1, text_embedding_dim).to(self.device)
            
            # Run inference
            with torch.no_grad():
                _ = self.model(dummy_ts, dummy_text)
            
            logger.info("Model warm-up complete")
        except Exception as e:
            logger.error(f"Model warm-up failed: {str(e)}")
            raise
    
    def process_technical_data(self, ohlcv_data):
        """
        Process OHLCV data for model input
        
        Args:
            ohlcv_data: DataFrame with OHLCV data
            
        Returns:
            Tensor with processed features
        """
        try:
            # Compute technical features
            tech_features = self.technical_processor.compute_features(ohlcv_data, include_targets=False)
            
            # Get feature columns (excluding OHLCV columns)
            feature_cols = [col for col in tech_features.columns 
                          if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
            
            # Create feature matrix
            features = tech_features[feature_cols].values
            
            # Standardize features
            self.technical_processor.fit_scaler(tech_features[feature_cols])
            features_scaled = self.technical_processor.transform_features(tech_features[feature_cols]).values
            
            # Get the last window_size rows
            window_size = self.config.get("window_size", 30)
            if len(features_scaled) >= window_size:
                window_data = features_scaled[-window_size:]
            else:
                # Pad with zeros if not enough data
                padding = np.zeros((window_size - len(features_scaled), features_scaled.shape[1]))
                window_data = np.vstack([padding, features_scaled])
            
            # Convert to tensor
            ts_tensor = torch.FloatTensor(window_data).unsqueeze(0).to(self.device)
            
            return ts_tensor
        except Exception as e:
            logger.error(f"Error processing technical data: {str(e)}")
            raise
    
    def process_news_data(self, news_data):
        """
        Process news data for model input
        
        Args:
            news_data: List of news headlines by day
            
        Returns:
            Tensor with processed features
        """
        try:
            # Lazy-load sentiment processor if needed
            if self.sentiment_processor is None:
                self.load_sentiment_processor()
            
            window_size = self.config.get("window_size", 30)
            
            # Ensure we have window_size days of news
            if len(news_data) < window_size:
                # Pad with empty strings
                news_data = [''] * (window_size - len(news_data)) + news_data
            elif len(news_data) > window_size:
                # Use the most recent window_size days
                news_data = news_data[-window_size:]
            
            # Process news to get embeddings
            embeddings = []
            
            for headlines in news_data:
                # Convert single string to list if needed
                if isinstance(headlines, str):
                    # Split by semicolons if multiple headlines are provided in one string
                    if ';' in headlines:
                        headlines = headlines.split(';')
                    else:
                        headlines = [headlines] if headlines.strip() else []
                
                # Get embeddings for non-empty headlines
                if headlines:
                    # Clean headlines
                    headlines = [h.strip() for h in headlines if h.strip()]
                    
                    if headlines:
                        # Get embeddings
                        emb = self.sentiment_processor.get_embeddings(headlines)
                        
                        # Average if multiple headlines
                        if len(emb) > 0:
                            day_emb = torch.mean(emb, dim=0)
                        else:
                            day_emb = torch.zeros(self.config.get("text_embedding_dim", 768))
                    else:
                        day_emb = torch.zeros(self.config.get("text_embedding_dim", 768))
                else:
                    # No news for this day
                    day_emb = torch.zeros(self.config.get("text_embedding_dim", 768))
                
                embeddings.append(day_emb)
            
            # Stack to create tensor [window_size, embedding_dim]
            text_tensor = torch.stack(embeddings)
            
            # Reshape to [window_size, 1, embedding_dim] for transformer input
            text_tensor = text_tensor.unsqueeze(1).to(self.device)
            
            return text_tensor
        except Exception as e:
            logger.error(f"Error processing news data: {str(e)}")
            raise
    
    def predict(self, ticker, ohlcv_data, news_data=None, use_cache=True):
        """
        Generate prediction for a stock
        
        Args:
            ticker: Stock ticker symbol
            ohlcv_data: DataFrame with OHLCV data
            news_data: List of news headlines by day (optional)
            use_cache: Whether to use the prediction cache
            
        Returns:
            Dictionary with prediction results
        """
        start_time = time.time()
        
        # Generate cache key
        cache_key = ticker + "_" + ohlcv_data.index[-1].strftime("%Y%m%d")
        
        # Check cache if enabled
        if use_cache and cache_key in self.prediction_cache:
            logger.info(f"Cache hit for {ticker}")
            # Update LRU cache order
            with self.lock:
                self.cache_keys.remove(cache_key)
                self.cache_keys.append(cache_key)
            
            result = self.prediction_cache[cache_key].copy()
            result['cache_hit'] = True
            result['latency'] = time.time() - start_time
            return result
        
        try:
            # Process technical data
            ts_tensor = self.process_technical_data(ohlcv_data)
            
            # Process news data if provided
            if news_data:
                text_tensor = self.process_news_data(news_data)
            else:
                # Create dummy text tensor
                window_size = self.config.get("window_size", 30)
                text_embedding_dim = self.config.get("text_embedding_dim", 768)
                text_tensor = torch.zeros(window_size, 1, text_embedding_dim).to(self.device)
            
            # Generate prediction
            with torch.no_grad():
                outputs = self.model(ts_tensor, text_tensor)
            
            # Extract results
            prediction = outputs['mean'].item()
            uncertainty = outputs['scale'].item()
            
            # Store attention weights for interpretability
            text_attn = outputs['text_attn_weights'].cpu().numpy().tolist()
            cross_attn = outputs['cross_attn_weights'].cpu().numpy().tolist()
            fusion_alpha = outputs['fusion_alpha'].item()
            
            # Create result
            result = {
                'ticker': ticker,
                'prediction': prediction,
                'uncertainty': uncertainty,
                'prediction_interval': [prediction - 1.96 * uncertainty, prediction + 1.96 * uncertainty],
                'fusion_alpha': fusion_alpha,
                'text_attention': text_attn,
                'cross_attention': cross_attn,
                'timestamp': datetime.now().isoformat(),
                'latency': time.time() - start_time,
                'cache_hit': False
            }
            
            # Update cache
            with self.lock:
                # Add new prediction to cache
                self.prediction_cache[cache_key] = result.copy()
                self.cache_keys.append(cache_key)
                
                # Evict oldest entry if cache is full
                if len(self.cache_keys) > self.cache_size:
                    oldest_key = self.cache_keys.pop(0)
                    del self.prediction_cache[oldest_key]
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            logger.error(traceback.format_exc())
            
            return {
                'ticker': ticker,
                'error': str(e),
                'latency': time.time() - start_time,
                'cache_hit': False
            }
    
    def clear_cache(self):
        """Clear the prediction cache"""
        with self.lock:
            self.prediction_cache = {}
            self.cache_keys = []
        logger.info("Prediction cache cleared")


# Initialize Flask app
app = Flask(__name__)

# Global model server instance
model_server = None

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    if model_server is not None:
        return jsonify({
            'status': 'healthy',
            'model': 'loaded',
            'device': str(model_server.device),
            'timestamp': datetime.now().isoformat()
        })
    else:
        return jsonify({
            'status': 'unhealthy',
            'model': 'not loaded'
        }), 503

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint for generating predictions
    
    Expected JSON input:
    {
        "ticker": "AAPL",
        "ohlcv": [...],  # List of dictionaries with OHLCV data
        "news": [...],   # Optional list of news headlines by day
        "use_cache": true  # Optional flag to control caching
    }
    """
    if model_server is None:
        return jsonify({
            'error': 'Model server not initialized'
        }), 503
    
    # Get request data
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        if 'ticker' not in data:
            return jsonify({'error': 'Ticker symbol is required'}), 400
        
        if 'ohlcv' not in data or not data['ohlcv']:
            return jsonify({'error': 'OHLCV data is required'}), 400
        
        # Extract parameters
        ticker = data['ticker']
        ohlcv_data = pd.DataFrame(data['ohlcv'])
        
        # Convert date column to datetime index if present
        if 'date' in ohlcv_data.columns:
            ohlcv_data['date'] = pd.to_datetime(ohlcv_data['date'])
            ohlcv_data.set_index('date', inplace=True)
        
        # Get optional parameters
        news_data = data.get('news', None)
        use_cache = data.get('use_cache', True)
        
        # Generate prediction
        result = model_server.predict(ticker, ohlcv_data, news_data, use_cache)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"API error: {str(e)}")
        logger.error(traceback.format_exc())
        
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """
    Endpoint for batch predictions
    
    Expected JSON input:
    {
        "requests": [
            {
                "ticker": "AAPL",
                "ohlcv": [...],
                "news": [...]
            },
            {
                "ticker": "MSFT",
                "ohlcv": [...],
                "news": [...]
            }
        ],
        "use_cache": true
    }
    """
    if model_server is None:
        return jsonify({
            'error': 'Model server not initialized'
        }), 503
    
    try:
        data = request.get_json()
        
        if not data or 'requests' not in data:
            return jsonify({'error': 'No requests provided'}), 400
        
        requests = data['requests']
        use_cache = data.get('use_cache', True)
        
        results = []
        
        for req in requests:
            ticker = req.get('ticker')
            
            if not ticker:
                results.append({
                    'error': 'Ticker symbol is required'
                })
                continue
                
            if 'ohlcv' not in req or not req['ohlcv']:
                results.append({
                    'ticker': ticker,
                    'error': 'OHLCV data is required'
                })
                continue
            
            try:
                ohlcv_data = pd.DataFrame(req['ohlcv'])
                
                # Convert date column to datetime index if present
                if 'date' in ohlcv_data.columns:
                    ohlcv_data['date'] = pd.to_datetime(ohlcv_data['date'])
                    ohlcv_data.set_index('date', inplace=True)
                
                # Get optional parameters
                news_data = req.get('news', None)
                
                # Generate prediction
                result = model_server.predict(ticker, ohlcv_data, news_data, use_cache)
                results.append(result)
                
            except Exception as e:
                results.append({
                    'ticker': ticker,
                    'error': str(e)
                })
        
        return jsonify({
            'results': results,
            'count': len(results),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Batch API error: {str(e)}")
        logger.error(traceback.format_exc())
        
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/clear_cache', methods=['POST'])
def clear_cache():
    """Endpoint to clear prediction cache"""
    if model_server is None:
        return jsonify({
            'error': 'Model server not initialized'
        }), 503
    
    model_server.clear_cache()
    
    return jsonify({
        'status': 'success',
        'message': 'Cache cleared',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/memory_stats', methods=['GET'])
def memory_stats():
    """Endpoint to get memory usage stats"""
    try:
        import psutil
        import torch
        
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        # Collect CUDA memory stats if available
        cuda_stats = {}
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            cuda_stats = {
                'total_memory': torch.cuda.get_device_properties(device).total_memory / 1024**3,
                'allocated_memory': torch.cuda.memory_allocated(device) / 1024**3,
                'cached_memory': torch.cuda.memory_reserved(device) / 1024**3,
                'device_name': torch.cuda.get_device_name(device)
            }
        
        return jsonify({
            'process_memory_mb': memory_info.rss / 1024**2,
            'virtual_memory_mb': memory_info.vms / 1024**2,
            'cuda_memory_gb': cuda_stats,
            'cache_size': len(model_server.cache_keys) if model_server else 0,
            'timestamp': datetime.now().isoformat()
        })
    
    except ImportError:
        return jsonify({
            'error': 'psutil not installed, cannot get memory stats'
        }), 500
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

def create_app(model_path, config_path=None, device=None, cache_size=100):
    """
    Create and configure the Flask application
    
    Args:
        model_path: Path to model weights
        config_path: Path to model config
        device: Device to run model on
        cache_size: Size of prediction cache
        
    Returns:
        Configured Flask app
    """
    global model_server
    
    # Initialize model server
    model_server = ModelServer(
        model_path=model_path,
        config_path=config_path,
        device=device,
        cache_size=cache_size
    )
    
    return app

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Start FinLLM API server')
    parser.add_argument('--model_path', required=True, help='Path to model weights')
    parser.add_argument('--config_path', help='Path to model configuration')
    parser.add_argument('--host', default='0.0.0.0', help='Host to run server on')
    parser.add_argument('--port', type=int, default=5000, help='Port to run server on')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    parser.add_argument('--cache_size', type=int, default=100, help='Size of prediction cache')
    parser.add_argument('--device', help='Device to run model on (e.g., "cuda:0", "cpu")')
    
    args = parser.parse_args()
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = None
    
    # Create app
    app = create_app(
        model_path=args.model_path,
        config_path=args.config_path,
        device=device,
        cache_size=args.cache_size
    )
    
    # Run server
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug
    )