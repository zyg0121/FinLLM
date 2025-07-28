import requests
import json
import pandas as pd
import numpy as np
import time
from typing import List, Dict, Any, Union, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FinLLMClient:
    """
    Client for interacting with the FinLLM API
    """
    def __init__(self, api_url="http://localhost:5000", api_key=None):
        """
        Initialize FinLLM client
        
        Args:
            api_url: URL of the FinLLM API
            api_key: API key for authentication (if required)
        """
        self.api_url = api_url.rstrip('/')
        self.api_key = api_key
        
        # Test connection
        self._check_connection()
    
    def _check_connection(self):
        """Check connection to the API"""
        try:
            response = self._make_request("GET", "/health")
            if response.get("status") != "healthy":
                logger.warning(f"API not fully healthy: {response}")
        except Exception as e:
            logger.error(f"Error connecting to API: {str(e)}")
            raise
    
    def _make_request(self, method, endpoint, data=None):
        """
        Make a request to the API
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            data: Request data
            
        Returns:
            Response data
        """
        url = f"{self.api_url}{endpoint}"
        headers = {}
        
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        if data is not None:
            headers["Content-Type"] = "application/json"
        
        try:
            response = requests.request(
                method=method,
                url=url,
                headers=headers,
                json=data
            )
            
            # Check for errors
            response.raise_for_status()
            
            return response.json()
        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:
                logger.warning("API rate limit exceeded. Retrying after delay...")
                time.sleep(2)  # Wait before retry
                return self._make_request(method, endpoint, data)
            
            # Log error response if available
            error_msg = f"HTTP error: {e}"
            try:
                error_data = response.json()
                error_msg += f" - {error_data.get('error', '')}"
            except:
                pass
                
            logger.error(error_msg)
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {str(e)}")
            raise
    
    def get_health(self):
        """
        Check API health
        
        Returns:
            Health status
        """
        return self._make_request("GET", "/health")
    
    def predict(
        self, 
        ticker: str, 
        ohlcv_data: Union[pd.DataFrame, List[Dict[str, Any]]], 
        news_data: Optional[List[str]] = None,
        use_cache: bool = True
    ):
        """
        Generate prediction for a stock
        
        Args:
            ticker: Stock ticker symbol
            ohlcv_data: DataFrame or list of dictionaries with OHLCV data
            news_data: List of news headlines by day (optional)
            use_cache: Whether to use prediction cache
            
        Returns:
            Prediction result
        """
        # Convert DataFrame to list of dictionaries if needed
        if isinstance(ohlcv_data, pd.DataFrame):
            # Reset index to include dates if they're in the index
            if isinstance(ohlcv_data.index, pd.DatetimeIndex):
                ohlcv_data = ohlcv_data.reset_index()
                
                # Rename index column to 'date' if it has a different name
                if ohlcv_data.columns[0] != 'date':
                    ohlcv_data = ohlcv_data.rename(columns={ohlcv_data.columns[0]: 'date'})
            
            # Convert dates to ISO format strings
            if 'date' in ohlcv_data.columns:
                ohlcv_data['date'] = ohlcv_data['date'].dt.strftime('%Y-%m-%d')
            
            # Convert to list of dictionaries
            ohlcv_list = ohlcv_data.to_dict(orient='records')
        else:
            ohlcv_list = ohlcv_data
        
        # Prepare request data
        data = {
            "ticker": ticker,
            "ohlcv": ohlcv_list,
            "use_cache": use_cache
        }
        
        if news_data is not None:
            data["news"] = news_data
        
        # Make request
        return self._make_request("POST", "/predict", data)
    
    def batch_predict(
        self,
        requests: List[Dict[str, Any]],
        use_cache: bool = True
    ):
        """
        Generate predictions for multiple stocks
        
        Args:
            requests: List of prediction requests
            use_cache: Whether to use prediction cache
            
        Returns:
            Batch prediction results
        """
        # Process each request
        processed_requests = []
        
        for req in requests:
            ticker = req.get("ticker")
            if not ticker:
                logger.warning(f"Skipping request without ticker: {req}")
                continue
                
            ohlcv_data = req.get("ohlcv")
            if not ohlcv_data:
                logger.warning(f"Skipping request without OHLCV data for {ticker}")
                continue
            
            # Convert DataFrame to list if needed
            if isinstance(ohlcv_data, pd.DataFrame):
                # Reset index to include dates if they're in the index
                if isinstance(ohlcv_data.index, pd.DatetimeIndex):
                    ohlcv_data = ohlcv_data.reset_index()
                    
                    # Rename index column to 'date' if it has a different name
                    if ohlcv_data.columns[0] != 'date':
                        ohlcv_data = ohlcv_data.rename(columns={ohlcv_data.columns[0]: 'date'})
                
                # Convert dates to ISO format strings
                if 'date' in ohlcv_data.columns:
                    ohlcv_data['date'] = ohlcv_data['date'].dt.strftime('%Y-%m-%d')
                
                # Convert to list of dictionaries
                ohlcv_list = ohlcv_data.to_dict(orient='records')
                req["ohlcv"] = ohlcv_list
            
            processed_requests.append(req)
        
        # Prepare request data
        data = {
            "requests": processed_requests,
            "use_cache": use_cache
        }
        
        # Make request
        return self._make_request("POST", "/batch_predict", data)
    
    def clear_cache(self):
        """
        Clear the prediction cache
        
        Returns:
            Result of cache clear operation
        """
        return self._make_request("POST", "/clear_cache")
    
    def get_memory_stats(self):
        """
        Get memory usage statistics
        
        Returns:
            Memory statistics
        """
        return self._make_request("GET", "/memory_stats")


# Example usage
if __name__ == "__main__":
    import yfinance as yf
    
    # Create client
    client = FinLLMClient(api_url="http://localhost:5000")
    
    # Check health
    health = client.get_health()
    print(f"API Health: {health}")
    
    # Download some data
    aapl = yf.download("AAPL", period="60d")
    msft = yf.download("MSFT", period="60d")
    
    # Generate prediction for Apple
    result = client.predict("AAPL", aapl)
    print(f"AAPL Prediction: {result['prediction']:.6f} ± {result['uncertainty']:.6f}")
    
    # Batch prediction for Apple and Microsoft
    batch_results = client.batch_predict([
        {"ticker": "AAPL", "ohlcv": aapl},
        {"ticker": "MSFT", "ohlcv": msft}
    ])
    
    for result in batch_results["results"]:
        ticker = result["ticker"]
        prediction = result["prediction"]
        uncertainty = result["uncertainty"]
        print(f"{ticker} Prediction: {prediction:.6f} ± {uncertainty:.6f}")