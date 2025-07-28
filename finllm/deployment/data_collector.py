import pandas as pd
import numpy as np
import requests
import json
import os
import time
import logging
import threading
import queue
from datetime import datetime, timedelta
import yfinance as yf
import schedule

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("data_collector.log")
    ]
)
logger = logging.getLogger(__name__)

class MarketDataCollector:
    """
    Real-time market data collector for FinLLM deployment
    """
    def __init__(self, output_dir="./data", api_client=None):
        """
        Initialize data collector
        
        Args:
            output_dir: Output directory for data
            api_client: FinLLM API client for real-time predictions
        """
        self.output_dir = output_dir
        self.api_client = api_client
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "market_data"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "news_data"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "predictions"), exist_ok=True)
        
        # Initialize data storage
        self.market_data = {}
        self.news_data = {}
        
        # Thread-safe queues for data collection and processing
        self.market_data_queue = queue.Queue()
        self.news_data_queue = queue.Queue()
        self.prediction_queue = queue.Queue()
        
        # Threading lock for data access
        self.lock = threading.Lock()
        
        # Threading events for stopping threads
        self.stop_event = threading.Event()
    
    def collect_market_data(self, tickers, period="60d", interval="1d"):
        """
        Collect market data for a list of tickers
        
        Args:
            tickers: List of ticker symbols
            period: Data period (e.g., '60d', '1y')
            interval: Data interval (e.g., '1d', '1h')
            
        Returns:
            Dictionary mapping tickers to DataFrames with OHLCV data
        """
        logger.info(f"Collecting market data for {len(tickers)} tickers...")
        
        for ticker in tickers:
            try:
                # Download data from Yahoo Finance
                data = yf.download(ticker, period=period, interval=interval, progress=False)
                
                if len(data) > 0:
                    # Save to file
                    file_path = os.path.join(self.output_dir, "market_data", f"{ticker}_ohlcv.csv")
                    data.to_csv(file_path)
                    
                    # Store in memory
                    with self.lock:
                        self.market_data[ticker] = data
                    
                    # Add to queue for processing
                    self.market_data_queue.put((ticker, data))
                    
                    logger.info(f"Collected {len(data)} records for {ticker}")
                else:
                    logger.warning(f"No data found for {ticker}")
                
            except Exception as e:
                logger.error(f"Error collecting data for {ticker}: {str(e)}")
        
        return self.market_data
    
    def collect_news_data(self, tickers, days=30, api_key=None):
        """
        Collect news data for a list of tickers
        
        Args:
            tickers: List of ticker symbols
            days: Number of days of news to collect
            api_key: API key for news service
            
        Returns:
            Dictionary mapping tickers to lists of news headlines
        """
        logger.info(f"Collecting news data for {len(tickers)} tickers...")
        
        # Example using NewsAPI (you'll need to sign up for an API key)
        if api_key is None:
            logger.warning("No API key provided for news collection")
            return {}
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        for ticker in tickers:
            try:
                # Construct API URL
                url = "https://newsapi.org/v2/everything"
                params = {
                    "q": f"{ticker} stock",
                    "from": start_date.strftime("%Y-%m-%d"),
                    "to": end_date.strftime("%Y-%m-%d"),
                    "language": "en",
                    "sortBy": "publishedAt",
                    "apiKey": api_key
                }
                
                # Make API request
                response = requests.get(url, params=params)
                data = response.json()
                
                if data["status"] == "ok":
                    articles = data["articles"]
                    
                    # Process articles
                    news_by_date = {}
                    
                    for article in articles:
                        # Extract date
                        pub_date = article["publishedAt"][:10]  # YYYY-MM-DD
                        
                        # Add headline to date
                        if pub_date not in news_by_date:
                            news_by_date[pub_date] = []
                        
                        news_by_date[pub_date].append(article["title"])
                    
                    # Convert to list for API input
                    news_list = []
                    for date in pd.date_range(start=start_date, end=end_date):
                        date_str = date.strftime("%Y-%m-%d")
                        headlines = news_by_date.get(date_str, [])
                        news_list.append(";".join(headlines))
                    
                    # Save to file
                    file_path = os.path.join(self.output_dir, "news_data", f"{ticker}_news.json")
                    with open(file_path, 'w') as f:
                        json.dump(news_list, f)
                    
                    # Store in memory
                    with self.lock:
                        self.news_data[ticker] = news_list
                    
                    # Add to queue for processing
                    self.news_data_queue.put((ticker, news_list))
                    
                    logger.info(f"Collected {len(articles)} news articles for {ticker}")
                else:
                    logger.warning(f"Failed to get news for {ticker}: {data.get('message', 'Unknown error')}")
                    
                # Sleep to avoid rate limiting
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error collecting news for {ticker}: {str(e)}")
        
        return self.news_data
    
    def generate_predictions(self, tickers=None):
        """
        Generate predictions using the FinLLM API
        
        Args:
            tickers: List of ticker symbols (None for all available)
            
        Returns:
            Dictionary mapping tickers to prediction results
        """
        if not self.api_client:
            logger.error("API client not initialized")
            return {}
        
        # Get tickers to process
        if tickers is None:
            with self.lock:
                tickers = list(set(self.market_data.keys()) & set(self.news_data.keys()))
        
        logger.info(f"Generating predictions for {len(tickers)} tickers...")
        
        predictions = {}
        batch_requests = []
        
        for ticker in tickers:
            try:
                with self.lock:
                    # Check if we have both market and news data
                    if ticker not in self.market_data or ticker not in self.news_data:
                        logger.warning(f"Missing data for {ticker}, skipping prediction")
                        continue
                    
                    # Get data for ticker
                    market_data = self.market_data[ticker]
                    news_data = self.news_data[ticker]
                
                # Create request
                request = {
                    "ticker": ticker,
                    "ohlcv": market_data,
                    "news": news_data
                }
                
                batch_requests.append(request)
                
            except Exception as e:
                logger.error(f"Error preparing prediction request for {ticker}: {str(e)}")
        
        # Generate predictions in batch
        if batch_requests:
            try:
                results = self.api_client.batch_predict(batch_requests)
                
                # Process results
                for result in results.get("results", []):
                    ticker = result.get("ticker")
                    
                    if ticker:
                        # Save to file
                        file_path = os.path.join(self.output_dir, "predictions", f"{ticker}_prediction.json")
                        with open(file_path, 'w') as f:
                            json.dump(result, f)
                        
                        # Store in memory
                        predictions[ticker] = result
                        
                        # Add to queue for further processing
                        self.prediction_queue.put((ticker, result))
                        
                        logger.info(f"Generated prediction for {ticker}: {result.get('prediction'):.6f} ± {result.get('uncertainty'):.6f}")
                
            except Exception as e:
                logger.error(f"Error in batch prediction: {str(e)}")
        
        return predictions
    
    def start_workers(self, num_workers=3):
        """
        Start worker threads for data processing
        
        Args:
            num_workers: Number of worker threads
        """
        # Market data worker
        threading.Thread(
            target=self._market_data_worker,
            daemon=True
        ).start()
        
        # News data worker
        threading.Thread(
            target=self._news_data_worker,
            daemon=True
        ).start()
        
        # Prediction workers
        for _ in range(num_workers):
            threading.Thread(
                target=self._prediction_worker,
                daemon=True
            ).start()
        
        logger.info(f"Started {num_workers + 2} worker threads")
    
    def _market_data_worker(self):
        """Worker thread for processing market data"""
        logger.info("Market data worker started")
        
        while not self.stop_event.is_set():
            try:
                # Get data from queue with timeout
                try:
                    ticker, data = self.market_data_queue.get(timeout=1)
                except queue.Empty:
                    continue
                
                # Process the data
                # Example: Calculate additional metrics or indicators
                logger.info(f"Processing market data for {ticker}")
                
                # Mark task as done
                self.market_data_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in market data worker: {str(e)}")
    
    def _news_data_worker(self):
        """Worker thread for processing news data"""
        logger.info("News data worker started")
        
        while not self.stop_event.is_set():
            try:
                # Get data from queue with timeout
                try:
                    ticker, news_list = self.news_data_queue.get(timeout=1)
                except queue.Empty:
                    continue
                
                # Process the data
                # Example: Calculate sentiment scores or extract entities
                logger.info(f"Processing news data for {ticker}")
                
                # Mark task as done
                self.news_data_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in news data worker: {str(e)}")
    
    def _prediction_worker(self):
        """Worker thread for processing predictions"""
        logger.info("Prediction worker started")
        
        while not self.stop_event.is_set():
            try:
                # Get data from queue with timeout
                try:
                    ticker, prediction = self.prediction_queue.get(timeout=1)
                except queue.Empty:
                    continue
                
                # Process the prediction
                # Example: Send alerts or update dashboard
                logger.info(f"Processing prediction for {ticker}")
                
                # Mark task as done
                self.prediction_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in prediction worker: {str(e)}")
    
    def schedule_collection(self, tickers, news_api_key=None):
        """
        Schedule regular data collection
        
        Args:
            tickers: List of ticker symbols
            news_api_key: API key for news service
        """
        # Define tasks
        def collect_market_task():
            self.collect_market_data(tickers)
        
        def collect_news_task():
            self.collect_news_data(tickers, api_key=news_api_key)
        
        def generate_predictions_task():
            self.generate_predictions()
        
        # Schedule market data collection (every day at market close)
        schedule.every().day.at("16:30").do(collect_market_task)
        
        # Schedule news collection (every 4 hours)
        schedule.every(4).hours.do(collect_news_task)
        
        # Schedule predictions (every hour)
        schedule.every(1).hours.do(generate_predictions_task)
        
        # Start scheduler
        logger.info("Starting scheduled data collection")
        
        # Initial collection
        collect_market_task()
        collect_news_task()
        generate_predictions_task()
        
        # Start workers
        self.start_workers()
        
        # Run scheduler
        while not self.stop_event.is_set():
            schedule.run_pending()
            time.sleep(1)
    
    def stop(self):
        """Stop all threads"""
        self.stop_event.set()
        logger.info("Stopping data collection")


def main():
    """Main function"""
    import argparse
    from finllm.deployment.client import FinLLMClient
    
    parser = argparse.ArgumentParser(description='FinLLM Data Collector')
    parser.add_argument('--output-dir', default='./data', help='Output directory')
    parser.add_argument('--api-url', default='http://localhost:5000', help='FinLLM API URL')
    parser.add_argument('--news-api-key', help='NewsAPI key for news collection')
    parser.add_argument('--tickers', help='Comma-separated list of tickers (default: S&P 500 top 20)')
    
    args = parser.parse_args()
    
    # Get tickers
    if args.tickers:
        tickers = args.tickers.split(',')
    else:
        # Default to S&P 500 top 20
        tickers = [
            'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 
            'TSLA', 'NVDA', 'UNH', 'JNJ', 'JPM',
            'V', 'PG', 'MA', 'HD', 'BAC',
            'XOM', 'CVX', 'ABBV', 'PFE', 'AVGO'
        ]
    
    # Create API client
    try:
        api_client = FinLLMClient(api_url=args.api_url)
        logger.info(f"Connected to API at {args.api_url}")
    except Exception as e:
        logger.error(f"Failed to connect to API: {str(e)}")
        api_client = None
    
    # Create data collector
    collector = MarketDataCollector(
        output_dir=args.output_dir,
        api_client=api_client
    )
    
    try:
        # Start scheduled collection
        collector.schedule_collection(
            tickers=tickers,
            news_api_key=args.news_api_key
        )
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        # Clean up
        collector.stop()
        logger.info("Data collection stopped")


if __name__ == "__main__":
    main()