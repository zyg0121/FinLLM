import argparse
import time
import concurrent.futures
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import requests
import json
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StressTest:
    """
    Stress test for FinLLM API
    """
    def __init__(self, api_url="http://localhost:5000"):
        """
        Initialize stress test
        
        Args:
            api_url: URL of the FinLLM API
        """
        self.api_url = api_url.rstrip('/')
        self.results = []
    
    def load_test_data(self, tickers, period="60d"):
        """
        Load test data for stress testing
        
        Args:
            tickers: List of ticker symbols
            period: Data period
            
        Returns:
            Dictionary mapping tickers to DataFrames with OHLCV data
        """
        test_data = {}
        
        for ticker in tqdm(tickers, desc="Loading test data"):
            try:
                data = yf.download(ticker, period=period, progress=False)
                
                if len(data) > 0:
                    test_data[ticker] = data
                    logger.info(f"Loaded {len(data)} records for {ticker}")
                else:
                    logger.warning(f"No data found for {ticker}")
            except Exception as e:
                logger.error(f"Error loading data for {ticker}: {str(e)}")
        
        return test_data
    
    def send_request(self, ticker, ohlcv_data):
        """
        Send a prediction request to the API
        
        Args:
            ticker: Ticker symbol
            ohlcv_data: OHLCV data for the ticker
            
        Returns:
            Dictionary with request results
        """
        start_time = time.time()
        
        try:
            # Prepare request data
            request_data = {
                "ticker": ticker,
                "ohlcv": ohlcv_data.reset_index().to_dict(orient="records"),
                "use_cache": False  # Disable cache for stress test
            }
            
            # Send request
            response = requests.post(
                f"{self.api_url}/predict",
                json=request_data,
                timeout=30
            )
            
            # Parse response
            if response.status_code == 200:
                result = response.json()
                latency = time.time() - start_time
                
                # Add metrics to result
                result['request_latency'] = latency
                result['status_code'] = response.status_code
                result['success'] = True
                
                return result
            else:
                logger.warning(f"Request failed for {ticker}: {response.status_code}")
                return {
                    'ticker': ticker,
                    'status_code': response.status_code,
                    'success': False,
                    'request_latency': time.time() - start_time,
                    'error': response.text
                }
                
        except Exception as e:
            logger.error(f"Error sending request for {ticker}: {str(e)}")
            return {
                'ticker': ticker,
                'success': False,
                'request_latency': time.time() - start_time,
                'error': str(e)
            }
    
    def run_concurrent_test(self, test_data, concurrency=4, requests_per_ticker=5):
        """
        Run concurrent stress test
        
        Args:
            test_data: Dictionary mapping tickers to test data
            concurrency: Number of concurrent requests
            requests_per_ticker: Number of requests to send per ticker
            
        Returns:
            List of request results
        """
        logger.info(f"Running concurrent test with {concurrency} workers, {requests_per_ticker} requests per ticker")
        
        # Prepare requests
        requests = []
        
        for ticker, data in test_data.items():
            for _ in range(requests_per_ticker):
                requests.append((ticker, data))
        
        # Run concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [
                executor.submit(self.send_request, ticker, data)
                for ticker, data in requests
            ]
            
            # Collect results
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing requests"):
                result = future.result()
                self.results.append(result)
        
        logger.info(f"Completed {len(self.results)} requests")
        
        return self.results
    
    def analyze_results(self):
        """
        Analyze test results
        
        Returns:
            Dictionary with analysis metrics
        """
        if not self.results:
            return {}
            
        # Extract latencies
        latencies = [r['request_latency'] for r in self.results if r.get('success', False)]
        
        # Calculate metrics
        metrics = {
            'total_requests': len(self.results),
            'successful_requests': len(latencies),
            'failed_requests': len(self.results) - len(latencies),
            'success_rate': len(latencies) / len(self.results) if self.results else 0,
            'mean_latency': np.mean(latencies) if latencies else 0,
            'median_latency': np.median(latencies) if latencies else 0,
            'p90_latency': np.percentile(latencies, 90) if latencies else 0,
            'p95_latency': np.percentile(latencies, 95) if latencies else 0,
            'p99_latency': np.percentile(latencies, 99) if latencies else 0,
            'min_latency': min(latencies) if latencies else 0,
            'max_latency': max(latencies) if latencies else 0,
            'requests_per_second': len(latencies) / sum(latencies) if latencies else 0
        }
        
        logger.info("Stress test results:")
        logger.info(f"  Total requests:      {metrics['total_requests']}")
        logger.info(f"  Successful requests: {metrics['successful_requests']}")
        logger.info(f"  Failed requests:     {metrics['failed_requests']}")
        logger.info(f"  Success rate:        {metrics['success_rate']:.2%}")
        logger.info(f"  Mean latency:        {metrics['mean_latency']:.3f} s")
        logger.info(f"  Median latency:      {metrics['median_latency']:.3f} s")
        logger.info(f"  90th percentile:     {metrics['p90_latency']:.3f} s")
        logger.info(f"  95th percentile:     {metrics['p95_latency']:.3f} s")
        logger.info(f"  99th percentile:     {metrics['p99_latency']:.3f} s")
        logger.info(f"  Requests per second: {metrics['requests_per_second']:.2f}")
        
        return metrics
    
    def plot_results(self, output_file=None):
        """
        Plot test results
        
        Args:
            output_file: File to save plot (None for display only)
        """
        if not self.results:
            logger.warning("No results to plot")
            return
            
        # Extract data for successful requests
        successful = [r for r in self.results if r.get('success', False)]
        
        if not successful:
            logger.warning("No successful requests to plot")
            return
            
        # Extract timestamps and latencies
        timestamps = [time.time() - r['request_latency'] for r in successful]
        start_time = min(timestamps)
        relative_times = [t - start_time for t in timestamps]
        latencies = [r['request_latency'] for r in successful]
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot latencies over time
        ax1.scatter(relative_times, latencies, alpha=0.6)
        ax1.set_title('Request Latency Over Time')
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Latency (seconds)')
        ax1.grid(True)
        
        # Plot latency histogram
        ax2.hist(latencies, bins=30, alpha=0.7)
        ax2.set_title('Latency Distribution')
        ax2.set_xlabel('Latency (seconds)')
        ax2.set_ylabel('Count')
        ax2.grid(True)
        
        # Add percentile lines
        percentiles = [50, 90, 95, 99]
        colors = ['green', 'orange', 'red', 'purple']
        
        for p, color in zip(percentiles, colors):
            percentile_val = np.percentile(latencies, p)
            ax2.axvline(x=percentile_val, color=color, linestyle='--', 
                       label=f'{p}th Percentile: {percentile_val:.3f}s')
        
        ax2.legend()
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file)
            logger.info(f"Plot saved to {output_file}")
        else:
            plt.show()


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='FinLLM API Stress Test')
    parser.add_argument('--api-url', default='http://localhost:5000', help='FinLLM API URL')
    parser.add_argument('--concurrency', type=int, default=4, help='Number of concurrent requests')
    parser.add_argument('--requests-per-ticker', type=int, default=5, help='Requests per ticker')
    parser.add_argument('--tickers', help='Comma-separated list of tickers (default: S&P 500 top 20)')
    parser.add_argument('--output-file', help='File to save plot')
    
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
    
    # Create stress test
    stress_test = StressTest(api_url=args.api_url)
    
    # Load test data
    test_data = stress_test.load_test_data(tickers)
    
    # Run test
    stress_test.run_concurrent_test(
        test_data=test_data,
        concurrency=args.concurrency,
        requests_per_ticker=args.requests_per_ticker
    )
    
    # Analyze results
    stress_test.analyze_results()
    
    # Plot results
    stress_test.plot_results(output_file=args.output_file)


if __name__ == "__main__":
    main()