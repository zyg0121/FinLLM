import pandas as pd
import requests
import os
import time
from datetime import datetime, timedelta
from tqdm import tqdm

class FinancialNewsCollector:
    """
    Collects financial news data for FinLLM
    """
    def __init__(self, api_key=None, data_dir="./data"):
        self.api_key = api_key  # NewsAPI key
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def get_newsapi_data(self, tickers, start_date, end_date=None, max_results=100):
        """
        Get news data from NewsAPI
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD), defaults to today
            max_results: Maximum number of results per ticker
            
        Returns:
            DataFrame with news data
        """
        if self.api_key is None:
            raise ValueError("NewsAPI key is required")
            
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
            
        all_news = []
        
        for ticker in tqdm(tickers, desc="Collecting news"):
            try:
                # NewsAPI only allows 1 month of historical data with free tier
                # So we'll need to make multiple requests for longer periods
                current_date = datetime.strptime(start_date, "%Y-%m-%d")
                end_dt = datetime.strptime(end_date, "%Y-%m-%d")
                
                while current_date < end_dt:
                    # Set period end date (max 30 days from start)
                    period_end = min(current_date + timedelta(days=29), end_dt)
                    
                    # Format dates for API
                    from_date = current_date.strftime("%Y-%m-%d")
                    to_date = period_end.strftime("%Y-%m-%d")
                    
                    # Make API request
                    url = f"https://newsapi.org/v2/everything"
                    params = {
                        "q": f"{ticker} stock OR {ticker} shares OR {ticker} company",
                        "language": "en",
                        "sortBy": "publishedAt",
                        "from": from_date,
                        "to": to_date,
                        "apiKey": self.api_key,
                        "pageSize": 100
                    }
                    
                    response = requests.get(url, params=params)
                    data = response.json()
                    
                    if data["status"] == "ok":
                        articles = data["articles"]
                        
                        for article in articles:
                            all_news.append({
                                "ticker": ticker,
                                "date": article["publishedAt"][:10],  # YYYY-MM-DD
                                "headline": article["title"],
                                "summary": article["description"],
                                "url": article["url"],
                                "source": article["source"]["name"]
                            })
                    
                    # Avoid rate limiting
                    time.sleep(0.5)
                    
                    # Move to next period
                    current_date = period_end + timedelta(days=1)
                    
            except Exception as e:
                print(f"Error collecting news for {ticker}: {str(e)}")
        
        # Convert to DataFrame
        news_df = pd.DataFrame(all_news)
        
        # Save to file
        file_path = os.path.join(self.data_dir, f"financial_news.csv")
        news_df.to_csv(file_path, index=False)
        
        print(f"Collected {len(news_df)} news articles")
        return news_df
    
    def get_dummy_news_data(self, tickers, start_date, end_date=None, avg_articles_per_day=3):
        """
        Generate dummy news data for testing when API key is not available
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD), defaults to today
            avg_articles_per_day: Average number of articles per day per ticker
            
        Returns:
            DataFrame with dummy news data
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
            
        # Create date range
        date_range = pd.date_range(start=start_date, end=end_date)
        
        all_news = []
        
        # Sample headlines and templates
        headlines = [
            "{ticker} Reports Strong Q{quarter} Earnings",
            "{ticker} Stock Rises on Positive Analyst Outlook",
            "{ticker} Announces New Product Launch",
            "Investors React to {ticker} Quarterly Results",
            "{ticker} CEO Discusses Future Growth Strategy",
            "Market Analysis: Is {ticker} a Buy?",
            "{ticker} Partners with {partner} on New Initiative",
            "Analysts Upgrade {ticker} Stock Rating",
            "{ticker} Faces Challenges in Current Market",
            "{ticker} Expands into New Markets"
        ]
        
        partners = ["Microsoft", "Amazon", "Google", "Apple", "Meta", "IBM", "Oracle", "SAP", "Salesforce"]
        
        for ticker in tickers:
            for date in date_range:
                # Randomly determine number of articles for this day
                num_articles = max(0, int(np.random.poisson(avg_articles_per_day)))
                
                for _ in range(num_articles):
                    # Select random headline template
                    headline = np.random.choice(headlines)
                    
                    # Fill in template
                    quarter = np.random.randint(1, 5)
                    partner = np.random.choice(partners)
                    headline = headline.format(ticker=ticker, quarter=quarter, partner=partner)
                    
                    all_news.append({
                        "ticker": ticker,
                        "date": date.strftime("%Y-%m-%d"),
                        "headline": headline,
                        "summary": f"This is a summary for {headline}",
                        "url": f"http://example.com/{ticker.lower()}-news-{date.strftime('%Y%m%d')}",
                        "source": np.random.choice(["Bloomberg", "Reuters", "CNBC", "WSJ", "Financial Times"])
                    })
        
        # Convert to DataFrame
        news_df = pd.DataFrame(all_news)
        
        # Save to file
        file_path = os.path.join(self.data_dir, f"dummy_financial_news.csv")
        news_df.to_csv(file_path, index=False)
        
        print(f"Generated {len(news_df)} dummy news articles")
        return news_df


# Example usage
if __name__ == "__main__":
    collector = FinancialNewsCollector(data_dir="./news_data")
    
    # Using NewsAPI (requires API key)
    # api_key = "YOUR_NEWSAPI_KEY"
    # collector = FinancialNewsCollector(api_key=api_key, data_dir="./news_data")
    # news_df = collector.get_newsapi_data(['AAPL', 'MSFT', 'GOOGL'], start_date="2023-01-01")
    
    # Using dummy data for testing
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    dummy_news = collector.get_dummy_news_data(tickers, start_date="2023-01-01", end_date="2023-12-31")