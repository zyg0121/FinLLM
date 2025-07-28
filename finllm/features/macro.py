import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime, timedelta
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from fredapi import Fred

class MacroFeatureProcessor:
    """
    Process macroeconomic data for FinLLM
    """
    def __init__(self, fred_api_key=None):
        """
        Initialize the macro feature processor
        
        Args:
            fred_api_key: FRED API key
        """
        self.fred_api_key = fred_api_key
        self.fred = Fred(api_key=fred_api_key) if fred_api_key else None
        
    def get_fred_data(self, series_list, start_date, end_date=None):
        """
        Get data from FRED API
        
        Args:
            series_list: List of FRED series IDs
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with FRED data
        """
        if self.fred is None:
            raise ValueError("FRED API key not provided")
            
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        print(f"Fetching FRED data for {len(series_list)} series...")
        
        all_data = {}
        
        for series_id in tqdm(series_list):
            try:
                # Get series data
                data = self.fred.get_series(series_id, start_date, end_date)
                
                if not data.empty:
                    # Remove weekends by resampling to business days
                    data = data.asfreq('B', method='ffill')
                    
                    # Store in dictionary
                    all_data[series_id] = data
                else:
                    print(f"No data for series {series_id}")
            except Exception as e:
                print(f"Error fetching series {series_id}: {e}")
        
        # Combine all series into one DataFrame
        if all_data:
            df = pd.DataFrame(all_data)
            print(f"Got data with {len(df)} rows and {len(df.columns)} columns")
            return df
        else:
            print("No data fetched")
            return pd.DataFrame()
    
    def get_dummy_macro_data(self, start_date, end_date=None, num_series=10):
        """
        Generate dummy macro data for testing
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            num_series: Number of series to generate
            
        Returns:
            DataFrame with dummy data
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
            
        # Create date range (business days only)
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')
        
        # Generate dummy data
        data = {}
        
        # Common macro series types
        series_types = [
            'GDP', 'CPI', 'UNEMPLOYMENT', 'INTEREST_RATE', 'PMI', 
            'INDUSTRIAL_PRODUCTION', 'HOUSING_STARTS', 'RETAIL_SALES',
            'CONSUMER_SENTIMENT', 'TRADE_BALANCE'
        ]
        
        for i in range(min(num_series, len(series_types))):
            series_id = series_types[i]
            
            # Generate series with different characteristics
            if series_id == 'INTEREST_RATE':
                # Step-like changes
                base = 2.5
                steps = np.random.randint(-1, 2, size=len(date_range) // 30)
                steps = np.repeat(steps, 30)[:len(date_range)] * 0.25
                cumulative_steps = np.cumsum(steps)
                series = base + cumulative_steps
                
            elif series_id == 'CPI':
                # Slowly increasing with small monthly jumps
                base = 100
                monthly_increase = 0.2 + np.random.randn(len(date_range) // 20) * 0.1
                monthly_increase = np.repeat(monthly_increase, 20)[:len(date_range)]
                series = base + np.cumsum(monthly_increase)
                
            elif series_id == 'UNEMPLOYMENT':
                # Cyclical with noise
                base = 5.0
                t = np.linspace(0, 4*np.pi, len(date_range))
                series = base + np.sin(t) + np.random.randn(len(date_range)) * 0.2
                series = np.maximum(series, 3.0)  # Unemployment can't go below 3%
                
            else:
                # Generic series with trend and noise
                base = 100
                trend = np.linspace(0, 10, len(date_range))
                noise = np.random.randn(len(date_range)) * 5
                series = base + trend + noise
            
            data[series_id] = pd.Series(series, index=date_range)
        
        return pd.DataFrame(data)
    
    def compute_derivative_features(self, macro_df):
        """
        Compute derivative features from macro data
        
        Args:
            macro_df: DataFrame with macro data
            
        Returns:
            DataFrame with original and derivative features
        """
        result = macro_df.copy()
        
        print("Computing derivative macro features...")
        
        # For each series, compute:
        # 1. Percentage change (1-day, 5-day, 20-day)
        # 2. Z-score over different windows
        # 3. Rolling averages
        for column in macro_df.columns:
            # Percentage changes
            result[f'{column}_pct_1d'] = macro_df[column].pct_change(1)
            result[f'{column}_pct_5d'] = macro_df[column].pct_change(5)
            result[f'{column}_pct_20d'] = macro_df[column].pct_change(20)
            
            # Z-scores
            for window in [10, 20, 60]:
                rolling = macro_df[column].rolling(window=window)
                result[f'{column}_zscore_{window}d'] = (macro_df[column] - rolling.mean()) / rolling.std()
            
            # Moving averages
            for window in [5, 10, 20, 60]:
                result[f'{column}_ma_{window}d'] = macro_df[column].rolling(window=window).mean()
                
                # MA crossovers (1 if short MA > long MA, 0 otherwise)
                if window > 5:
                    result[f'{column}_cross_5_{window}'] = (
                        result[f'{column}_ma_5d'] > result[f'{column}_ma_{window}d']
                    ).astype(int)
            
            # Momentum features
            result[f'{column}_mom_20d'] = macro_df[column].diff(20)
            result[f'{column}_mom_60d'] = macro_df[column].diff(60)
            
            # Acceleration (change in momentum)
            result[f'{column}_accel'] = result[f'{column}_mom_20d'].diff(5)
        
        # Cross-asset features
        if 'INTEREST_RATE' in macro_df.columns and 'CPI' in macro_df.columns:
            # Real interest rate approximation
            result['REAL_RATE'] = macro_df['INTEREST_RATE'] - macro_df['CPI_pct_1d'] * 100
        
        # Fill NaN values
        result = result.replace([np.inf, -np.inf], np.nan)
        result = result.fillna(method='ffill').fillna(method='bfill')
        
        print(f"Generated {len(result.columns) - len(macro_df.columns)} derivative macro features")
        
        return result
    
    def create_attention_masks(self, macro_df, market_dates):
        """
        Create attention masks for macro data
        
        The mask indicates when the data was actually updated
        (many macro series are only updated monthly or quarterly)
        
        Args:
            macro_df: DataFrame with macro data
            market_dates: DatetimeIndex of market dates
            
        Returns:
            DataFrame with attention masks (1 for updated data, 0 otherwise)
        """
        # Reindex macro data to market dates
        aligned_df = macro_df.reindex(market_dates, method='ffill')
        
        # Initialize mask DataFrame with zeros
        masks = pd.DataFrame(0, index=market_dates, columns=macro_df.columns)
        
        # For each series, set mask to 1 on days where the data changes
        for column in macro_df.columns:
            # Get differences (where data changes)
            diff = aligned_df[column].diff() != 0
            
            # Set mask to 1 where data changes
            masks.loc[diff, column] = 1
            
            # First row should be 1 (initial data)
            if len(masks) > 0:
                masks.iloc[0, masks.columns.get_loc(column)] = 1
        
        return masks