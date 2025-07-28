import pandas as pd
import numpy as np
import talib
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, RobustScaler
import warnings
warnings.filterwarnings('ignore')

class TechnicalFeatureProcessor:
    """
    Comprehensive technical feature engineering for the FinLLM model
    """
    def __init__(self, window_sizes=[5, 10, 20, 50, 200]):
        self.window_sizes = window_sizes
        self.scaler = None
    
    def compute_features(self, df, include_targets=True):
        """
        Compute comprehensive technical indicators from OHLCV data
        
        Args:
            df: DataFrame with OHLCV data
            include_targets: Whether to include target variables
            
        Returns:
            DataFrame with technical indicators
        """
        print("Computing technical features...")
        result = df.copy()
        
        # Extract price and volume data
        close = df['Close'].values
        high = df['High'].values
        low = df['Low'].values
        open_price = df['Open'].values
        volume = df['Volume'].values
        
        # === Price Features ===
        
        # Moving Averages for different window sizes
        for window in self.window_sizes:
            result[f'MA{window}'] = talib.SMA(close, timeperiod=window)
            result[f'EMA{window}'] = talib.EMA(close, timeperiod=window)
            
            # Distance from MA (percentage)
            result[f'Dist_MA{window}'] = (close - result[f'MA{window}']) / result[f'MA{window}'] * 100
        
        # Moving Average Crossovers
        result['MA_Cross_5_20'] = (result['MA5'] > result['MA20']).astype(int)
        result['MA_Cross_10_50'] = (result['MA10'] > result['MA50']).astype(int)
        result['MA_Cross_50_200'] = (result['MA50'] > result['MA200']).astype(int)
        
        # Price Momentum Features
        for window in [5, 10, 20]:
            # Return over window
            result[f'Return_{window}d'] = df['Close'].pct_change(window)
            
            # Log return over window
            result[f'Log_Return_{window}d'] = np.log(df['Close']).diff(window)
            
            # Z-score of price over window
            rolling = df['Close'].rolling(window=window)
            result[f'Price_Z_{window}d'] = (df['Close'] - rolling.mean()) / rolling.std()
        
        # Price Gaps
        result['Overnight_Gap'] = (df['Open'] / df['Close'].shift(1) - 1) * 100
        result['Intraday_Change'] = (df['Close'] / df['Open'] - 1) * 100
        
        # High-Low Range
        result['HL_Range'] = (df['High'] - df['Low']) / df['Close'] * 100
        
        # === Volatility Features ===
        
        # Historical Volatility
        for window in [5, 10, 20, 50]:
            result[f'Volatility_{window}d'] = df['Close'].pct_change().rolling(window).std() * np.sqrt(252)
        
        # Average True Range
        result['ATR_14'] = talib.ATR(high, low, close, timeperiod=14)
        result['ATR_Pct'] = result['ATR_14'] / close * 100
        
        # Bollinger Bands
        for window in [10, 20]:
            upper, middle, lower = talib.BBANDS(close, timeperiod=window, nbdevup=2, nbdevdn=2)
            result[f'BB_Upper_{window}'] = upper
            result[f'BB_Middle_{window}'] = middle
            result[f'BB_Lower_{window}'] = lower
            result[f'BB_Width_{window}'] = (upper - lower) / middle
            
            # BB Position - where is price within the bands (0-1)
            result[f'BB_Pos_{window}'] = (close - lower) / (upper - lower)
        
        # === Momentum Indicators ===
        
        # RSI
        for window in [7, 14, 21]:
            result[f'RSI_{window}'] = talib.RSI(close, timeperiod=window)
        
        # MACD
        macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        result['MACD'] = macd
        result['MACD_Signal'] = macd_signal
        result['MACD_Hist'] = macd_hist
        result['MACD_Hist_Diff'] = macd_hist - macd_hist.shift(1)
        
        # Stochastic Oscillator
        for period in [14, 21]:
            slowk, slowd = talib.STOCH(high, low, close, fastk_period=period, 
                                       slowk_period=3, slowk_matype=0, 
                                       slowd_period=3, slowd_matype=0)
            result[f'SlowK_{period}'] = slowk
            result[f'SlowD_{period}'] = slowd
            result[f'Stoch_Cross_{period}'] = (slowk > slowd).astype(int)
        
        # Williams %R
        for period in [14, 28]:
            result[f'Williams_R_{period}'] = talib.WILLR(high, low, close, timeperiod=period)
        
        # Commodity Channel Index (CCI)
        for period in [14, 20, 40]:
            result[f'CCI_{period}'] = talib.CCI(high, low, close, timeperiod=period)
        
        # Average Directional Index (ADX)
        result['ADX_14'] = talib.ADX(high, low, close, timeperiod=14)
        result['ADX_Trend'] = (result['ADX_14'] > 25).astype(int)
        
        # Rate of Change
        for period in [5, 10, 20]:
            result[f'ROC_{period}'] = talib.ROC(close, timeperiod=period)
        
        # === Volume Features ===
        
        # Volume Moving Averages
        for window in [5, 10, 20, 50]:
            result[f'Volume_MA_{window}'] = talib.SMA(volume, timeperiod=window)
            
            # Volume Ratio - current volume to moving average
            result[f'Volume_Ratio_{window}'] = volume / result[f'Volume_MA_{window}']
        
        # On-Balance Volume and changes
        result['OBV'] = talib.OBV(close, volume)
        result['OBV_ROC_5'] = talib.ROC(result['OBV'].values, timeperiod=5)
        result['OBV_ROC_10'] = talib.ROC(result['OBV'].values, timeperiod=10)
        
        # Money Flow Index
        result['MFI_14'] = talib.MFI(high, low, close, volume, timeperiod=14)
        
        # Accumulation/Distribution Line
        result['AD'] = talib.AD(high, low, close, volume)
        result['ADOSC'] = talib.ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10)
        
        # Price-Volume Trend
        result['PVT'] = (close.pct_change() * volume).cumsum()
        
        # Chaikin Money Flow
        for period in [10, 20]:
            mfm = ((close - low) - (high - close)) / (high - low)
            mfm = np.where((high - low) == 0, 0, mfm)  # Handle division by zero
            mfv = mfm * volume
            result[f'CMF_{period}'] = mfv.rolling(period).sum() / volume.rolling(period).sum()
        
        # Volume Spike Detection
        result['Volume_Z_Score'] = (volume - volume.rolling(20).mean()) / volume.rolling(20).std()
        result['Volume_Spike'] = (result['Volume_Z_Score'] > 2).astype(int)
        
        # === Pattern Recognition ===
        
        # Candlestick Patterns
        result['Doji'] = talib.CDLDOJI(open_price, high, low, close)
        result['Hammer'] = talib.CDLHAMMER(open_price, high, low, close)
        result['Engulfing'] = talib.CDLENGULFING(open_price, high, low, close)
        result['Morning_Star'] = talib.CDLMORNINGSTAR(open_price, high, low, close)
        result['Evening_Star'] = talib.CDLEVENINGSTAR(open_price, high, low, close)
        
        # Target Features if requested
        if include_targets:
            # Future returns at different horizons
            result['Target_1d'] = result['Close'].pct_change(1).shift(-1)
            result['Target_5d'] = result['Close'].pct_change(5).shift(-5)
            result['Target_10d'] = result['Close'].pct_change(10).shift(-10)
            result['Target_20d'] = result['Close'].pct_change(20).shift(-20)
            
            # Binary targets (up/down)
            result['Target_Direction_1d'] = (result['Target_1d'] > 0).astype(int)
            result['Target_Direction_5d'] = (result['Target_5d'] > 0).astype(int)
        
        # Fill NaN values
        result = result.replace([np.inf, -np.inf], np.nan)
        result = result.fillna(method='ffill').fillna(method='bfill')
        
        print(f"Generated {len(result.columns) - 5} technical features")  # Subtract OHLCV columns
        
        return result
    
    def fit_scaler(self, df):
        """
        Fit scaler to feature data
        
        Args:
            df: DataFrame with features
            
        Returns:
            Self
        """
        # Exclude OHLCV and target columns
        exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        target_cols = [col for col in df.columns if col.startswith('Target_')]
        
        feature_cols = [col for col in df.columns 
                       if col not in exclude_cols and col not in target_cols]
        
        # Use RobustScaler to handle outliers better
        self.scaler = RobustScaler()
        self.scaler.fit(df[feature_cols])
        
        return self
    
    def transform_features(self, df):
        """
        Scale features
        
        Args:
            df: DataFrame with features
            
        Returns:
            DataFrame with scaled features
        """
        if self.scaler is None:
            raise ValueError("Scaler not fitted. Call fit_scaler first.")
            
        # Copy DataFrame
        result = df.copy()
        
        # Exclude OHLCV and target columns
        exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        target_cols = [col for col in df.columns if col.startswith('Target_')]
        
        feature_cols = [col for col in df.columns 
                       if col not in exclude_cols and col not in target_cols]
        
        # Scale features
        result[feature_cols] = self.scaler.transform(df[feature_cols])
        
        return result
    
    def select_features(self, df, n_features=50, method='lightgbm'):
        """
        Select top features using feature importance
        
        Args:
            df: DataFrame with features and target
            n_features: Number of features to select
            method: Feature selection method ('lightgbm' or 'correlation')
            
        Returns:
            List of selected feature names
        """
        from sklearn.feature_selection import SelectKBest, f_regression
        import lightgbm as lgb
        
        # Exclude OHLCV columns
        exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # Get target column
        target_col = 'Target_1d'
        
        # Get feature columns
        feature_cols = [col for col in df.columns 
                       if col not in exclude_cols and not col.startswith('Target_')]
        
        # Drop rows with NaN
        data = df[[*feature_cols, target_col]].dropna()
        
        if method == 'lightgbm':
            # Train LightGBM model for feature importance
            X = data[feature_cols]
            y = data[target_col]
            
            lgb_params = {
                'objective': 'regression',
                'boosting_type': 'gbdt',
                'learning_rate': 0.05,
                'n_estimators': 100,
                'max_depth': 5,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            }
            
            lgb_model = lgb.LGBMRegressor(**lgb_params)
            lgb_model.fit(X, y)
            
            # Get feature importance
            importance = lgb_model.feature_importances_
            feature_importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            # Select top features
            selected_features = feature_importance.head(n_features)['feature'].tolist()
            
        elif method == 'correlation':
            # Use correlation with target
            corr = data.corr()[target_col].abs().sort_values(ascending=False)
            selected_features = corr[1:n_features+1].index.tolist()  # Skip target itself
        
        else:
            # Use f_regression for univariate feature selection
            X = data[feature_cols]
            y = data[target_col]
            
            selector = SelectKBest(f_regression, k=n_features)
            selector.fit(X, y)
            
            # Get selected feature indices
            selected_indices = selector.get_support(indices=True)
            selected_features = [feature_cols[i] for i in selected_indices]
        
        print(f"Selected {len(selected_features)} features")
        return selected_features