import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
from tqdm import tqdm


class ARIMAModel:
    """
    ARIMA baseline model for time series forecasting
    """
    def __init__(self, p=5, d=1, q=0):
        self.p = p
        self.d = d
        self.q = q
        self.models = {}
        self.order = (p, d, q)
        
    def optimize_parameters(self, series, max_p=5, max_d=2, max_q=2):
        """
        Find optimal ARIMA parameters using AIC
        
        Args:
            series: Time series data
            max_p, max_d, max_q: Maximum values to try
            
        Returns:
            Tuple of optimal (p, d, q) parameters
        """
        best_aic = float('inf')
        best_order = None
        
        # Try different combinations of parameters
        for p in range(max_p + 1):
            for d in range(max_d + 1):
                for q in range(max_q + 1):
                    if p == 0 and q == 0:
                        continue  # Skip invalid model
                        
                    try:
                        model = ARIMA(series, order=(p, d, q))
                        result = model.fit()
                        aic = result.aic
                        
                        if aic < best_aic:
                            best_aic = aic
                            best_order = (p, d, q)
                            
                    except Exception as e:
                        continue  # Skip if model fails to converge
        
        if best_order is None:
            # Fallback to default
            best_order = (self.p, self.d, self.q)
            
        return best_order
    
    def fit(self, data, ticker=None, optimize=False):
        """
        Fit ARIMA model to time series
        
        Args:
            data: Time series data (numpy array or pandas Series)
            ticker: Identifier for the time series
            optimize: Whether to optimize ARIMA parameters
            
        Returns:
            Fitted model
        """
        if ticker is None:
            ticker = 'series'
            
        try:
            # Convert to pandas Series if numpy array
            if isinstance(data, np.ndarray):
                data = pd.Series(data)
            
            # Optimize parameters if requested
            if optimize:
                order = self.optimize_parameters(data)
            else:
                order = self.order
                
            # Fit model
            model = ARIMA(data, order=order)
            fitted_model = model.fit()
            
            # Store model
            self.models[ticker] = {
                'model': fitted_model,
                'order': order
            }
            
            return fitted_model
            
        except Exception as e:
            print(f"Error fitting ARIMA model for {ticker}: {e}")
            return None
    
    def predict(self, ticker=None, steps=1):
        """
        Generate predictions
        
        Args:
            ticker: Identifier for the time series
            steps: Number of steps to forecast
            
        Returns:
            Forecasted values
        """
        if ticker is None:
            ticker = 'series'
            
        if ticker not in self.models:
            raise ValueError(f"No model fitted for {ticker}")
            
        try:
            # Generate forecast
            model = self.models[ticker]['model']
            forecast = model.forecast(steps=steps)
            
            return forecast
            
        except Exception as e:
            print(f"Error generating forecast for {ticker}: {e}")
            return None
    
    def batch_fit_predict(self, data_dict, optimize=False, steps=1):
        """
        Fit models and generate predictions for multiple time series
        
        Args:
            data_dict: Dictionary mapping tickers to time series
            optimize: Whether to optimize ARIMA parameters
            steps: Number of steps to forecast
            
        Returns:
            Dictionary of predictions
        """
        predictions = {}
        
        for ticker, data in tqdm(data_dict.items(), desc="ARIMA fitting"):
            # Fit model
            model = self.fit(data, ticker, optimize)
            
            if model is not None:
                # Generate forecast
                forecast = self.predict(ticker, steps)
                predictions[ticker] = forecast
                
        return predictions


class BiLSTMModel(nn.Module):
    """
    Bidirectional LSTM model for time series prediction
    """
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # BiLSTM layer
        self.bilstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output layer
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            
        Returns:
            Predictions [batch_size, 1]
        """
        # BiLSTM
        output, (hidden, cell) = self.bilstm(x)
        
        # Get final hidden state from both directions
        final_hidden_fw = hidden[-2]  # Forward direction
        final_hidden_bw = hidden[-1]  # Backward direction
        
        # Concatenate
        final_hidden = torch.cat([final_hidden_fw, final_hidden_bw], dim=1)
        
        # Project to output
        predictions = self.projection(final_hidden)
        
        return predictions


class TransformerEncoder(nn.Module):
    """
    Transformer encoder for text data
    """
    def __init__(self, embedding_dim, num_heads=4, num_layers=2, dropout=0.2):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        
        # Positional encoding
        self.pos_encoder = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
        
        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        
        # Stack encoder layers
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )
        
        # Output layer
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, 1)
        )
        
    def forward(self, x, src_mask=None, src_key_padding_mask=None):
        """
        Forward pass
        
        Args:
            x: Input tensor [batch_size, seq_len, embedding_dim]
            src_mask: Self-attention mask
            src_key_padding_mask: Key padding mask
            
        Returns:
            Predictions [batch_size, 1]
        """
        # Apply positional encoding
        x = self.pos_encoder(x)
        
        # Pass through transformer
        encoded = self.transformer_encoder(x, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        
        # Mean pooling over sequence dimension
        pooled = torch.mean(encoded, dim=1)
        
        # Project to output
        predictions = self.projection(pooled)
        
        return predictions


class HybridModel(nn.Module):
    """
    Hybrid model combining BiLSTM and Transformer
    """
    def __init__(self, ts_input_dim, text_embedding_dim, hidden_dim=64, num_heads=4, num_layers=2, dropout=0.2):
        super().__init__()
        
        self.ts_input_dim = ts_input_dim
        self.text_embedding_dim = text_embedding_dim
        self.hidden_dim = hidden_dim
        
        # Time-series branch with BiLSTM
        self.bilstm = nn.LSTM(
            input_size=ts_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Text branch with Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=text_embedding_dim,
            nhead=num_heads,
            dim_feedforward=text_embedding_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )
        
        # Fusion layer
        fusion_input_dim = (hidden_dim * 2) + text_embedding_dim
        
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, ts_data, text_data, text_mask=None):
        """
        Forward pass
        
        Args:
            ts_data: Time series input [batch_size, seq_len, ts_input_dim]
            text_data: Text embeddings [batch_size, seq_len, text_embedding_dim]
            text_mask: Mask for transformer
            
        Returns:
            Predictions [batch_size, 1]
        """
        # Process time-series with BiLSTM
        ts_output, (hidden, cell) = self.bilstm(ts_data)
        
        # Get final hidden state from both directions
        final_hidden_fw = hidden[-2]  # Forward direction
        final_hidden_bw = hidden[-1]  # Backward direction
        
        # Concatenate bidirectional hidden states
        ts_features = torch.cat([final_hidden_fw, final_hidden_bw], dim=1)
        
        # Process text with transformer
        text_output = self.transformer(text_data, mask=text_mask)
        
        # Mean pooling over sequence dimension
        text_features = torch.mean(text_output, dim=1)
        
        # Concatenate features from both branches
        combined = torch.cat([ts_features, text_features], dim=1)
        
        # Generate predictions
        predictions = self.fusion(combined)
        
        return predictions