import torch
import torch.nn as nn
import torch.nn.functional as F
from statsmodels.tsa.arima.model import ARIMA


class ARIMAModel:
    """
    ARIMA baseline model for time-series forecasting
    """
    def __init__(self, p=5, d=1, q=0):
        self.p = p
        self.d = d
        self.q = q
        self.models = {}
    
    def fit(self, series, ticker):
        """
        Fit ARIMA model to a time series
        
        Args:
            series: Time series data
            ticker: Identifier for the series
        """
        try:
            model = ARIMA(series, order=(self.p, self.d, self.q))
            self.models[ticker] = model.fit()
            return True
        except Exception as e:
            print(f"Failed to fit ARIMA for {ticker}: {e}")
            return False
    
    def predict(self, ticker, steps=1):
        """
        Generate predictions
        
        Args:
            ticker: Identifier for the series
            steps: Number of steps to forecast
            
        Returns:
            Forecasted values
        """
        if ticker in self.models:
            forecast = self.models[ticker].forecast(steps=steps)
            return forecast
        else:
            raise ValueError(f"No model fitted for ticker {ticker}")


class BiLSTMModel(nn.Module):
    """
    Bidirectional LSTM model for technical indicators
    """
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        
        self.bilstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, input_dim]
        Returns:
            Predicted returns
        """
        output, (hidden, _) = self.bilstm(x)
        
        # Use the final hidden state from both directions
        final_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        
        # Project to output
        prediction = self.projection(final_hidden)
        
        return prediction


class TransformerModel(nn.Module):
    """
    Transformer model for sentiment data
    """
    def __init__(self, embedding_dim=768, num_heads=4, num_layers=2, dropout=0.2):
        super().__init__()
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )
        
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
    
    def forward(self, x, mask=None):
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, embedding_dim]
            mask: Optional attention mask
        Returns:
            Predicted returns
        """
        # Pass through transformer
        encoded = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        # Average pooling over sequence length
        pooled = encoded.mean(dim=1)
        
        # Project to output
        prediction = self.projection(pooled)
        
        return prediction


class HybridModel(nn.Module):
    """
    Hybrid model combining BiLSTM for technical data and Transformer for sentiment
    """
    def __init__(
        self,
        ts_input_dim,
        text_embedding_dim=768,
        hidden_dim=64,
        num_heads=4,
        num_layers=2,
        dropout=0.2
    ):
        super().__init__()
        
        # BiLSTM branch
        self.bilstm = BiLSTMModel(
            input_dim=ts_input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # Transformer branch
        self.transformer = TransformerModel(
            embedding_dim=text_embedding_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2 + text_embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
    
    def forward(self, ts_data, text_data, mask=None):
        """
        Args:
            ts_data: Technical indicators [batch_size, seq_len, ts_input_dim]
            text_data: Text embeddings [batch_size, seq_len, text_embedding_dim]
            mask: Optional attention mask for text data
        Returns:
            Predicted returns
        """
        # Process time-series with BiLSTM
        output, (hidden, _) = self.bilstm.bilstm(ts_data)
        ts_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        
        # Process text with transformer
        encoded = self.transformer.transformer_encoder(text_data, src_key_padding_mask=mask)
        text_repr = encoded.mean(dim=1)
        
        # Concatenate representations
        combined = torch.cat([ts_hidden, text_repr], dim=1)
        
        # Project to output
        prediction = self.fusion(combined)
        
        return prediction