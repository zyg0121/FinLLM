import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttentionModule(nn.Module):
    """
    Self-attention mechanism for processing financial text data
    """
    def __init__(self, embedding_dim=768, num_heads=4, dropout=0.2):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        
        # Multi-head attention
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 4, embedding_dim),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: Tensor of shape [seq_len, batch_size, embedding_dim]
            mask: Optional attention mask
        Returns:
            Tensor of shape [seq_len, batch_size, embedding_dim]
        """
        # Self-attention with residual connection
        attn_output, attn_weights = self.multihead_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=mask
        )
        x = self.norm1(x + attn_output)
        
        # Feed-forward with residual
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        
        return x, attn_weights


class TimeSeriesModule(nn.Module):
    """
    Bidirectional LSTM for time-series feature extraction
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
        
        self.output_dim = hidden_dim * 2  # Bidirectional
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, input_dim]
        Returns:
            output: Tensor of shape [batch_size, seq_len, hidden_dim*2]
            hidden: Last hidden state
        """
        output, (hidden, cell) = self.bilstm(x)
        
        # Concatenate the last hidden state from both directions
        last_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        
        return output, last_hidden


class RiskAwareHead(nn.Module):
    """
    Risk-aware prediction head with Expected Shortfall objective
    """
    def __init__(self, input_dim, alpha=0.95):
        super().__init__()
        self.alpha = alpha  # Confidence level for ES (e.g., 0.95 for ES_95)
        
        self.mean_predictor = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Separate branch to predict the scale parameter for ES calculation
        self.scale_predictor = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus()  # Ensure positive scale
        )
    
    def forward(self, x):
        """
        Args:
            x: Feature representation [batch_size, input_dim]
        Returns:
            mean: Predicted mean returns
            scale: Predicted scale (for ES calculation)
        """
        mean = self.mean_predictor(x)
        scale = self.scale_predictor(x)
        
        return mean, scale
    
    def expected_shortfall_loss(self, mean, scale, targets):
        """
        Compute Expected Shortfall loss
        Args:
            mean: Predicted mean returns
            scale: Predicted scale parameter
            targets: Actual returns
        """
        # Compute standardized residuals
        z = (targets - mean) / scale
        
        # ES loss based on the quantile at alpha
        quantile = torch.quantile(z, 1 - self.alpha, dim=0)
        es_loss = scale * torch.mean(z[z <= quantile]) + mean
        
        # Combine with MSE for more stable training
        mse_loss = F.mse_loss(mean, targets)
        
        return mse_loss + 0.1 * torch.abs(es_loss)


class FinLLM(nn.Module):
    """
    Complete FinLLM architecture combining text and time-series data
    """
    def __init__(
        self,
        ts_input_dim=30,      # Technical indicators dimension
        text_embedding_dim=768,  # FinBERT embedding dimension
        hidden_dim=64,
        num_heads=4,
        num_layers=2,
        dropout=0.2,
        alpha=0.95
    ):
        super().__init__()
        
        # Time-series processing branch
        self.ts_module = TimeSeriesModule(
            input_dim=ts_input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # Text processing branch
        self.text_module = SelfAttentionModule(
            embedding_dim=text_embedding_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Cross-modal attention for fusion
        self.cross_modal_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,  # BiLSTM output dimension
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Fusion learnable scalar
        self.fusion_scalar = nn.Parameter(torch.tensor(0.5))
        
        # Final projection layers
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim * 2 + text_embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Risk-aware prediction head
        self.risk_head = RiskAwareHead(input_dim=64, alpha=alpha)
    
    def forward(self, ts_data, text_data):
        """
        Forward pass through the FinLLM model
        
        Args:
            ts_data: Technical indicators [batch_size, seq_len, ts_input_dim]
            text_data: Text embeddings [seq_len, batch_size, text_embedding_dim]
        
        Returns:
            mean: Predicted mean returns
            scale: Predicted scale for risk estimation
            attention_weights: Attention weights for interpretability
        """
        # Process time-series data
        ts_output, ts_hidden = self.ts_module(ts_data)
        
        # Process text data
        text_output, text_attn_weights = self.text_module(text_data)
        
        # Reshape text_output for cross attention
        text_output_mean = text_output.mean(dim=0)  # [batch_size, text_embedding_dim]
        
        # Project ts_hidden to match dimensions if needed
        
        # Fusion with cross-modal attention
        # First, reshape ts_hidden to [1, batch_size, hidden_dim*2]
        ts_hidden_reshaped = ts_hidden.unsqueeze(0)
        
        # Cross-attention between ts_hidden and text_output
        cross_output, cross_attn_weights = self.cross_modal_attention(
            query=ts_hidden_reshaped,
            key=text_output,
            value=text_output
        )
        
        # Combine with fusion scalar
        alpha = torch.sigmoid(self.fusion_scalar)
        fused_repr = alpha * ts_hidden + (1 - alpha) * cross_output.squeeze(0)
        
        # Concatenate with text information
        combined = torch.cat([fused_repr, text_output_mean], dim=1)
        
        # Project to final representation
        final_repr = self.projection(combined)
        
        # Get predictions from risk-aware head
        mean, scale = self.risk_head(final_repr)
        
        return {
            'mean': mean,
            'scale': scale,
            'ts_hidden': ts_hidden,
            'text_attn_weights': text_attn_weights,
            'cross_attn_weights': cross_attn_weights,
            'fusion_alpha': alpha
        }