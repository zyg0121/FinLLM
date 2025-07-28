import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer models
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Add positional encoding to input
        
        Args:
            x: Tensor of shape [seq_len, batch_size, embedding_dim]
            
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:x.size(0), :]
        return x


class SelfAttentionBlock(nn.Module):
    """
    Self-attention block with multi-head attention and feed-forward layers
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        
        # Multi-head attention
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        Forward pass through the self-attention block
        
        Args:
            src: Input tensor [seq_len, batch_size, embed_dim]
            src_mask: Mask for self-attention
            src_key_padding_mask: Padding mask
            
        Returns:
            Output tensor and attention weights
        """
        # Self-attention with residual connection and normalization
        src2, attn_weights = self.self_attn(
            query=src, 
            key=src, 
            value=src, 
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # Feed-forward with residual connection and normalization
        src2 = self.feed_forward(src)
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src, attn_weights


class TimeSeries_BiLSTM_Module(nn.Module):
    """
    BiLSTM module for time-series feature extraction
    """
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        
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
        
        # Output size is doubled due to bidirectional
        self.output_dim = hidden_dim * 2
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        for name, param in self.bilstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)
        
    def forward(self, x):
        """
        Forward pass through BiLSTM
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            
        Returns:
            Output sequence and final hidden state
        """
        # Run through BiLSTM
        output, (hidden, cell) = self.bilstm(x)
        
        # Concatenate the final hidden states from both directions
        # hidden shape: [num_layers * num_directions, batch_size, hidden_dim]
        batch_size = x.size(0)
        
        # Get the last layer's hidden states from both directions
        last_layer_h_fw = hidden[2*self.num_layers-2]  # Forward direction, last layer
        last_layer_h_bw = hidden[2*self.num_layers-1]  # Backward direction, last layer
        
        # Concatenate forward and backward hidden states
        final_hidden = torch.cat([last_layer_h_fw, last_layer_h_bw], dim=1)
        
        return output, final_hidden


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention mechanism for fusing time-series and text features
    """
    def __init__(self, ts_dim, text_dim, output_dim, num_heads=4, dropout=0.1):
        super().__init__()
        
        self.ts_dim = ts_dim
        self.text_dim = text_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        
        # Project time-series features to match output dimension
        self.ts_projection = nn.Linear(ts_dim, output_dim)
        
        # Project text features to match output dimension
        self.text_projection = nn.Linear(text_dim, output_dim)
        
        # Multi-head attention for cross-modal fusion
        self.cross_attention = nn.MultiheadAttention(output_dim, num_heads, dropout=dropout)
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(output_dim, output_dim * 2),
            nn.ReLU(),
            nn.Linear(output_dim * 2, output_dim)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(output_dim)
        self.norm2 = nn.LayerNorm(output_dim)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # Learnable fusion scalar
        self.fusion_scalar = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, ts_features, text_features):
        """
        Fuse time-series and text features with cross-modal attention
        
        Args:
            ts_features: Time-series features [batch_size, ts_dim]
            text_features: Text features [seq_len, batch_size, text_dim]
            
        Returns:
            Fused features and attention weights
        """
        batch_size = ts_features.size(0)
        
        # Project time-series features
        ts_proj = self.ts_projection(ts_features)  # [batch_size, output_dim]
        
        # Project text features
        text_seq_len = text_features.size(0)
        text_proj = self.text_projection(text_features)  # [seq_len, batch_size, output_dim]
        
        # Reshape ts_features for attention input (seq_len=1)
        ts_proj = ts_proj.unsqueeze(0)  # [1, batch_size, output_dim]
        
        # Cross-attention: ts_features attends to text_features
        attn_output, attn_weights = self.cross_attention(
            query=ts_proj,
            key=text_proj,
            value=text_proj
        )
        
        # Apply fusion scalar to control the contribution of each modality
        alpha = torch.sigmoid(self.fusion_scalar)
        
        # Residual connection and normalization
        fused = alpha * ts_proj + (1 - alpha) * attn_output
        fused = self.norm1(fused + self.dropout1(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(fused)
        fused = self.norm2(fused + self.dropout2(ff_output))
        
        # Remove sequence dimension (seq_len=1)
        fused = fused.squeeze(0)
        
        return fused, attn_weights, alpha


class RiskAwareHead(nn.Module):
    """
    Risk-aware prediction head with Expected Shortfall objective
    """
    def __init__(self, input_dim, dropout=0.1, alpha=0.95):
        super().__init__()
        
        self.alpha = alpha  # Confidence level for ES
        
        # Mean prediction branch
        self.mean_branch = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, 1)
        )
        
        # Scale prediction branch (for uncertainty/volatility)
        self.scale_branch = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, 1),
            nn.Softplus()  # Ensure positive scale
        )
        
    def forward(self, x):
        """
        Predict mean and scale
        
        Args:
            x: Input features [batch_size, input_dim]
            
        Returns:
            Dictionary with mean and scale predictions
        """
        mean = self.mean_branch(x)
        scale = self.scale_branch(x) + 1e-6  # Add small constant for numerical stability
        
        return {
            'mean': mean,
            'scale': scale
        }
    
    def expected_shortfall_loss(self, mean, scale, targets):
        """
        Compute Expected Shortfall (ES) loss
        
        Args:
            mean: Predicted mean [batch_size, 1]
            scale: Predicted scale [batch_size, 1]
            targets: Target values [batch_size, 1]
            
        Returns:
            Combined loss value
        """
        # Compute standardized residuals
        z = (targets - mean) / scale
        
        # Compute quantile at alpha level
        quantile = torch.quantile(z, 1 - self.alpha)
        
        # Compute ES for points beyond the quantile
        excess_mask = z <= quantile
        if excess_mask.sum() > 0:
            es = torch.mean(z[excess_mask]) * scale.mean() + mean.mean()
            es_loss = torch.abs(es)
        else:
            # Fallback if no points beyond quantile
            es_loss = torch.tensor(0.0, device=mean.device)
        
        # Regular MSE loss
        mse_loss = F.mse_loss(mean, targets)
        
        # Scale prediction loss - penalize large uncertainty
        scale_reg_loss = 0.01 * torch.mean(scale)
        
        # Combined loss
        total_loss = mse_loss + 0.1 * es_loss + scale_reg_loss
        
        return total_loss


class FinLLM(nn.Module):
    """
    Complete FinLLM architecture for stock price prediction
    """
    def __init__(
        self,
        ts_input_dim,
        text_embedding_dim,
        hidden_dim=64,
        output_dim=64,
        num_heads=4,
        num_layers=2,
        dropout=0.2,
        alpha=0.95
    ):
        super().__init__()
        
        self.ts_input_dim = ts_input_dim
        self.text_embedding_dim = text_embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Time-series processing branch
        self.ts_module = TimeSeries_BiLSTM_Module(
            input_dim=ts_input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # Text processing branch
        self.positional_encoding = PositionalEncoding(text_embedding_dim)
        
        self.text_module = nn.ModuleList([
            SelfAttentionBlock(
                embed_dim=text_embedding_dim,
                num_heads=num_heads,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        # Cross-modal fusion
        self.cross_modal_fusion = CrossModalAttention(
            ts_dim=hidden_dim * 2,  # BiLSTM output is 2*hidden_dim
            text_dim=text_embedding_dim,
            output_dim=output_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Final projection
        self.projection = nn.Sequential(
            nn.Linear(output_dim + text_embedding_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Risk-aware prediction head
        self.risk_head = RiskAwareHead(
            input_dim=output_dim,
            dropout=dropout,
            alpha=alpha
        )
        
    def forward(self, ts_data, text_data):
        """
        Forward pass through the FinLLM model
        
        Args:
            ts_data: Time series data [batch_size, seq_len, ts_input_dim]
            text_data: Text embeddings [seq_len, batch_size, text_embedding_dim]
                       Note: seq_len first for text data as it's the transformer convention
            
        Returns:
            Dictionary with predictions and attention weights
        """
        batch_size = ts_data.size(0)
        
        # Process time-series data with BiLSTM
        ts_output, ts_hidden = self.ts_module(ts_data)
        
        # Process text data with self-attention blocks
        text_output = self.positional_encoding(text_data)
        text_attn_weights = []
        
        for layer in self.text_module:
            text_output, attn_weights = layer(text_output)
            text_attn_weights.append(attn_weights)
        
        # Get mean of text embeddings across sequence dimension
        text_mean = torch.mean(text_output, dim=0)
        
        # Fuse time-series and text features
        fused_features, cross_attn_weights, fusion_alpha = self.cross_modal_fusion(
            ts_hidden, text_output
        )
        
        # Concatenate fused features with text mean
        combined = torch.cat([fused_features, text_mean], dim=1)
        
        # Project to final representation
        final_repr = self.projection(combined)
        
        # Get predictions from risk-aware head
        predictions = self.risk_head(final_repr)
        
        # Add attention weights and other metadata to outputs
        predictions.update({
            'ts_hidden': ts_hidden,
            'text_output': text_output,
            'text_attn_weights': text_attn_weights[-1],  # Last layer's attention
            'cross_attn_weights': cross_attn_weights,
            'fusion_alpha': fusion_alpha
        })
        
        return predictions
    
    def compute_loss(self, predictions, targets):
        """
        Compute loss for training
        
        Args:
            predictions: Dictionary with model predictions
            targets: Target values
            
        Returns:
            Loss value
        """
        return self.risk_head.expected_shortfall_loss(
            predictions['mean'],
            predictions['scale'],
            targets
        )