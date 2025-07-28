import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import lime
import lime.lime_tabular
from sklearn.pipeline import Pipeline
import torch


class ModelExplainer:
    """
    Tools for model explainability and interpretation
    """
    
    def __init__(self, model, model_type, feature_names=None, device=None):
        """
        Initialize model explainer
        
        Args:
            model: Trained model
            model_type: Model type ('finllm', 'bilstm', 'transformer', 'hybrid')
            feature_names: List of feature names
            device: Device for PyTorch models
        """
        self.model = model
        self.model_type = model_type
        self.feature_names = feature_names
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # Move model to device and set to eval mode if it's a PyTorch model
        if isinstance(model, torch.nn.Module):
            self.model.to(self.device)
            self.model.eval()
    
    def create_prediction_function(self):
        """
        Create a prediction function that works with numpy arrays
        
        Returns:
            Function that takes numpy input and returns numpy predictions
        """
        if self.model_type == 'finllm':
            def predict_fn(x):
                # Convert to PyTorch tensors
                if isinstance(x, np.ndarray):
                    # Assume the first half is ts_input and second half is text_input
                    mid_idx = x.shape[1] // 2
                    ts_input = torch.FloatTensor(x[:, :mid_idx]).to(self.device)
                    text_input = torch.FloatTensor(x[:, mid_idx:]).to(self.device)
                    
                    # For FinLLM, text_input needs to be [seq_len, batch, dim]
                    # Reshape assuming text_input is [batch, seq_len * dim]
                    seq_len = 14  # Typical window size
                    text_dim = text_input.shape[1] // seq_len
                    text_input = text_input.reshape(-1, seq_len, text_dim)
                    text_input = text_input.permute(1, 0, 2)
                    
                    # Forward pass
                    with torch.no_grad():
                        outputs = self.model(ts_input, text_input)
                    
                    # Return mean predictions
                    return outputs['mean'].cpu().numpy()
                else:
                    raise ValueError("Input must be a numpy array")
        
        elif self.model_type == 'bilstm':
            def predict_fn(x):
                if isinstance(x, np.ndarray):
                    x = torch.FloatTensor(x).to(self.device)
                    with torch.no_grad():
                        outputs = self.model(x)
                    return outputs.cpu().numpy()
                else:
                    raise ValueError("Input must be a numpy array")
        
        elif self.model_type == 'transformer':
            def predict_fn(x):
                if isinstance(x, np.ndarray):
                    x = torch.FloatTensor(x).to(self.device)
                    with torch.no_grad():
                        outputs = self.model(x)
                    return outputs.cpu().numpy()
                else:
                    raise ValueError("Input must be a numpy array")
        
        elif self.model_type == 'hybrid':
            def predict_fn(x):
                if isinstance(x, np.ndarray):
                    # Assume the first half is ts_input and second half is text_input
                    mid_idx = x.shape[1] // 2
                    ts_input = torch.FloatTensor(x[:, :mid_idx]).to(self.device)
                    text_input = torch.FloatTensor(x[:, mid_idx:]).to(self.device)
                    
                    # Forward pass
                    with torch.no_grad():
                        outputs = self.model(ts_input, text_input)
                    
                    return outputs.cpu().numpy()
                else:
                    raise ValueError("Input must be a numpy array")
        
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        return predict_fn
    
    def shap_analysis(self, background_data, test_data):
        """
        Perform SHAP analysis for model interpretability
        
        Args:
            background_data: Background data for explainer
            test_data: Test data to explain
            
        Returns:
            SHAP explainer and values
        """
        # Create prediction function
        predict_fn = self.create_prediction_function()
        
        # Create explainer
        explainer = shap.KernelExplainer(predict_fn, background_data)
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(test_data)
        
        return {
            "explainer": explainer,
            "shap_values": shap_values
        }
    
    def plot_shap_summary(self, shap_results):
        """
        Plot SHAP summary
        
        Args:
            shap_results: Results from shap_analysis
        """
        plt.figure(figsize=(12, 8))
        
        # Check if feature names are available
        if self.feature_names is not None and len(self.feature_names) == shap_results["shap_values"].shape[1]:
            shap.summary_plot(shap_results["shap_values"], features=self.feature_names)
        else:
            shap.summary_plot(shap_results["shap_values"])
    
    def lime_analysis(self, instance, background_data, num_features=10):
        """
        Perform LIME analysis for a specific instance
        
        Args:
            instance: Instance to explain
            background_data: Background data for the explainer
            num_features: Number of features to include in explanation
            
        Returns:
            LIME explainer and explanation
        """
        # Create prediction function
        predict_fn = self.create_prediction_function()
        
        # Create LIME explainer
        if self.feature_names is not None:
            explainer = lime.lime_tabular.LimeTabularExplainer(
                background_data,
                feature_names=self.feature_names,
                mode='regression'
            )
        else:
            explainer = lime.lime_tabular.LimeTabularExplainer(
                background_data,
                mode='regression'
            )
        
        # Generate explanation
        explanation = explainer.explain_instance(
            instance,
            predict_fn,
            num_features=num_features
        )
        
        return {
            "explainer": explainer,
            "explanation": explanation
        }
    
    def plot_lime_explanation(self, lime_results):
        """
        Plot LIME explanation
        
        Args:
            lime_results: Results from lime_analysis
        """
        lime_results["explanation"].show_in_notebook()
    
    def visualize_attention_weights(self, text_input, ts_input=None):
        """
        Visualize attention weights for transformer-based models
        
        Args:
            text_input: Text input tensor
            ts_input: Time series input tensor (required for FinLLM and Hybrid)
            
        Returns:
            Dictionary with attention visualizations
        """
        if self.model_type not in ['transformer', 'hybrid', 'finllm']:
            print("Attention visualization is only available for transformer-based models")
            return None
        
        # Move inputs to device
        text_input = text_input.to(self.device)
        if ts_input is not None:
            ts_input = ts_input.to(self.device)
        
        with torch.no_grad():
            if self.model_type == 'transformer':
                # For transformer model
                outputs = self.model(text_input)
                
                # Extract attention weights (depends on model implementation)
                attn_weights = None  # Modify based on model implementation
                
            elif self.model_type == 'hybrid':
                # For hybrid model
                outputs = self.model(ts_input, text_input)
                
                # Extract attention weights (depends on model implementation)
                attn_weights = None  # Modify based on model implementation
                
            elif self.model_type == 'finllm':
                # For FinLLM model
                # Text input should be [seq_len, batch, dim] for transformer
                if len(text_input.shape) == 3:
                    text_input = text_input.permute(1, 0, 2)
                
                outputs = self.model(ts_input, text_input)
                
                # Extract attention weights
                text_attn_weights = outputs['text_attn_weights']
                cross_attn_weights = outputs['cross_attn_weights']
                
                attn_weights = {
                    'text': text_attn_weights.cpu().numpy(),
                    'cross': cross_attn_weights.cpu().numpy()
                }
        
        return attn_weights
    
    def plot_attention_heatmap(self, attention_weights, title='Attention Weights', layer_idx=0):
        """
        Plot attention weights as a heatmap
        
        Args:
            attention_weights: Attention weights (dictionary or tensor)
            title: Plot title
            layer_idx: Index of attention layer to visualize (if multiple layers)
        """
        plt.figure(figsize=(10, 8))
        
        if isinstance(attention_weights, dict):
            # For FinLLM with multiple attention types
            if 'text' in attention_weights and 'cross' in attention_weights:
                plt.subplot(1, 2, 1)
                text_attn = attention_weights['text']
                
                # For multi-head attention, average across heads
                if len(text_attn.shape) > 2:
                    text_attn = np.mean(text_attn, axis=0)
                
                sns.heatmap(text_attn, cmap='viridis')
                plt.title('Text Self-Attention')
                
                plt.subplot(1, 2, 2)
                cross_attn = attention_weights['cross']
                
                # For multi-head attention, average across heads
                if len(cross_attn.shape) > 2:
                    cross_attn = np.mean(cross_attn, axis=0)
                
                sns.heatmap(cross_attn, cmap='viridis')
                plt.title('Cross-Modal Attention')
            else:
                # Generic dictionary case
                for i, (name, attn) in enumerate(attention_weights.items()):
                    if len(attention_weights) > 1:
                        plt.subplot(1, len(attention_weights), i+1)
                    
                    # For multi-head attention, average across heads
                    if len(attn.shape) > 2:
                        attn = np.mean(attn, axis=0)
                    
                    sns.heatmap(attn, cmap='viridis')
                    plt.title(f'{name} Attention')
        else:
            # Single attention tensor
            # For multi-head attention, average across heads
            if len(attention_weights.shape) > 2:
                attention_weights = np.mean(attention_weights, axis=0)
            
            sns.heatmap(attention_weights, cmap='viridis')
            plt.title(title)
        
        plt.tight_layout()
        plt.show()
    
    def visualize_risk_aware_predictions(self, ts_input, text_input, targets=None):
        """
        Visualize predictions from FinLLM's risk-aware head
        
        Args:
            ts_input: Time series input tensor
            text_input: Text input tensor
            targets: Optional target values
            
        Returns:
            Dictionary with visualization data
        """
        if self.model_type != 'finllm':
            print("Risk-aware predictions are only available for FinLLM model")
            return None
        
        # Move inputs to device
        ts_input = ts_input.to(self.device)
        text_input = text_input.to(self.device)
        
        # For FinLLM, text_input should be [seq_len, batch, dim]
        if len(text_input.shape) == 3:
            text_input = text_input.permute(1, 0, 2)
        
        with torch.no_grad():
            # Get predictions
            outputs = self.model(ts_input, text_input)
            
            # Extract mean and scale predictions
            means = outputs['mean'].cpu().numpy()
            scales = outputs['scale'].cpu().numpy()
            
            # Calculate prediction intervals
            lower_bound = means - 1.96 * scales
            upper_bound = means + 1.96 * scales
            
            # Plot results
            plt.figure(figsize=(12, 6))
            
            x = np.arange(len(means))
            plt.errorbar(x, means, yerr=1.96*scales, fmt='o', capsize=5, 
                        label='Mean ± 1.96×Scale')
            
            if targets is not None:
                plt.plot(x, targets.cpu().numpy(), 'rx', label='Actual')
            
            plt.legend()
            plt.title('Risk-Aware Predictions with Uncertainty')
            plt.xlabel('Sample Index')
            plt.ylabel('Return')
            plt.grid(True, alpha=0.3)
            plt.show()
            
            return {
                'means': means,
                'scales': scales,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }