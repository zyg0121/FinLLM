import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from captum.attr import IntegratedGradients, LayerConductance, DeepLift
import shap


class ModelInterpreter:
    """
    Interpretability tools for FinLLM and baseline models
    """
    def __init__(self, model, model_type, device=None):
        self.model = model
        self.model_type = model_type
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # Move model to device
        self.model.to(self.device)
        self.model.eval()
    
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
        
        with torch.no_grad():
            if self.model_type == 'transformer':
                # For transformer model
                text_input = text_input.to(self.device)
                outputs = self.model(text_input)
                
                # Extract attention weights (implementation depends on model)
                attn_weights = None  # Need to modify transformer model to return attention
                
            elif self.model_type == 'hybrid':
                # For hybrid model
                ts_input = ts_input.to(self.device)
                text_input = text_input.to(self.device)
                outputs = self.model(ts_input, text_input)
                
                # Extract attention weights (implementation depends on model)
                attn_weights = None  # Need to modify hybrid model to return attention
                
            elif self.model_type == 'finllm':
                # For FinLLM model
                ts_input = ts_input.to(self.device)
                text_input = text_input.to(self.device).permute(1, 0, 2)  # [seq_len, batch, dim]
                outputs = self.model(ts_input, text_input)
                
                # Extract attention weights
                text_attn_weights = outputs['text_attn_weights']
                cross_attn_weights = outputs['cross_attn_weights']
                
                attn_weights = {
                    'text': text_attn_weights.cpu().numpy(),
                    'cross': cross_attn_weights.cpu().numpy()
                }
        
        return attn_weights
    
    def plot_attention_heatmap(self, attention_weights, title='Attention Weights'):
        """
        Plot attention weights as a heatmap
        
        Args:
            attention_weights: Attention weights tensor or numpy array
            title: Plot title
        """
        if isinstance(attention_weights, torch.Tensor):
            attention_weights = attention_weights.cpu().numpy()
        
        # For multi-head attention, average across heads
        if len(attention_weights.shape) > 2:
            attention_weights = np.mean(attention_weights, axis=0)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(attention_weights, cmap='viridis')
        plt.title(title)
        plt.xlabel('Key')
        plt.ylabel('Query')
        plt.show()
    
    def integrated_gradients(self, inputs, target_idx=0, steps=50):
        """
        Compute integrated gradients for input attribution
        
        Args:
            inputs: Dictionary with model inputs
            target_idx: Index of target output (usually 0 for regression)
            steps: Number of steps for path integral
            
        Returns:
            Dictionary with attributions
        """
        # Initialize IntegratedGradients
        ig = IntegratedGradients(self.model_forward_wrapper())
        
        # Create baseline (zeros)
        baselines = {}
        for key, value in inputs.items():
            baselines[key] = torch.zeros_like(value).to(self.device)
        
        # Move inputs to device
        device_inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Compute attributions
        attributions = {}
        for key in inputs:
            # Set up attribution function for this input
            attribution_func = lambda x: self.model_forward_wrapper()(
                {**device_inputs, key: x})[target_idx]
            
            # Compute attributions
            attr = ig.attribute(
                inputs=device_inputs[key],
                baselines=baselines[key],
                target=target_idx,
                n_steps=steps
            )
            
            attributions[key] = attr.cpu().numpy()
        
        return attributions
    
    def model_forward_wrapper(self):
        """
        Create a wrapper for the model's forward function
        
        Returns:
            Wrapped forward function
        """
        if self.model_type == 'bilstm':
            def wrapped_forward(inputs):
                if isinstance(inputs, dict):
                    return self.model(inputs['ts_input'])
                else:
                    return self.model(inputs)
                
        elif self.model_type == 'transformer':
            def wrapped_forward(inputs):
                if isinstance(inputs, dict):
                    return self.model(inputs['text_input'])
                else:
                    return self.model(inputs)
                
        elif self.model_type == 'hybrid':
            def wrapped_forward(inputs):
                if isinstance(inputs, dict):
                    return self.model(inputs['ts_input'], inputs['text_input'])
                else:
                    # Assume inputs is a tuple (ts_input, text_input)
                    return self.model(inputs[0], inputs[1])
                
        elif self.model_type == 'finllm':
            def wrapped_forward(inputs):
                if isinstance(inputs, dict):
                    ts_input = inputs['ts_input']
                    text_input = inputs['text_input']
                    if len(text_input.shape) == 3:
                        # Permute if needed
                        text_input = text_input.permute(1, 0, 2)
                    outputs = self.model(ts_input, text_input)
                    return outputs['mean']
                else:
                    # Assume inputs is a tuple (ts_input, text_input)
                    ts_input, text_input = inputs
                    if len(text_input.shape) == 3:
                        text_input = text_input.permute(1, 0, 2)
                    outputs = self.model(ts_input, text_input)
                    return outputs['mean']
        
        return wrapped_forward
    
    def shap_analysis(self, background_data, test_data, feature_names=None):
        """
        Perform SHAP analysis for model interpretability
        
        Args:
            background_data: Background data for SHAP explainer
            test_data: Test data for explanation
            feature_names: Names of features
            
        Returns:
            SHAP explainer and values
        """
        # Create a function that returns the model output
        def model_fn(x):
            if isinstance(x, np.ndarray):
                x = torch.tensor(x, dtype=torch.float32).to(self.device)
            
            if self.model_type == 'bilstm':
                return self.model(x).cpu().detach().numpy()
            elif self.model_type == 'transformer':
                return self.model(x).cpu().detach().numpy()
            elif self.model_type == 'hybrid':
                # Split input into ts and text parts
                # Assumes the first part is ts, second part is text
                mid_idx = x.shape[1] // 2
                ts_x = x[:, :mid_idx]
                text_x = x[:, mid_idx:]
                return self.model(ts_x, text_x).cpu().detach().numpy()
            elif self.model_type == 'finllm':
                # Similar to hybrid
                mid_idx = x.shape[1] // 2
                ts_x = x[:, :mid_idx]
                text_x = x[:, mid_idx:].permute(1, 0, 2)
                outputs = self.model(ts_x, text_x)
                return outputs['mean'].cpu().detach().numpy()
        
        # Create the explainer
        try:
            explainer = shap.DeepExplainer(model_fn, background_data)
            shap_values = explainer.shap_values(test_data)
            
            if feature_names is not None and len(feature_names) == test_data.shape[1]:
                plt.figure(figsize=(12, 8))
                shap.summary_plot(shap_values, test_data, feature_names=feature_names)
                
            return {
                'explainer': explainer,
                'shap_values': shap_values
            }
            
        except Exception as e:
            print(f"Error in SHAP analysis: {e}")
            return None
    
    def visualize_risk_aware_predictions(self, inputs, targets=None):
        """
        Visualize predictions from the risk-aware head
        
        Args:
            inputs: Dictionary with model inputs
            targets: Optional target values
            
        Returns:
            Dictionary with visualization data
        """
        if self.model_type != 'finllm':
            print("Risk-aware predictions are only available for FinLLM model")
            return None
        
        # Move inputs to device
        device_inputs = {}
        for key, value in inputs.items():
            device_inputs[key] = value.to(self.device)
        
        with torch.no_grad():
            # Get predictions
            ts_input = device_inputs['ts_input']
            text_input = device_inputs['text_input'].permute(1, 0, 2)
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
            plt.plot(x, means, 'b-', label='Predicted Mean')
            plt.fill_between(x, lower_bound.flatten(), upper_bound.flatten(), 
                            alpha=0.3, label='95% Prediction Interval')
            
            if targets is not None:
                plt.plot(x, targets.cpu().numpy(), 'r.', label='Actual')
            
            plt.legend()
            plt.title('Risk-Aware Predictions with Uncertainty')
            plt.xlabel('Sample')
            plt.ylabel('Return')
            plt.show()
            
            return {
                'means': means,
                'scales': scales,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }