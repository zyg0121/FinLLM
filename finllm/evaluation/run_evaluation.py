import argparse
import os
import torch
import pandas as pd
import numpy as np
from datetime import datetime
import json
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import FinLLM components
from finllm.models.core_model import FinLLM
from finllm.models.baseline_models import BiLSTMModel, TransformerModel, HybridModel, ARIMAModel
from finllm.evaluation.evaluator import ModelEvaluator
from finllm.evaluation.reporting import FinLLMReportGenerator
from finllm.data.datasets import FinLLMDataset, load_processed_data


def load_model(model_type, model_path, config_path=None, device=None):
    """
    Load a trained model
    
    Args:
        model_type: Type of model to load
        model_path: Path to model weights
        config_path: Path to model configuration
        device: Device to load model on
        
    Returns:
        Loaded model
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model configuration
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        # Default configuration
        config = {
            "ts_input_dim": 30,
            "text_embedding_dim": 768,
            "hidden_dim": 64,
            "output_dim": 64,
            "num_heads": 4,
            "num_layers": 2,
            "dropout": 0.2
        }
    
    # Initialize model based on type
    if model_type == 'finllm':
        model = FinLLM(
            ts_input_dim=config.get('ts_input_dim', 30),
            text_embedding_dim=config.get('text_embedding_dim', 768),
            hidden_dim=config.get('hidden_dim', 64),
            output_dim=config.get('output_dim', 64),
            num_heads=config.get('num_heads', 4),
            num_layers=config.get('num_layers', 2),
            dropout=config.get('dropout', 0.2),
            alpha=config.get('alpha', 0.95)
        )
    elif model_type == 'bilstm':
        model = BiLSTMModel(
            input_dim=config.get('ts_input_dim', 30),
            hidden_dim=config.get('hidden_dim', 64),
            num_layers=config.get('num_layers', 2),
            dropout=config.get('dropout', 0.2)
        )
    elif model_type == 'transformer':
        model = TransformerModel(
            embedding_dim=config.get('text_embedding_dim', 768),
            num_heads=config.get('num_heads', 4),
            num_layers=config.get('num_layers', 2),
            dropout=config.get('dropout', 0.2)
        )
    elif model_type == 'hybrid':
        model = HybridModel(
            ts_input_dim=config.get('ts_input_dim', 30),
            text_embedding_dim=config.get('text_embedding_dim', 768),
            hidden_dim=config.get('hidden_dim', 64),
            num_heads=config.get('num_heads', 4),
            num_layers=config.get('num_layers', 2),
            dropout=config.get('dropout', 0.2)
        )
    elif model_type == 'arima':
        model = ARIMAModel(p=5, d=1, q=0)
        return model  # No need to load weights for ARIMA
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load model weights
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded {model_type} model from {model_path}")
    else:
        print(f"Warning: Model path {model_path} not found")
    
    model.to(device)
    model.eval()
    
    return model


def create_test_loader(data_dir, tickers, batch_size=32):
    """
    Create test data loader from processed data
    
    Args:
        data_dir: Directory with processed data
        tickers: List of tickers to include
        batch_size: Batch size
        
    Returns:
        Test DataLoader
    """
    from torch.utils.data import DataLoader, Dataset
    
    class TestDataset(Dataset):
        def __init__(self, data_dir, tickers):
            self.samples = []
            
            for ticker in tickers:
                try:
                    data = load_processed_data(data_dir, ticker)
                    
                    # Get windows and targets
                    windows = data['windows']
                    targets = data['targets']
                    dates = data['dates']
                    
                    # Append to samples
                    for i in range(len(windows)):
                        self.samples.append({
                            'ts_input': windows[i].get('technical', None),
                            'text_input': windows[i].get('sentiment', None),
                            'target': targets[i],
                            'date': dates[i][1],  # Target date
                            'ticker': ticker
                        })
                        
                except Exception as e:
                    print(f"Error loading data for {ticker}: {e}")
        
        def __len__(self):
            return len(self.samples)
        
        def __getitem__(self, idx):
            return self.samples[idx]
    
    # Create dataset
    dataset = TestDataset(data_dir, tickers)
    
    # Create collate function to handle different sample structures
    def collate_fn(batch):
        # Get all keys in batch
        keys = batch[0].keys()
        result = {}
        
        for key in keys:
            if key in ['ts_input', 'text_input', 'target']:
                # Convert to tensor
                tensors = [torch.FloatTensor(sample[key]) if sample[key] is not None else None for sample in batch]
                
                # Check if all tensors are valid
                if all(t is not None for t in tensors):
                    result[key] = torch.stack(tensors)
                else:
                    # Handle missing data
                    print(f"Warning: Missing {key} data in batch")
                    if key == 'ts_input':
                        # Create dummy tensor
                        result[key] = torch.zeros((len(batch), 30, 30))  # Adjust dimensions as needed
                    elif key == 'text_input':
                        result[key] = torch.zeros((len(batch), 14, 768))  # Adjust dimensions as needed
                    else:
                        result[key] = torch.zeros((len(batch), 1))
            else:
                # Keep non-tensor data as is
                result[key] = [sample[key] for sample in batch]
        
        return result
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    return dataloader


def main():
    parser = argparse.ArgumentParser(description='Evaluate FinLLM models')
    parser.add_argument('--models', nargs='+', required=True, help='List of model types to evaluate')
    parser.add_argument('--model_paths', nargs='+', required=True, help='Paths to model weights')
    parser.add_argument('--config_paths', nargs='+', default=None, help='Paths to model configurations')
    parser.add_argument('--data_dir', required=True, help='Directory with processed test data')
    parser.add_argument('--tickers', nargs='+', required=True, help='List of tickers to evaluate on')
    parser.add_argument('--output_dir', default='./evaluation_results', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--report', action='store_true', help='Generate HTML report')
    
    args = parser.parse_args()
    
    # Validate arguments
    if len(args.models) != len(args.model_paths):
        raise ValueError("Number of models and model_paths must match")
    
    if args.config_paths and len(args.config_paths) != len(args.models):
        raise ValueError("Number of models and config_paths must match")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create test data loader
    test_loader = create_test_loader(args.data_dir, args.tickers, args.batch_size)
    print(f"Created test loader with {len(test_loader.dataset)} samples")
    
    # Evaluate each model
    all_results = []
    
    for i, model_type in enumerate(args.models):
        print(f"\n=== Evaluating {model_type} model ===")
        
        # Load model
        model_path = args.model_paths[i]
        config_path = args.config_paths[i] if args.config_paths else None
        
        model = load_model(model_type, model_path, config_path, device)
        
        # Initialize evaluator
        evaluator = ModelEvaluator(
            model=model,
            model_type=model_type,
            device=device,
            output_dir=args.output_dir
        )
        
        # Evaluate model
        results = evaluator.evaluate_model(test_loader)
        all_results.append(results)
        
        print(f"Completed evaluation for {model_type}")
    
    # Generate comprehensive report if requested
    if args.report:
        print("\n=== Generating evaluation report ===")
        
        report_generator = FinLLMReportGenerator(
            results=all_results,
            output_dir=args.output_dir
        )
        
        # Generate full HTML report
        report_path = report_generator.generate_full_report()
        
        # Export raw results to JSON
        json_path = report_generator.export_to_json()
        
        print(f"Report generated at: {report_path}")
        print(f"Raw results exported to: {json_path}")
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()