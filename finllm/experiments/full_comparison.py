import os
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import argparse
from datetime import datetime
import json

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import from our modules
from finllm.models.core import FinLLM
from finllm.models.baselines import ARIMAModel, BiLSTMModel, TransformerModel, HybridModel
from finllm.features.processing import compute_technical_indicators, prepare_time_series_data
from finllm.evaluation.metrics import calc_daily_ic, calc_ic_metrics, calc_portfolio_performance
from finllm.training.trainer import ModelTrainer


def download_and_prepare_data(tickers, start_date, end_date, window_size=30):
    """
    Download and prepare data for all models
    """
    # Download data
    raw_data = {}
    for ticker in tqdm(tickers, desc="Downloading data"):
        try:
            stock_data = yf.download(ticker, start=start_date, end=end_date)
            if len(stock_data) > 0:
                raw_data[ticker] = stock_data
        except Exception as e:
            print(f"Error downloading {ticker}: {e}")
    
    print(f"Downloaded data for {len(raw_data)} out of {len(tickers)} tickers")
    
    # Process data
    processed_data = {}
    for ticker, data in tqdm(raw_data.items(), desc="Computing indicators"):
        try:
            processed_data[ticker] = compute_technical_indicators(data)
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
    
    # Split into chronological train/val/test periods
    train_data = {}
    val_data = {}
    test_data = {}
    
    for ticker, df in processed_data.items():
        # Get dates
        dates = df.index.tolist()
        n_samples = len(dates)
        
        # Define split points - 60% train, 20% val, 20% test
        train_idx = int(n_samples * 0.6)
        val_idx = int(n_samples * 0.8)
        
        train_data[ticker] = df.iloc[:train_idx]
        val_data[ticker] = df.iloc[train_idx:val_idx]
        test_data[ticker] = df.iloc[val_idx:]
    
    return train_data, val_data, test_data


def prepare_time_series_datasets(train_data, val_data, test_data, window_size=30):
    """
    Prepare time series datasets for BiLSTM and other models
    """
    X_train_all = []
    y_train_all = []
    X_val_all = []
    y_val_all = []
    X_test_all = []
    y_test_all = []
    
    # Process each ticker
    for ticker in tqdm(train_data.keys(), desc="Preparing time series data"):
        try:
            # Prepare training data
            X_train, y_train = prepare_time_series_data(
                train_data[ticker], 
                window_size=window_size, 
                target_col='target_1d'
            )
            X_train_all.append(X_train)
            y_train_all.append(y_train)
            
            # Prepare validation data
            X_val, y_val = prepare_time_series_data(
                val_data[ticker], 
                window_size=window_size, 
                target_col='target_1d'
            )
            X_val_all.append(X_val)
            y_val_all.append(y_val)
            
            # Prepare test data
            X_test, y_test = prepare_time_series_data(
                test_data[ticker], 
                window_size=window_size, 
                target_col='target_1d'
            )
            X_test_all.append(X_test)
            y_test_all.append(y_test)
            
        except Exception as e:
            print(f"Error preparing time series data for {ticker}: {e}")
    
    # Combine all data
    X_train = np.vstack(X_train_all) if X_train_all else np.array([])
    y_train = np.concatenate(y_train_all) if y_train_all else np.array([])
    
    X_val = np.vstack(X_val_all) if X_val_all else np.array([])
    y_val = np.concatenate(y_val_all) if y_val_all else np.array([])
    
    X_test = np.vstack(X_test_all) if X_test_all else np.array([])
    y_test = np.concatenate(y_test_all) if y_test_all else np.array([])
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    
    # Convert to PyTorch tensors
    train_tensors = (
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train).view(-1, 1)
    )
    
    val_tensors = (
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_val).view(-1, 1)
    )
    
    test_tensors = (
        torch.FloatTensor(X_test),
        torch.FloatTensor(y_test).view(-1, 1)
    )
    
    return train_tensors, val_tensors, test_tensors


def create_dummy_sentiment_data(batch_size, seq_len=14, embedding_dim=768):
    """
    Create dummy sentiment data for testing
    In a real implementation, this would be replaced with actual FinBERT embeddings
    """
    return torch.randn(batch_size, seq_len, embedding_dim)


def evaluate_model_performance(model, test_loader, device, model_type):
    """
    Evaluate model performance on test set
    """
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        if model_type == 'bilstm':
            for X, y in test_loader:
                X = X.to(device)
                outputs = model(X)
                predictions.extend(outputs.cpu().numpy().flatten())
                actuals.extend(y.numpy().flatten())
        
        elif model_type == 'transformer':
            for X, y in test_loader:
                # In a real implementation, X would be text embeddings
                # Here we're just using dummy data of the right shape
                X = create_dummy_sentiment_data(X.shape[0]).to(device)
                outputs = model(X)
                predictions.extend(outputs.cpu().numpy().flatten())
                actuals.extend(y.numpy().flatten())
        
        elif model_type in ['hybrid', 'finllm']:
            for X, y in test_loader:
                X_ts = X.to(device)
                X_text = create_dummy_sentiment_data(X.shape[0]).to(device)
                
                if model_type == 'hybrid':
                    outputs = model(X_ts, X_text)
                    predictions.extend(outputs.cpu().numpy().flatten())
                else:  # finllm
                    outputs = model(X_ts, X_text.permute(1, 0, 2))  # seq_len, batch, dim
                    predictions.extend(outputs['mean'].cpu().numpy().flatten())
                
                actuals.extend(y.numpy().flatten())
    
    # Calculate metrics
    ic = np.corrcoef(predictions, actuals)[0, 1]
    mse = np.mean((np.array(predictions) - np.array(actuals))**2)
    
    return {
        'predictions': predictions,
        'actuals': actuals,
        'ic': ic,
        'mse': mse
    }


def run_full_comparison():
    """
    Run full comparison of all models
    """
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Configuration
    parser = argparse.ArgumentParser(description='FinLLM Experiment')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--window_size', type=int, default=30)
    parser.add_argument('--num_tickers', type=int, default=25)
    parser.add_argument('--start_date', type=str, default='2018-01-01')
    parser.add_argument('--end_date', type=str, default='2023-12-31')
    parser.add_argument('--output_dir', type=str, default='./experiment_results')
    args = parser.parse_args()
    
    # Create output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Define experiment ID
    experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Experiment ID: {experiment_id}")
    
    # Define a subset of S&P 500 tickers
    sp500_tickers = [
        'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NVDA', 'JPM', 
        'V', 'PG', 'UNH', 'HD', 'MA', 'BAC', 'DIS', 'CMCSA', 'VZ', 'ADBE',
        'CRM', 'NFLX', 'INTC', 'PFE', 'KO', 'PEP', 'T'
    ]
    
    # Use specified number of tickers
    tickers = sp500_tickers[:args.num_tickers]
    print(f"Using {len(tickers)} tickers: {tickers}")
    
    # Download and prepare data
    train_data, val_data, test_data = download_and_prepare_data(
        tickers, 
        args.start_date, 
        args.end_date, 
        args.window_size
    )
    
    # Prepare time series datasets
    train_tensors, val_tensors, test_tensors = prepare_time_series_datasets(
        train_data, 
        val_data, 
        test_data, 
        args.window_size
    )
    
    # Create data loaders
    train_dataset = TensorDataset(*train_tensors)
    val_dataset = TensorDataset(*val_tensors)
    test_dataset = TensorDataset(*test_tensors)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Model parameters
    input_dim = train_tensors[0].shape[2]  # Number of features
    text_embedding_dim = 768  # FinBERT embedding dim
    hidden_dim = 64
    num_heads = 4
    num_layers = 2
    dropout = 0.2
    
    # Initialize models
    models = {}
    
    # BiLSTM
    print("\n=== Initializing BiLSTM Model ===")
    models['bilstm'] = BiLSTMModel(input_dim=input_dim, hidden_dim=hidden_dim, 
                                   num_layers=num_layers, dropout=dropout)
    
    # Transformer
    print("\n=== Initializing Transformer Model ===")
    models['transformer'] = TransformerModel(embedding_dim=text_embedding_dim, 
                                           num_heads=num_heads, 
                                           num_layers=num_layers, 
                                           dropout=dropout)
    
    # Hybrid
    print("\n=== Initializing Hybrid Model ===")
    models['hybrid'] = HybridModel(ts_input_dim=input_dim, 
                                  text_embedding_dim=text_embedding_dim, 
                                  hidden_dim=hidden_dim, 
                                  num_heads=num_heads, 
                                  num_layers=num_layers, 
                                  dropout=dropout)
    
    # FinLLM
    print("\n=== Initializing FinLLM Model ===")
    models['finllm'] = FinLLM(ts_input_dim=input_dim, 
                             text_embedding_dim=text_embedding_dim, 
                             hidden_dim=hidden_dim, 
                             num_heads=num_heads, 
                             num_layers=num_layers, 
                             dropout=dropout)
    
    # Train and evaluate models
    results = {}
    
    # Train BiLSTM
    print("\n=== Training BiLSTM Model ===")
    bilstm_trainer = ModelTrainer(
        model=models['bilstm'],
        learning_rate=args.lr,
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        patience=args.patience,
        device=device
    )
    bilstm_history = bilstm_trainer.train_bilstm(train_loader, val_loader)
    
    # Evaluate BiLSTM
    print("\n=== Evaluating BiLSTM Model ===")
    bilstm_results = evaluate_model_performance(
        models['bilstm'], 
        test_loader, 
        device, 
        'bilstm'
    )
    results['bilstm'] = bilstm_results
    
    # Save BiLSTM results
    np.savez(
        os.path.join(args.output_dir, f"{experiment_id}_bilstm_results.npz"),
        predictions=bilstm_results['predictions'],
        actuals=bilstm_results['actuals']
    )
    
    # Save all results to JSON
    summary = {
        'experiment_id': experiment_id,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'config': vars(args),
        'results': {
            'bilstm': {
                'ic': bilstm_results['ic'],
                'mse': bilstm_results['mse']
            }
        }
    }
    
    with open(os.path.join(args.output_dir, f"{experiment_id}_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n=== Results Summary ===")
    print(f"BiLSTM: IC = {bilstm_results['ic']:.4f}, MSE = {bilstm_results['mse']:.6f}")
    print(f"Results saved to {args.output_dir}/{experiment_id}_summary.json")


if __name__ == "__main__":
    run_full_comparison()