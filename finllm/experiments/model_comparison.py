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

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import from our modules
from finllm.models.core import FinLLM
from finllm.models.baselines import ARIMAModel, BiLSTMModel, TransformerModel, HybridModel
from finllm.features.processing import compute_technical_indicators, prepare_time_series_data
from finllm.evaluation.metrics import calc_daily_ic, calc_ic_metrics, calc_portfolio_performance
from finllm.training.trainer import ModelTrainer


def download_data(tickers, start_date, end_date):
    """
    Download historical data for a list of tickers
    """
    data = {}
    for ticker in tqdm(tickers, desc="Downloading data"):
        try:
            # Download data
            stock_data = yf.download(ticker, start=start_date, end=end_date)
            if len(stock_data) > 0:
                data[ticker] = stock_data
        except Exception as e:
            print(f"Error downloading {ticker}: {e}")
    
    print(f"Downloaded data for {len(data)} out of {len(tickers)} tickers")
    return data


def prepare_datasets(raw_data, window_size=30, test_size=0.2, val_size=0.2):
    """
    Prepare datasets for all tickers
    """
    all_features = {}
    all_targets = {}
    
    # Process each ticker
    for ticker, data in tqdm(raw_data.items(), desc="Processing tickers"):
        # Compute technical indicators
        processed_data = compute_technical_indicators(data)
        
        # Prepare time series data
        X, y = prepare_time_series_data(
            processed_data, 
            window_size=window_size, 
            target_col='target_1d'
        )
        
        if len(X) > window_size:  # Ensure we have enough data
            all_features[ticker] = X
            all_targets[ticker] = y
    
    # Combine all data
    X_all = np.vstack([all_features[ticker] for ticker in all_features])
    y_all = np.concatenate([all_targets[ticker] for ticker in all_features])
    
    # Train/val/test split based on chronological order
    n_samples = len(X_all)
    test_idx = int(n_samples * (1 - test_size))
    val_idx = int(n_samples * (1 - test_size - val_size))
    
    X_train = X_all[:val_idx]
    y_train = y_all[:val_idx]
    
    X_val = X_all[val_idx:test_idx]
    y_val = y_all[val_idx:test_idx]
    
    X_test = X_all[test_idx:]
    y_test = y_all[test_idx:]
    
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


def run_experiment():
    """
    Run control experiment to compare different models
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Configuration
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    MAX_EPOCHS = 100
    PATIENCE = 5
    WINDOW_SIZE = 30
    
    # Define S&P 500 tickers (using a smaller subset for experiment)
    tickers = [
        'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NVDA', 'JPM', 
        'V', 'PG', 'UNH', 'HD', 'MA', 'BAC', 'DIS', 'CMCSA', 'VZ', 'ADBE',
        'CRM', 'NFLX', 'INTC', 'PFE', 'KO', 'PEP', 'T'
    ]
    
    # Download data
    start_date = '2018-01-01'
    end_date = '2023-12-31'
    raw_data = download_data(tickers, start_date, end_date)
    
    # Prepare datasets
    train_tensors, val_tensors, test_tensors = prepare_datasets(
        raw_data, 
        window_size=WINDOW_SIZE
    )
    
    # Create data loaders
    train_dataset = TensorDataset(*train_tensors)
    val_dataset = TensorDataset(*val_tensors)
    test_dataset = TensorDataset(*test_tensors)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # Model parameters
    input_dim = train_tensors[0].shape[2]  # Number of features
    
    # Initialize models
    bilstm_model = BiLSTMModel(input_dim=input_dim)
    
    # Train BiLSTM model
    print("\n=== Training BiLSTM Model ===")
    bilstm_trainer = ModelTrainer(
        model=bilstm_model,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        max_epochs=MAX_EPOCHS,
        patience=PATIENCE
    )
    bilstm_history = bilstm_trainer.train_bilstm(train_loader, val_loader)
    
    # Evaluate models on test set
    results = {}
    
    # Evaluate BiLSTM
    bilstm_model.eval()
    bilstm_preds = []
    actual_returns = []
    
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(bilstm_trainer.device)
            
            outputs = bilstm_model(X)
            bilstm_preds.extend(outputs.cpu().numpy().flatten())
            actual_returns.extend(y.numpy().flatten())
    
    # Calculate metrics
    ic = np.corrcoef(bilstm_preds, actual_returns)[0, 1]
    mse = np.mean((np.array(bilstm_preds) - np.array(actual_returns))**2)
    
    results['BiLSTM'] = {
        'IC': ic,
        'MSE': mse,
        'predictions': bilstm_preds,
        'actual': actual_returns
    }
    
    print("\n=== Results ===")
    print(f"BiLSTM: IC = {ic:.4f}, MSE = {mse:.6f}")
    
    # Plot predictions vs actual for a sample
    plt.figure(figsize=(10, 6))
    sample_size = 100
    plt.plot(actual_returns[:sample_size], label='Actual', alpha=0.7)
    plt.plot(bilstm_preds[:sample_size], label='BiLSTM', alpha=0.7)
    plt.legend()
    plt.title('Sample Predictions vs Actual')
    plt.savefig('predictions_vs_actual.png')
    
    # Save results to CSV for further analysis
    results_df = pd.DataFrame({
        'Actual': actual_returns,
        'BiLSTM': bilstm_preds
    })
    results_df.to_csv('experiment_results.csv', index=False)
    
    print("Experiment completed. Results saved to experiment_results.csv")


if __name__ == "__main__":
    run_experiment()