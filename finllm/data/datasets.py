import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from sklearn.model_selection import train_test_split


class FinLLMDataset(Dataset):
    """
    PyTorch Dataset for FinLLM model
    """
    def __init__(self, ts_inputs, text_inputs, targets):
        """
        Args:
            ts_inputs: Time series inputs [N, window_size, ts_features]
            text_inputs: Text inputs [N, window_size, embedding_dim]
            targets: Target values [N]
        """
        self.ts_inputs = torch.FloatTensor(ts_inputs)
        self.text_inputs = torch.FloatTensor(text_inputs)
        self.targets = torch.FloatTensor(targets).view(-1, 1)
        
        assert len(self.ts_inputs) == len(self.text_inputs) == len(self.targets), "Input lengths don't match"
        
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        return {
            'ts_input': self.ts_inputs[idx],
            'text_input': self.text_inputs[idx],
            'target': self.targets[idx]
        }


def load_processed_data(processed_dir, ticker):
    """
    Load processed data for a ticker
    
    Args:
        processed_dir: Directory with processed data
        ticker: Ticker symbol
        
    Returns:
        Dictionary with processed data
    """
    data_path = os.path.join(processed_dir, f"{ticker}_processed.npz")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"No processed data found for {ticker}")
    
    data = np.load(data_path)
    
    return {
        'ts_inputs': data['ts_inputs'],
        'text_inputs': data['text_inputs'],
        'targets': data['targets'],
        'dates': data['dates']
    }


def create_dataloaders(processed_dir, tickers, batch_size=32, val_size=0.2, test_size=0.2, random_state=42):
    """
    Create train, validation, and test dataloaders
    
    Args:
        processed_dir: Directory with processed data
        tickers: List of ticker symbols
        batch_size: Batch size
        val_size: Validation set size (proportion of total)
        test_size: Test set size (proportion of total)
        random_state: Random state for reproducibility
        
    Returns:
        Dictionary with dataloaders
    """
    all_ts_inputs = []
    all_text_inputs = []
    all_targets = []
    
    # Load data for all tickers
    for ticker in tickers:
        try:
            data = load_processed_data(processed_dir, ticker)
            
            all_ts_inputs.append(data['ts_inputs'])
            all_text_inputs.append(data['text_inputs'])
            all_targets.append(data['targets'])
            
            print(f"Loaded {len(data['targets'])} samples for {ticker}")
            
        except FileNotFoundError as e:
            print(f"Warning: {str(e)}")
    
    # Combine all data
    ts_inputs = np.vstack(all_ts_inputs)
    text_inputs = np.vstack(all_text_inputs)
    targets = np.concatenate(all_targets)
    
    print(f"Total samples: {len(targets)}")
    
    # Split into train, validation, and test sets
    # First split off test set
    ts_temp, ts_test, text_temp, text_test, y_temp, y_test = train_test_split(
        ts_inputs, text_inputs, targets,
        test_size=test_size,
        random_state=random_state
    )
    
    # Then split remaining data into train and validation
    adjusted_val_size = val_size / (1 - test_size)
    
    ts_train, ts_val, text_train, text_val, y_train, y_val = train_test_split(
        ts_temp, text_temp, y_temp,
        test_size=adjusted_val_size,
        random_state=random_state
    )
    
    print(f"Train samples: {len(y_train)}")
    print(f"Validation samples: {len(y_val)}")
    print(f"Test samples: {len(y_test)}")
    
    # Create datasets
    train_dataset = FinLLMDataset(ts_train, text_train, y_train)
    val_dataset = FinLLMDataset(ts_val, text_val, y_val)
    test_dataset = FinLLMDataset(ts_test, text_test, y_test)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'test_dataset': test_dataset
    }


# Example usage
if __name__ == "__main__":
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    dataloaders = create_dataloaders(
        processed_dir="./processed_data",
        tickers=tickers,
        batch_size=32
    )
    
    # Test a batch
    for batch in dataloaders['train_loader']:
        print("Batch shapes:")
        print(f"Time series input: {batch['ts_input'].shape}")
        print(f"Text input: {batch['text_input'].shape}")
        print(f"Target: {batch['target'].shape}")
        break