import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import os
import json
import time
from datetime import datetime

class ModelTrainer:
    """
    Trainer class for FinLLM and baseline models
    """
    def __init__(
        self,
        model,
        learning_rate=1e-3,
        weight_decay=1e-5,
        batch_size=64,
        max_epochs=100,
        patience=5,
        model_type='finllm',
        device=None,
        checkpoint_dir='./checkpoints',
        log_dir='./logs'
    ):
        self.model = model
        self.model_type = model_type
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        
        # Create directories if they don't exist
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        # Move model to device
        self.model.to(self.device)
        
        # Set optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Set learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            verbose=True
        )
        
        # Set loss function based on model type
        if model_type == 'finllm':
            # FinLLM uses its own loss function
            self.criterion = None
        else:
            # Other models use MSE
            self.criterion = nn.MSELoss()
    
    def train_epoch(self, train_loader):
        """
        Train for one epoch
        
        Args:
            train_loader: DataLoader for training data
            
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Get data and move to device
            if self.model_type == 'bilstm':
                x = batch['ts_input'].to(self.device)
                y = batch['target'].to(self.device)
                
                # Forward pass
                outputs = self.model(x)
                loss = self.criterion(outputs, y)
                
            elif self.model_type == 'transformer':
                x = batch['text_input'].to(self.device)
                y = batch['target'].to(self.device)
                
                # Forward pass
                outputs = self.model(x)
                loss = self.criterion(outputs, y)
                
            elif self.model_type in ['hybrid', 'finllm']:
                ts_input = batch['ts_input'].to(self.device)
                text_input = batch['text_input'].to(self.device)
                y = batch['target'].to(self.device)
                
                if self.model_type == 'hybrid':
                    # For hybrid model
                    outputs = self.model(ts_input, text_input)
                    loss = self.criterion(outputs, y)
                else:
                    # For FinLLM
                    # Note: text_input needs to be [seq_len, batch, dim] for transformer
                    text_input = text_input.permute(1, 0, 2)
                    predictions = self.model(ts_input, text_input)
                    loss = self.model.compute_loss(predictions, y)
            
            # Backward pass and optimization
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            total_loss += loss.item()
        
        # Calculate average loss
        avg_loss = total_loss / len(train_loader)
        return avg_loss
    
    def evaluate(self, val_loader):
        """
        Evaluate model on validation data
        
        Args:
            val_loader: DataLoader for validation data
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                # Get data and move to device
                if self.model_type == 'bilstm':
                    x = batch['ts_input'].to(self.device)
                    y = batch['target'].to(self.device)
                    
                    # Forward pass
                    outputs = self.model(x)
                    loss = self.criterion(outputs, y)
                    predictions = outputs
                    
                elif self.model_type == 'transformer':
                    x = batch['text_input'].to(self.device)
                    y = batch['target'].to(self.device)
                    
                    # Forward pass
                    outputs = self.model(x)
                    loss = self.criterion(outputs, y)
                    predictions = outputs
                    
                elif self.model_type in ['hybrid', 'finllm']:
                    ts_input = batch['ts_input'].to(self.device)
                    text_input = batch['text_input'].to(self.device)
                    y = batch['target'].to(self.device)
                    
                    if self.model_type == 'hybrid':
                        # For hybrid model
                        outputs = self.model(ts_input, text_input)
                        loss = self.criterion(outputs, y)
                        predictions = outputs
                    else:
                        # For FinLLM
                        text_input = text_input.permute(1, 0, 2)
                        outputs = self.model(ts_input, text_input)
                        loss = self.model.compute_loss(outputs, y)
                        predictions = outputs['mean']
                
                total_loss += loss.item()
                
                # Store predictions and targets
                all_predictions.extend(predictions.cpu().numpy().flatten())
                all_targets.extend(y.cpu().numpy().flatten())
        
        # Calculate average loss
        avg_loss = total_loss / len(val_loader)
        
        # Calculate metrics
        metrics = self.calculate_metrics(all_predictions, all_targets)
        metrics['loss'] = avg_loss
        
        return metrics
    
    def calculate_metrics(self, predictions, targets):
        """
        Calculate evaluation metrics
        
        Args:
            predictions: List of predictions
            targets: List of targets
            
        Returns:
            Dictionary of metrics
        """
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        # MSE
        mse = np.mean((predictions - targets) ** 2)
        
        # MAE
        mae = np.mean(np.abs(predictions - targets))
        
        # R^2
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        ss_res = np.sum((targets - predictions) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        # Information Coefficient (Spearman rank correlation)
        from scipy.stats import spearmanr
        ic, ic_pvalue = spearmanr(predictions, targets)
        
        # Directional accuracy
        pred_direction = np.sign(predictions)
        target_direction = np.sign(targets)
        directional_acc = np.mean(pred_direction == target_direction)
        
        return {
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'ic': ic,
            'ic_pvalue': ic_pvalue,
            'directional_acc': directional_acc
        }
    
    def train(self, train_loader, val_loader, model_name=None):
        """
        Train model
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            model_name: Name for saving model
            
        Returns:
            Dictionary with training history
        """
        # Initialize history
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_metrics': []
        }
        
        # Initialize early stopping variables
        best_val_loss = float('inf')
        best_epoch = 0
        epochs_without_improvement = 0
        
        # Generate model name if not provided
        if model_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"{self.model_type}_{timestamp}"
        
        print(f"Starting training for {model_name}...")
        print(f"Using device: {self.device}")
        
        for epoch in range(self.max_epochs):
            start_time = time.time()
            
            # Train
            train_loss = self.train_epoch(train_loader)
            history['train_loss'].append(train_loss)
            
            # Evaluate
            val_metrics = self.evaluate(val_loader)
            val_loss = val_metrics['loss']
            history['val_loss'].append(val_loss)
            history['val_metrics'].append(val_metrics)
            
            # Print progress
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch+1}/{self.max_epochs} - "
                 f"Time: {epoch_time:.2f}s - "
                 f"Train Loss: {train_loss:.6f} - "
                 f"Val Loss: {val_loss:.6f} - "
                 f"Val IC: {val_metrics['ic']:.4f} - "
                 f"Val Dir Acc: {val_metrics['directional_acc']:.4f}")
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                epochs_without_improvement = 0
                
                # Save best model
                self.save_checkpoint(model_name, epoch, val_metrics)
                
                print(f"Saved new best model with val_loss: {val_loss:.6f}")
            else:
                epochs_without_improvement += 1
                
                # Early stopping
                if epochs_without_improvement >= self.patience:
                    print(f"Early stopping at epoch {epoch+1}, "
                         f"best epoch was {best_epoch+1} with val_loss: {best_val_loss:.6f}")
                    break
        
        # Save training history
        self.save_history(history, model_name)
        
        # Load best model
        self.load_checkpoint(model_name)
        
        print(f"Training completed. Best epoch was {best_epoch+1} with val_loss: {best_val_loss:.6f}")
        
        return history
    
    def save_checkpoint(self, model_name, epoch, metrics):
        """
        Save model checkpoint
        
        Args:
            model_name: Model name
            epoch: Current epoch
            metrics: Validation metrics
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{model_name}_best.pt")
        
        # Save model state
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics
        }, checkpoint_path)
    
    def load_checkpoint(self, model_name):
        """
        Load model checkpoint
        
        Args:
            model_name: Model name
            
        Returns:
            Loaded checkpoint
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{model_name}_best.pt")
        
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            print(f"Loaded checkpoint from epoch {checkpoint['epoch']+1}")
            return checkpoint
        else:
            print(f"No checkpoint found at {checkpoint_path}")
            return None
    
    def save_history(self, history, model_name):
        """
        Save training history
        
        Args:
            history: Training history
            model_name: Model name
        """
        # Convert to serializable format
        serializable_history = {
            'train_loss': [float(x) for x in history['train_loss']],
            'val_loss': [float(x) for x in history['val_loss']],
            'val_metrics': []
        }
        
        for metrics in history['val_metrics']:
            serializable_metrics = {}
            for k, v in metrics.items():
                serializable_metrics[k] = float(v) if isinstance(v, (np.floating, float)) else v
            serializable_history['val_metrics'].append(serializable_metrics)
        
        # Save to file
        history_path = os.path.join(self.log_dir, f"{model_name}_history.json")
        with open(history_path, 'w') as f:
            json.dump(serializable_history, f, indent=2)