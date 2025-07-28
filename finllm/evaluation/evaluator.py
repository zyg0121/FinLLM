import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from tqdm import tqdm
import time
import os
import json
from datetime import datetime

# Import evaluation modules
from finllm.evaluation.metrics import ForecastingMetrics
from finllm.evaluation.backtest import PortfolioBacktester, RiskAnalyzer
from finllm.evaluation.explainability import ModelExplainer


class ModelEvaluator:
    """
    Main evaluator class for FinLLM models
    """
    
    def __init__(self, model, model_type, device=None, output_dir="./evaluation_results"):
        """
        Initialize evaluator
        
        Args:
            model: Model to evaluate
            model_type: Model type ('finllm', 'bilstm', 'transformer', 'hybrid')
            device: Torch device
            output_dir: Directory for saving results
        """
        self.model = model
        self.model_type = model_type
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        # Move model to device and set to eval mode if it's a PyTorch model
        if isinstance(model, torch.nn.Module):
            self.model.to(self.device)
            self.model.eval()
    
    def predict(self, test_loader):
        """
        Generate predictions from model
        
        Args:
            test_loader: DataLoader with test data
            
        Returns:
            Dictionary with predictions and ground truth
        """
        all_predictions = []
        all_targets = []
        all_dates = []
        all_tickers = []
        
        start_time = time.time()
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Generating predictions"):
                # Get data
                if 'date' in batch:
                    dates = batch['date']
                else:
                    dates = None
                    
                if 'ticker' in batch:
                    tickers = batch['ticker']
                else:
                    tickers = None
                
                # Generate predictions based on model type
                if self.model_type == 'bilstm':
                    ts_input = batch['ts_input'].to(self.device)
                    targets = batch['target']
                    
                    outputs = self.model(ts_input)
                    predictions = outputs.cpu().numpy()
                    
                elif self.model_type == 'transformer':
                    text_input = batch['text_input'].to(self.device)
                    targets = batch['target']
                    
                    outputs = self.model(text_input)
                    predictions = outputs.cpu().numpy()
                    
                elif self.model_type in ['hybrid', 'finllm']:
                    ts_input = batch['ts_input'].to(self.device)
                    text_input = batch['text_input'].to(self.device)
                    targets = batch['target']
                    
                    if self.model_type == 'hybrid':
                        outputs = self.model(ts_input, text_input)
                        predictions = outputs.cpu().numpy()
                    else:
                        # For FinLLM, text_input needs to be [seq_len, batch, dim]
                        text_input = text_input.permute(1, 0, 2)
                        outputs = self.model(ts_input, text_input)
                        predictions = outputs['mean'].cpu().numpy()
                
                # Store predictions and targets
                all_predictions.append(predictions)
                all_targets.append(targets.numpy())
                
                if dates is not None:
                    all_dates.extend(dates)
                    
                if tickers is not None:
                    all_tickers.extend(tickers)
        
        end_time = time.time()
        inference_time = end_time - start_time
        
        # Combine results
        predictions = np.vstack(all_predictions).flatten()
        targets = np.vstack(all_targets).flatten()
        
        return {
            "predictions": predictions,
            "targets": targets,
            "dates": all_dates if all_dates else None,
            "tickers": all_tickers if all_tickers else None,
            "inference_time": inference_time,
            "num_samples": len(predictions)
        }
    
    def evaluate_forecasting_accuracy(self, predictions, targets):
        """
        Evaluate forecasting accuracy metrics
        
        Args:
            predictions: Model predictions
            targets: Ground truth
            
        Returns:
            Dictionary of metrics
        """
        # Calculate regression metrics
        regression_metrics = ForecastingMetrics.regression_metrics(predictions, targets)
        
        # Calculate Information Coefficient
        ic = ForecastingMetrics.information_coefficient(predictions, targets)
        
        # Calculate directional accuracy
        dir_acc = ForecastingMetrics.directional_accuracy(predictions, targets)
        
        # Calculate bootstrap confidence intervals for IC
        ic_bootstrap = ForecastingMetrics.bootstrap_confidence_intervals(
            predictions, targets, ForecastingMetrics.information_coefficient
        )
        
        metrics = {
            "mse": regression_metrics["mse"],
            "rmse": regression_metrics["rmse"],
            "mae": regression_metrics["mae"],
            "r2": regression_metrics["r2"],
            "ic": ic,
            "ic_95ci_lower": ic_bootstrap["lower"],
            "ic_95ci_upper": ic_bootstrap["upper"],
            "directional_accuracy": dir_acc
        }
        
        return metrics
    
    def evaluate_cross_sectional(self, predictions_dict):
        """
        Evaluate cross-sectional performance
        
        Args:
            predictions_dict: Dictionary with predictions, targets, dates, tickers
            
        Returns:
            Dictionary with evaluation results
        """
        predictions = predictions_dict["predictions"]
        targets = predictions_dict["targets"]
        dates = predictions_dict["dates"]
        tickers = predictions_dict["tickers"]
        
        if dates is None or tickers is None:
            print("Cross-sectional evaluation requires dates and tickers")
            return None
        
        # Create DataFrame with predictions and targets
        results_df = pd.DataFrame({
            "date": dates,
            "ticker": tickers,
            "prediction": predictions,
            "target": targets
        })
        
        # Calculate daily IC
        daily_ics = []
        unique_dates = results_df["date"].unique()
        
        for date in unique_dates:
            day_data = results_df[results_df["date"] == date]
            day_preds = day_data["prediction"].values
            day_targets = day_data["target"].values
            
            ic = ForecastingMetrics.information_coefficient(day_preds, day_targets)
            daily_ics.append({"date": date, "ic": ic})
        
        daily_ic_df = pd.DataFrame(daily_ics)
        
        # Calculate IC metrics
        ic_metrics = ForecastingMetrics.calc_ic_metrics(daily_ic_df.set_index("date")["ic"])
        
        # Create predictions and returns DataFrames for backtesting
        pivot_preds = results_df.pivot(index="date", columns="ticker", values="prediction")
        pivot_returns = results_df.pivot(index="date", columns="ticker", values="target")
        
        # Initialize backtester
        backtester = PortfolioBacktester(
            predictions_df=pivot_preds,
            returns_df=pivot_returns
        )
        
        # Run backtest
        backtest_results = backtester.backtest_long_short_portfolio(
            n_long=10,
            n_short=10,
            transaction_cost=0.001,  # 10 bps
            market_neutral=True
        )
        
        # Calculate transaction cost sensitivity
        tc_sensitivity = backtester.run_transaction_cost_sensitivity()
        
        # Find break-even transaction cost
        break_even_row = tc_sensitivity[tc_sensitivity["annualized_return"] <= 0].iloc[0] if len(tc_sensitivity[tc_sensitivity["annualized_return"] <= 0]) > 0 else None
        break_even_tc = break_even_row["transaction_cost_bps"] if break_even_row is not None else np.nan
        
        return {
            "daily_ic": daily_ic_df,
            "ic_metrics": ic_metrics,
            "backtest_results": backtest_results,
            "tc_sensitivity": tc_sensitivity,
            "break_even_tc": break_even_tc
        }
    
    def evaluate_model(self, test_loader):
        """
        Comprehensive model evaluation
        
        Args:
            test_loader: DataLoader with test data
            
        Returns:
            Dictionary with evaluation results
        """
        # Generate predictions
        predictions_dict = self.predict(test_loader)
        
        # Evaluate forecasting accuracy
        accuracy_metrics = self.evaluate_forecasting_accuracy(
            predictions_dict["predictions"],
            predictions_dict["targets"]
        )
        
        # Evaluate cross-sectional performance if dates and tickers are available
        if predictions_dict["dates"] is not None and predictions_dict["tickers"] is not None:
            cross_sectional_results = self.evaluate_cross_sectional(predictions_dict)
        else:
            cross_sectional_results = None
        
        # Calculate inference performance
        inference_time = predictions_dict["inference_time"]
        num_samples = predictions_dict["num_samples"]
        inference_time_per_sample = inference_time / num_samples
        
        inference_metrics = {
            "total_inference_time": inference_time,
            "num_samples": num_samples,
            "inference_time_per_sample": inference_time_per_sample,
            "samples_per_second": num_samples / inference_time if inference_time > 0 else 0
        }
        
        # Combine all results
        evaluation_results = {
            "model_type": self.model_type,
            "accuracy_metrics": accuracy_metrics,
            "inference_metrics": inference_metrics,
            "cross_sectional_results": cross_sectional_results,
            "evaluation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Save results
        self.save_evaluation_results(evaluation_results)
        
        return evaluation_results
    
    def save_evaluation_results(self, results, filename=None):
        """
        Save evaluation results to file
        
        Args:
            results: Evaluation results
            filename: Output filename (default: model_type_evaluation.json)
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.model_type}_evaluation_{timestamp}.json"
        
        file_path = os.path.join(self.output_dir, filename)
        
        # Prepare serializable results
        serializable_results = self._make_serializable(results)
        
        # Save to file
        with open(file_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Evaluation results saved to {file_path}")
    
    def _make_serializable(self, obj):
        """
        Convert objects to JSON serializable formats
        
        Args:
            obj: Object to convert
            
        Returns:
            JSON serializable version of object
        """
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(i) for i in obj]
        elif isinstance(obj, (pd.DataFrame, pd.Series)):
            return obj.to_dict()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, (datetime, np.datetime64)):
            return str(obj)
        else:
            return obj
    
    def plot_evaluation_results(self, results):
        """
        Plot key evaluation results
        
        Args:
            results: Evaluation results from evaluate_model
        """
        # Create figure with multiple subplots
        plt.figure(figsize=(20, 15))
        
        # Plot accuracy metrics
        plt.subplot(2, 2, 1)
        accuracy = results["accuracy_metrics"]
        metrics = ["ic", "r2", "directional_accuracy"]
        values = [accuracy[m] for m in metrics]
        plt.bar(metrics, values)
        plt.title("Accuracy Metrics")
        plt.ylim(0, 1)
        
        # Plot IC over time if available
        if results["cross_sectional_results"] is not None:
            plt.subplot(2, 2, 2)
            daily_ic = pd.DataFrame(results["cross_sectional_results"]["daily_ic"])
            daily_ic["date"] = pd.to_datetime(daily_ic["date"])
            daily_ic.set_index("date", inplace=True)
            daily_ic.sort_index(inplace=True)
            
            plt.plot(daily_ic.index, daily_ic["ic"])
            plt.title("Information Coefficient Over Time")
            plt.grid(True, alpha=0.3)
            plt.axhline(y=0, color='r', linestyle='--')
            
            # Plot equity curve if available
            plt.subplot(2, 2, 3)
            portfolio_values = pd.Series(
                results["cross_sectional_results"]["backtest_results"]["portfolio_values"]
            )
            plt.plot(portfolio_values)
            plt.title("Portfolio Equity Curve")
            plt.grid(True, alpha=0.3)
            
            # Plot transaction cost sensitivity
            plt.subplot(2, 2, 4)
            tc_sensitivity = pd.DataFrame(results["cross_sectional_results"]["tc_sensitivity"])
            plt.plot(tc_sensitivity["transaction_cost_bps"], tc_sensitivity["annualized_return"])
            plt.axhline(y=0, color='r', linestyle='--')
            plt.title("Transaction Cost Sensitivity")
            plt.xlabel("Transaction Cost (bps)")
            plt.ylabel("Annualized Return")
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def compare_models(self, evaluation_results_list):
        """
        Compare evaluation results across multiple models
        
        Args:
            evaluation_results_list: List of evaluation results from different models
            
        Returns:
            DataFrame with comparison
        """
        comparison = []
        
        for results in evaluation_results_list:
            model_type = results["model_type"]
            accuracy = results["accuracy_metrics"]
            inference = results["inference_metrics"]
            
            # Extract key metrics
            model_metrics = {
                "model_type": model_type,
                "mean_ic": accuracy["ic"],
                "ic_95ci_lower": accuracy["ic_95ci_lower"],
                "ic_95ci_upper": accuracy["ic_95ci_upper"],
                "directional_accuracy": accuracy["directional_accuracy"],
                "rmse": accuracy["rmse"],
                "inference_time_ms": inference["inference_time_per_sample"] * 1000
            }
            
            # Add backtest metrics if available
            if results["cross_sectional_results"] is not None:
                backtest = results["cross_sectional_results"]["backtest_results"]["performance_metrics"]
                ic_metrics = results["cross_sectional_results"]["ic_metrics"]
                
                model_metrics.update({
                    "ic_ir": ic_metrics["ic_ir"],
                    "annualized_return": backtest["annualized_return"],
                    "sharpe_ratio": backtest["sharpe_ratio"],
                    "max_drawdown": backtest["max_drawdown"],
                    "calmar_ratio": backtest["calmar_ratio"],
                    "win_rate": backtest["win_rate"]
                })
            
            comparison.append(model_metrics)
        
        return pd.DataFrame(comparison)
    
    def plot_model_comparison(self, comparison_df):
        """
        Plot model comparison
        
        Args:
            comparison_df: DataFrame from compare_models
        """
        # Set model_type as index for plotting
        comparison_df = comparison_df.set_index("model_type")
        
        # Create figure with multiple subplots
        plt.figure(figsize=(20, 15))
        
        # Plot Information Coefficient with error bars
        plt.subplot(2, 2, 1)
        plt.errorbar(
            comparison_df.index,
            comparison_df["mean_ic"],
            yerr=[
                comparison_df["mean_ic"] - comparison_df["ic_95ci_lower"],
                comparison_df["ic_95ci_upper"] - comparison_df["mean_ic"]
            ],
            fmt='o',
            capsize=5
        )
        plt.title("Information Coefficient (95% CI)")
        plt.grid(True, alpha=0.3)
        
        # Plot Directional Accuracy
        plt.subplot(2, 2, 2)
        comparison_df["directional_accuracy"].plot(kind='bar')
        plt.title("Directional Accuracy")
        plt.ylim(0.5, 0.7)  # Typical range for this metric
        plt.grid(True, alpha=0.3)
        
        # Plot Annualized Return if available
        if "annualized_return" in comparison_df.columns:
            plt.subplot(2, 2, 3)
            comparison_df["annualized_return"].plot(kind='bar')
            plt.title("Annualized Return (%)")
            plt.grid(True, alpha=0.3)
            
            # Plot Sharpe Ratio if available
            plt.subplot(2, 2, 4)
            comparison_df["sharpe_ratio"].plot(kind='bar')
            plt.title("Sharpe Ratio")
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()