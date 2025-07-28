import argparse
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime


def load_evaluation_results(results_file):
    """
    Load evaluation results from JSON file
    """
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    return results


def plot_model_comparison(results_list, output_dir='.'):
    """
    Generate plots comparing model performance
    
    Args:
        results_list: List of model evaluation results
        output_dir: Output directory for plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create dataframe for comparison
    comparison_data = []
    
    for result in results_list:
        model_type = result["model_type"]
        accuracy = result["accuracy_metrics"]
        
        model_data = {
            "Model": model_type,
            "IC": accuracy["ic"],
            "RMSE": accuracy["rmse"],
            "Directional Accuracy": accuracy["directional_accuracy"],
            "R²": accuracy["r2"],
            "IC Lower CI": accuracy["ic_95ci_lower"],
            "IC Upper CI": accuracy["ic_95ci_upper"]
        }
        
        # Add backtest metrics if available
        if "cross_sectional_results" in result and result["cross_sectional_results"]:
            backtest = result["cross_sectional_results"]["backtest_results"]["performance_metrics"]
            ic_metrics = result["cross_sectional_results"]["ic_metrics"]
            
            model_data.update({
                "IC IR": ic_metrics["ic_ir"],
                "Annualized Return (%)": backtest["annualized_return"],
                "Sharpe Ratio": backtest["sharpe_ratio"],
                "Max Drawdown (%)": backtest["max_drawdown"] * 100,
                "Calmar Ratio": backtest["calmar_ratio"],
                "Win Rate (%)": backtest["win_rate"] * 100
            })
        
        # Add inference metrics
        inference = result["inference_metrics"]
        model_data["Inference Time (ms)"] = inference["inference_time_per_sample"] * 1000
        
        comparison_data.append(model_data)
    
    # Create comparison dataframe
    comparison_df = pd.DataFrame(comparison_data)
    
    # Save comparison table
    comparison_table_path = os.path.join(output_dir, "model_comparison_table.csv")
    comparison_df.to_csv(comparison_table_path, index=False)
    print(f"Comparison table saved to {comparison_table_path}")
    
    # Plot IC comparison with error bars
    plt.figure(figsize=(12, 6))
    plt.errorbar(
        comparison_df["Model"],
        comparison_df["IC"],
        yerr=[
            comparison_df["IC"] - comparison_df["IC Lower CI"],
            comparison_df["IC Upper CI"] - comparison_df["IC"]
        ],
        fmt='o',
        capsize=5,
        markersize=8
    )
    plt.title("Information Coefficient with 95% Confidence Intervals", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.ylabel("Information Coefficient (IC)", fontsize=12)
    plt.xticks(fontsize=11)
    plt.tight_layout()
    
    ic_plot_path = os.path.join(output_dir, "ic_comparison.png")
    plt.savefig(ic_plot_path)
    print(f"IC comparison plot saved to {ic_plot_path}")
    
    # Plot performance metrics if available
    if "Annualized Return (%)" in comparison_df.columns:
        # Performance metrics to plot
        perf_metrics = [
            "Annualized Return (%)",
            "Sharpe Ratio",
            "Max Drawdown (%)",
            "Calmar Ratio"
        ]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(perf_metrics):
            ax = axes[i]
            # Use different colors for positive/negative values for some metrics
            if metric in ["Annualized Return (%)", "Sharpe Ratio", "Calmar Ratio"]:
                colors = ['green' if x > 0 else 'red' for x in comparison_df[metric]]
            elif metric == "Max Drawdown (%)":
                colors = ['red' if x > 10 else 'orange' if x > 5 else 'green' for x in comparison_df[metric]]
            else:
                colors = 'blue'
            
            ax.bar(comparison_df["Model"], comparison_df[metric], color=colors)
            ax.set_title(metric, fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        perf_plot_path = os.path.join(output_dir, "performance_comparison.png")
        plt.savefig(perf_plot_path)
        print(f"Performance comparison plot saved to {perf_plot_path}")
    
    # Plot inference time comparison
    plt.figure(figsize=(12, 6))
    bars = plt.bar(comparison_df["Model"], comparison_df["Inference Time (ms)"])
    
    # Add value labels above bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2., height + 0.1,
            f'{height:.2f}',
            ha='center', va='bottom', 
            fontsize=10
        )
    
    plt.title("Inference Time per Sample", fontsize=14)
    plt.ylabel("Time (milliseconds)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    inf_plot_path = os.path.join(output_dir, "inference_time_comparison.png")
    plt.savefig(inf_plot_path)
    print(f"Inference time comparison plot saved to {inf_plot_path}")


def main():
    parser = argparse.ArgumentParser(description='Compare FinLLM model evaluation results')
    parser.add_argument('--results_files', nargs='+', required=True, 
                       help='Paths to evaluation result JSON files')
    parser.add_argument('--output_dir', default='./model_comparison', 
                       help='Output directory for comparison results')
    
    args = parser.parse_args()
    
    # Load all evaluation results
    all_results = []
    
    for results_file in args.results_files:
        results = load_evaluation_results(results_file)
        
        # Check if results is a list or single result
        if isinstance(results, list):
            all_results.extend(results)
        else:
            all_results.append(results)
    
    # Generate comparison plots
    plot_model_comparison(all_results, args.output_dir)
    
    print("Model comparison completed successfully!")


if __name__ == "__main__":
    main()