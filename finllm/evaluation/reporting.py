import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
import json
import os
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px


class FinLLMReportGenerator:
    """
    Generate comprehensive reports for FinLLM model evaluation
    """
    
    def __init__(self, results, output_dir="./reports"):
        """
        Initialize report generator
        
        Args:
            results: Evaluation results or list of results for comparison
            output_dir: Directory for saving reports
        """
        self.results = results
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Determine if results is a single model or multiple models
        if isinstance(results, list):
            self.multi_model = True
        else:
            self.multi_model = False
            self.results = [self.results]  # Convert to list for uniform processing
    
    def generate_summary_table(self):
        """
        Generate summary table with key metrics
        
        Returns:
            DataFrame with summary metrics
        """
        summary_data = []
        
        for result in self.results:
            model_type = result["model_type"]
            accuracy = result["accuracy_metrics"]
            
            # Basic metrics available for all models
            model_summary = {
                "Model": model_type,
                "IC": f"{accuracy['ic']:.4f} [{accuracy['ic_95ci_lower']:.4f}, {accuracy['ic_95ci_upper']:.4f}]",
                "RMSE": f"{accuracy['rmse']:.6f}",
                "Dir. Accuracy": f"{accuracy['directional_accuracy']*100:.2f}%",
                "R²": f"{accuracy['r2']:.4f}",
            }
            
            # Add backtest metrics if available
            if result["cross_sectional_results"] is not None:
                backtest = result["cross_sectional_results"]["backtest_results"]["performance_metrics"]
                ic_metrics = result["cross_sectional_results"]["ic_metrics"]
                
                model_summary.update({
                    "IC IR": f"{ic_metrics['ic_ir']:.4f}",
                    "Ann. Return": f"{backtest['annualized_return']:.2f}%",
                    "Sharpe": f"{backtest['sharpe_ratio']:.2f}",
                    "Max DD": f"{backtest['max_drawdown']*100:.2f}%",
                    "Calmar": f"{backtest['calmar_ratio']:.2f}",
                })
            
            # Add inference metrics
            inference = result["inference_metrics"]
            model_summary["Inf. Time"] = f"{inference['inference_time_per_sample']*1000:.2f} ms"
            
            summary_data.append(model_summary)
        
        return pd.DataFrame(summary_data)
    
    def generate_full_report(self, filename=None):
        """
        Generate comprehensive HTML report
        
        Args:
            filename: Output filename (default: finllm_report_TIMESTAMP.html)
            
        Returns:
            Path to generated report
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"finllm_report_{timestamp}.html"
        
        file_path = os.path.join(self.output_dir, filename)
        
        # Generate report content
        html_content = []
        
        # Add header
        html_content.append("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>FinLLM Evaluation Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1, h2, h3 { color: #2c3e50; }
                table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
                th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
                th { background-color: #f2f2f2; }
                .section { margin-top: 30px; margin-bottom: 30px; }
                .plot { margin-top: 20px; margin-bottom: 20px; text-align: center; }
                .warning { color: orange; }
                .good { color: green; }
                .bad { color: red; }
            </style>
        </head>
        <body>
            <h1>FinLLM Evaluation Report</h1>
            <p>Generated on: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
        """)
        
        # Add summary section
        html_content.append("""
        <div class="section">
            <h2>Summary</h2>
        """)
        
        summary_df = self.generate_summary_table()
        html_content.append(summary_df.to_html(index=False, classes="dataframe"))
        
        html_content.append("</div>")
        
        # Add detailed sections for each model
        for result in self.results:
            model_type = result["model_type"]
            
            html_content.append(f"""
            <div class="section">
                <h2>Model: {model_type}</h2>
                
                <h3>Accuracy Metrics</h3>
            """)
            
            # Add accuracy metrics
            accuracy = result["accuracy_metrics"]
            accuracy_df = pd.DataFrame({
                "Metric": list(accuracy.keys()),
                "Value": list(accuracy.values())
            })
            html_content.append(accuracy_df.to_html(index=False, classes="dataframe"))
            
            # Add cross-sectional results if available
            if result["cross_sectional_results"] is not None:
                cross_sectional = result["cross_sectional_results"]
                backtest = cross_sectional["backtest_results"]["performance_metrics"]
                
                html_content.append("""
                <h3>Trading Performance</h3>
                """)
                
                # Create trading performance table
                trading_metrics = {
                    "Annualized Return (%)": backtest["annualized_return"],
                    "Sharpe Ratio": backtest["sharpe_ratio"],
                    "Information Ratio": backtest["information_ratio"],
                    "Sortino Ratio": backtest["sortino_ratio"],
                    "Maximum Drawdown (%)": backtest["max_drawdown"] * 100,
                    "Calmar Ratio": backtest["calmar_ratio"],
                    "Win Rate (%)": backtest["win_rate"],
                    "Expected Shortfall 95% (%)": backtest["expected_shortfall_95"] * 100
                }
                
                trading_df = pd.DataFrame({
                    "Metric": list(trading_metrics.keys()),
                    "Value": list(trading_metrics.values())
                })
                html_content.append(trading_df.to_html(index=False, classes="dataframe"))
                
                # Add transaction cost sensitivity
                html_content.append("""
                <h3>Transaction Cost Sensitivity</h3>
                <p>Break-even transaction cost: """ + f"{cross_sectional['break_even_tc']:.2f} bps</p>")
                
                # Generate and save TC sensitivity plot
                tc_fig = plt.figure(figsize=(10, 6))
                tc_sensitivity = pd.DataFrame(cross_sectional["tc_sensitivity"])
                plt.plot(tc_sensitivity["transaction_cost_bps"], 
                         tc_sensitivity["annualized_return"])
                plt.axhline(y=0, color='r', linestyle='--')
                plt.title("Transaction Cost Sensitivity")
                plt.xlabel("Transaction Cost (bps)")
                plt.ylabel("Annualized Return (%)")
                plt.grid(True, alpha=0.3)
                tc_plot_path = os.path.join(self.output_dir, f"{model_type}_tc_sensitivity.png")
                plt.savefig(tc_plot_path)
                plt.close()
                
                html_content.append(f"""
                <div class="plot">
                    <img src="{os.path.basename(tc_plot_path)}" alt="TC Sensitivity">
                </div>
                """)
                
                # Add daily IC statistics
                ic_metrics = cross_sectional["ic_metrics"]
                html_content.append("""
                <h3>Information Coefficient (IC) Statistics</h3>
                """)
                
                ic_stats = {
                    "Mean IC": ic_metrics["mean_ic"],
                    "Median IC": ic_metrics["median_ic"],
                    "IC Standard Deviation": ic_metrics["std_ic"],
                    "IC Information Ratio": ic_metrics["ic_ir"],
                    "t-statistic": ic_metrics["t_stat"],
                    "p-value": ic_metrics["p_value"]
                }
                
                ic_df = pd.DataFrame({
                    "Statistic": list(ic_stats.keys()),
                    "Value": list(ic_stats.values())
                })
                html_content.append(ic_df.to_html(index=False, classes="dataframe"))
                
                # Generate and save IC time series plot
                ic_fig = plt.figure(figsize=(10, 6))
                daily_ic = pd.DataFrame(cross_sectional["daily_ic"])
                daily_ic["date"] = pd.to_datetime(daily_ic["date"])
                daily_ic.set_index("date", inplace=True)
                daily_ic.sort_index(inplace=True)
                
                plt.plot(daily_ic.index, daily_ic["ic"])
                plt.title("Information Coefficient Over Time")
                plt.grid(True, alpha=0.3)
                plt.axhline(y=0, color='r', linestyle='--')
                ic_plot_path = os.path.join(self.output_dir, f"{model_type}_daily_ic.png")
                plt.savefig(ic_plot_path)
                plt.close()
                
                html_content.append(f"""
                <div class="plot">
                    <img src="{os.path.basename(ic_plot_path)}" alt="Daily IC">
                </div>
                """)
                
                # Generate and save equity curve plot
                eq_fig = plt.figure(figsize=(10, 6))
                portfolio_values = pd.Series(
                    cross_sectional["backtest_results"]["portfolio_values"]
                )
                plt.plot(portfolio_values)
                plt.title("Portfolio Equity Curve")
                plt.grid(True, alpha=0.3)
                eq_plot_path = os.path.join(self.output_dir, f"{model_type}_equity_curve.png")
                plt.savefig(eq_plot_path)
                plt.close()
                
                html_content.append(f"""
                <div class="plot">
                    <img src="{os.path.basename(eq_plot_path)}" alt="Equity Curve">
                </div>
                """)
                
                # Generate and save monthly returns heatmap
                returns = pd.Series(
                    cross_sectional["backtest_results"]["portfolio_returns"],
                    index=cross_sectional["daily_ic"]["date"]
                )
                returns.index = pd.to_datetime(returns.index)
                
                # Resample to monthly returns
                monthly_returns = (returns + 1).resample('M').prod() - 1
                
                # Create a pivot table for the heatmap
                monthly_pivot = pd.pivot_table(
                    monthly_returns.reset_index(),
                    values=0,
                    index=monthly_returns.index.strftime('%Y'),
                    columns=monthly_returns.index.strftime('%b'),
                    aggfunc='first'
                )
                
                monthly_fig = plt.figure(figsize=(12, 8))
                sns.heatmap(monthly_pivot * 100, 
                           annot=True, 
                           fmt=".2f", 
                           cmap="RdYlGn",
                           cbar_kws={'label': '%'},
                           center=0)
                
                plt.title("Monthly Returns (%)")
                plt.ylabel("Year")
                plt.xlabel("Month")
                monthly_plot_path = os.path.join(self.output_dir, f"{model_type}_monthly_returns.png")
                plt.savefig(monthly_plot_path)
                plt.close()
                
                html_content.append(f"""
                <div class="plot">
                    <img src="{os.path.basename(monthly_plot_path)}" alt="Monthly Returns">
                </div>
                """)
            
            # Add inference metrics
            html_content.append("""
            <h3>Inference Performance</h3>
            """)
            
            inference = result["inference_metrics"]
            inference_df = pd.DataFrame({
                "Metric": [
                    "Total Inference Time (s)",
                    "Number of Samples",
                    "Inference Time per Sample (ms)",
                    "Samples per Second"
                ],
                "Value": [
                    f"{inference['total_inference_time']:.4f}",
                    inference['num_samples'],
                    f"{inference['inference_time_per_sample']*1000:.4f}",
                    f"{inference['samples_per_second']:.2f}"
                ]
            })
            html_content.append(inference_df.to_html(index=False, classes="dataframe"))
            
            html_content.append("</div>")  # End of model section
        
        # If multiple models, add comparison plots
        if self.multi_model and len(self.results) > 1:
            html_content.append("""
            <div class="section">
                <h2>Model Comparison</h2>
            """)
            
            # Generate IC comparison plot
            ic_fig = plt.figure(figsize=(10, 6))
            ic_data = []
            for result in self.results:
                model_type = result["model_type"]
                accuracy = result["accuracy_metrics"]
                ic_data.append({
                    "Model": model_type,
                    "IC": accuracy["ic"],
                    "Lower CI": accuracy["ic_95ci_lower"],
                    "Upper CI": accuracy["ic_95ci_upper"]
                })
            
            ic_df = pd.DataFrame(ic_data)
            plt.figure(figsize=(10, 6))
            plt.errorbar(
                ic_df["Model"],
                ic_df["IC"],
                yerr=[
                    ic_df["IC"] - ic_df["Lower CI"],
                    ic_df["Upper CI"] - ic_df["IC"]
                ],
                fmt='o',
                capsize=5
            )
            plt.title("Information Coefficient Comparison (with 95% CI)")
            plt.grid(True, alpha=0.3)
            ic_comp_path = os.path.join(self.output_dir, "ic_comparison.png")
            plt.savefig(ic_comp_path)
            plt.close()
            
            html_content.append(f"""
            <div class="plot">
                <img src="{os.path.basename(ic_comp_path)}" alt="IC Comparison">
            </div>
            """)
            
            # Add performance metrics comparison if available
            has_backtest = all(r["cross_sectional_results"] is not None for r in self.results)
            if has_backtest:
                # Generate return comparison plot
                ret_data = []
                for result in self.results:
                    model_type = result["model_type"]
                    backtest = result["cross_sectional_results"]["backtest_results"]["performance_metrics"]
                    ret_data.append({
                        "Model": model_type,
                        "Ann. Return": backtest["annualized_return"],
                        "Sharpe": backtest["sharpe_ratio"],
                        "Calmar": backtest["calmar_ratio"]
                    })
                
                ret_df = pd.DataFrame(ret_data)
                
                # Annualized return comparison
                plt.figure(figsize=(10, 6))
                plt.bar(ret_df["Model"], ret_df["Ann. Return"])
                plt.title("Annualized Return Comparison (%)")
                plt.grid(True, alpha=0.3)
                ret_comp_path = os.path.join(self.output_dir, "return_comparison.png")
                plt.savefig(ret_comp_path)
                plt.close()
                
                html_content.append(f"""
                <div class="plot">
                    <img src="{os.path.basename(ret_comp_path)}" alt="Return Comparison">
                </div>
                """)
                
                # Sharpe ratio comparison
                plt.figure(figsize=(10, 6))
                plt.bar(ret_df["Model"], ret_df["Sharpe"])
                plt.title("Sharpe Ratio Comparison")
                plt.grid(True, alpha=0.3)
                sharpe_comp_path = os.path.join(self.output_dir, "sharpe_comparison.png")
                plt.savefig(sharpe_comp_path)
                plt.close()
                
                html_content.append(f"""
                <div class="plot">
                    <img src="{os.path.basename(sharpe_comp_path)}" alt="Sharpe Comparison">
                </div>
                """)
            
            html_content.append("</div>")  # End of comparison section
        
        # Close HTML document
        html_content.append("""
        </body>
        </html>
        """)
        
        # Write HTML content to file
        with open(file_path, 'w') as f:
            f.write('\n'.join(html_content))
        
        print(f"Full report generated at: {file_path}")
        return file_path
    
    def generate_presentation_slides(self, filename=None):
        """
        Generate presentation-ready PDF slides
        
        Args:
            filename: Output filename (default: finllm_slides_TIMESTAMP.pdf)
            
        Returns:
            Path to generated slides
        """
        try:
            from fpdf import FPDF
            
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"finllm_slides_{timestamp}.pdf"
            
            file_path = os.path.join(self.output_dir, filename)
            
            # Create PDF object
            pdf = FPDF()
            pdf.set_author("FinLLM Evaluation System")
            pdf.set_title("FinLLM Model Evaluation")
            
            # Add title slide
            pdf.add_page()
            pdf.set_font("Arial", "B", 24)
            pdf.cell(0, 20, "FinLLM Model Evaluation", ln=True, align='C')
            pdf.set_font("Arial", "", 16)
            pdf.cell(0, 10, f"Generated on {datetime.now().strftime('%Y-%m-%d')}", ln=True, align='C')
            
            # Add summary slide
            pdf.add_page()
            pdf.set_font("Arial", "B", 20)
            pdf.cell(0, 15, "Summary of Results", ln=True)
            
            # Convert summary table to image and add to PDF
            summary_df = self.generate_summary_table()
            
            # Generate plots for each model
            for result in self.results:
                model_type = result["model_type"]
                
                # Add model overview slide
                pdf.add_page()
                pdf.set_font("Arial", "B", 20)
                pdf.cell(0, 15, f"Model: {model_type}", ln=True)
                
                # Add accuracy metrics
                pdf.set_font("Arial", "B", 16)
                pdf.cell(0, 10, "Accuracy Metrics", ln=True)
                
                accuracy = result["accuracy_metrics"]
                pdf.set_font("Arial", "", 12)
                pdf.cell(0, 8, f"Information Coefficient (IC): {accuracy['ic']:.4f}", ln=True)
                pdf.cell(0, 8, f"95% CI: [{accuracy['ic_95ci_lower']:.4f}, {accuracy['ic_95ci_upper']:.4f}]", ln=True)
                pdf.cell(0, 8, f"Directional Accuracy: {accuracy['directional_accuracy']*100:.2f}%", ln=True)
                pdf.cell(0, 8, f"RMSE: {accuracy['rmse']:.6f}", ln=True)
                
                # Add backtest results if available
                if result["cross_sectional_results"] is not None:
                    cross_sectional = result["cross_sectional_results"]
                    backtest = cross_sectional["backtest_results"]["performance_metrics"]
                    
                    # Add trading performance slide
                    pdf.add_page()
                    pdf.set_font("Arial", "B", 16)
                    pdf.cell(0, 10, "Trading Performance", ln=True)
                    
                    pdf.set_font("Arial", "", 12)
                    pdf.cell(0, 8, f"Annualized Return: {backtest['annualized_return']:.2f}%", ln=True)
                    pdf.cell(0, 8, f"Sharpe Ratio: {backtest['sharpe_ratio']:.2f}", ln=True)
                    pdf.cell(0, 8, f"Information Ratio: {backtest['information_ratio']:.2f}", ln=True)
                    pdf.cell(0, 8, f"Maximum Drawdown: {backtest['max_drawdown']*100:.2f}%", ln=True)
                    pdf.cell(0, 8, f"Calmar Ratio: {backtest['calmar_ratio']:.2f}", ln=True)
                    pdf.cell(0, 8, f"Win Rate: {backtest['win_rate']*100:.2f}%", ln=True)
            
            # Save PDF
            pdf.output(file_path)
            print(f"Presentation slides generated at: {file_path}")
            return file_path
            
        except ImportError:
            print("FPDF library not found. Please install it with 'pip install fpdf'")
            return None
    
    def export_to_json(self, filename=None):
        """
        Export evaluation results to JSON file
        
        Args:
            filename: Output filename (default: finllm_results_TIMESTAMP.json)
            
        Returns:
            Path to JSON file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"finllm_results_{timestamp}.json"
        
        file_path = os.path.join(self.output_dir, filename)
        
        # Prepare serializable results
        serializable_results = self._make_serializable(self.results)
        
        # Write to file
        with open(file_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Results exported to: {file_path}")
        return file_path
    
    def _make_serializable(self, obj):
        """
        Make object JSON serializable
        
        Args:
            obj: Object to convert
            
        Returns:
            JSON serializable object
        """
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        else:
            return obj