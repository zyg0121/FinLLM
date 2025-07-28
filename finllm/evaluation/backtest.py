import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


class PortfolioBacktester:
    """
    Backtester for evaluating trading performance based on model predictions
    """
    
    def __init__(self, predictions_df=None, returns_df=None, prices_df=None):
        """
        Initialize backtester
        
        Args:
            predictions_df: DataFrame with predictions (index: date, columns: assets)
            returns_df: DataFrame with actual returns (index: date, columns: assets)
            prices_df: DataFrame with asset prices (index: date, columns: assets)
        """
        self.predictions_df = predictions_df
        self.returns_df = returns_df
        self.prices_df = prices_df
        
        # If returns_df is not provided but prices_df is, calculate returns
        if returns_df is None and prices_df is not None:
            self.returns_df = prices_df.pct_change()
    
    def backtest_long_short_portfolio(self, n_long=10, n_short=10, 
                                     transaction_cost=0.001, 
                                     market_neutral=True,
                                     capital=1000000):
        """
        Backtest a long-short portfolio strategy
        
        Args:
            n_long: Number of assets to long
            n_short: Number of assets to short
            transaction_cost: One-way transaction cost as fraction
            market_neutral: If True, equal dollar long and short
            capital: Initial capital
            
        Returns:
            Dictionary with backtest results
        """
        if self.predictions_df is None or self.returns_df is None:
            raise ValueError("Both predictions_df and returns_df are required for backtesting")
            
        # Get common dates and assets
        common_dates = sorted(set(self.predictions_df.index) & set(self.returns_df.index))
        common_assets = sorted(set(self.predictions_df.columns) & set(self.returns_df.columns))
        
        # Filter to common dates and assets
        preds = self.predictions_df.loc[common_dates, common_assets]
        rets = self.returns_df.loc[common_dates, common_assets]
        
        # Initialize results
        portfolio_returns = []
        portfolio_values = [capital]
        positions = []
        turnover_ratios = []
        position_counts = []
        long_exposure = []
        short_exposure = []
        
        # Previous positions
        prev_positions = None
        
        # Loop through dates
        for date in tqdm(common_dates, desc="Backtesting"):
            # Get predictions for current date
            day_preds = preds.loc[date]
            
            # Remove NaN predictions
            valid_preds = day_preds.dropna()
            
            # Get assets to long and short
            if len(valid_preds) >= n_long + n_short:
                long_assets = valid_preds.nlargest(n_long).index
                short_assets = valid_preds.nsmallest(n_short).index
            else:
                # Not enough assets, skip day
                portfolio_returns.append(0)
                portfolio_values.append(portfolio_values[-1])
                positions.append(prev_positions if prev_positions is not None else pd.Series())
                turnover_ratios.append(0)
                position_counts.append(0)
                long_exposure.append(0)
                short_exposure.append(0)
                continue
            
            # Create position weights
            new_positions = pd.Series(0.0, index=common_assets)
            
            # Equal weighting within long and short portfolios
            if market_neutral:
                # Equal dollars long and short
                new_positions[long_assets] = 1.0 / n_long
                new_positions[short_assets] = -1.0 / n_short
            else:
                # Not market neutral (more flexible)
                new_positions[long_assets] = 1.0 / n_long
                new_positions[short_assets] = -1.0 / n_short
                # Scale to sum to 1.0 (net 100% long)
                net_exposure = new_positions.sum()
                new_positions = new_positions / abs(net_exposure)
            
            # Calculate turnover
            if prev_positions is not None:
                # Fill missing assets with 0
                prev_full = pd.Series(0.0, index=common_assets)
                prev_full[prev_positions.index] = prev_positions
                
                # Calculate turnover (sum of absolute weight changes)
                turnover = (new_positions - prev_full).abs().sum() / 2
                turnover_ratios.append(turnover)
            else:
                # First day, turnover is 1.0 (100% of portfolio)
                turnover = 1.0
                turnover_ratios.append(turnover)
            
            # Store new positions for next iteration
            prev_positions = new_positions.copy()
            positions.append(new_positions)
            
            # Calculate transaction costs
            tc = turnover * transaction_cost
            
            # Get day's returns
            day_returns = rets.loc[date]
            
            # Calculate portfolio return
            port_return = (new_positions * day_returns).sum() - tc
            portfolio_returns.append(port_return)
            
            # Update portfolio value
            portfolio_values.append(portfolio_values[-1] * (1 + port_return))
            
            # Store position metrics
            position_counts.append(len(new_positions[new_positions != 0]))
            long_exposure.append(new_positions[new_positions > 0].sum())
            short_exposure.append(abs(new_positions[new_positions < 0].sum()))
        
        # Convert lists to Series/DataFrame
        portfolio_returns = pd.Series(portfolio_returns, index=common_dates)
        portfolio_values = pd.Series(portfolio_values, index=[common_dates[0] - pd.Timedelta(days=1)] + common_dates)
        turnover_ratios = pd.Series(turnover_ratios, index=common_dates)
        position_counts = pd.Series(position_counts, index=common_dates)
        long_exposure = pd.Series(long_exposure, index=common_dates)
        short_exposure = pd.Series(short_exposure, index=common_dates)
        
        # Calculate performance metrics
        performance_metrics = self.calculate_performance_metrics(portfolio_returns, capital)
        
        return {
            "portfolio_returns": portfolio_returns,
            "portfolio_values": portfolio_values,
            "turnover": turnover_ratios,
            "position_counts": position_counts,
            "long_exposure": long_exposure,
            "short_exposure": short_exposure,
            "performance_metrics": performance_metrics
        }
    
    def calculate_performance_metrics(self, returns, initial_capital=1000000):
        """
        Calculate portfolio performance metrics
        
        Args:
            returns: Series of portfolio returns
            initial_capital: Initial capital
            
        Returns:
            Dictionary of performance metrics
        """
        if len(returns) < 2:
            return {
                "total_return": 0,
                "annualized_return": 0,
                "annualized_volatility": 0,
                "sharpe_ratio": 0,
                "information_ratio": 0,
                "sortino_ratio": 0,
                "max_drawdown": 0,
                "calmar_ratio": 0,
                "win_rate": 0,
                "avg_win": 0,
                "avg_loss": 0,
                "profit_factor": 0,
                "recovery_factor": 0,
                "trading_days": 0
            }
        
        # Calculate basic metrics
        annual_factor = 252  # Trading days per year
        total_return = (1 + returns).prod() - 1
        annualized_return = (1 + returns).prod() ** (annual_factor / len(returns)) - 1
        annualized_volatility = returns.std() * np.sqrt(annual_factor)
        
        # Sharpe ratio (assuming 0 risk-free rate)
        sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else 0
        
        # Information ratio (assuming benchmark return is 0)
        # In practice, you'd compare to a benchmark
        benchmark_returns = pd.Series(0, index=returns.index)
        tracking_error = (returns - benchmark_returns).std() * np.sqrt(annual_factor)
        information_ratio = annualized_return / tracking_error if tracking_error > 0 else 0
        
        # Sortino ratio (downside risk)
        downside_returns = returns[returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(annual_factor) if len(downside_returns) > 0 else 0
        sortino_ratio = annualized_return / downside_volatility if downside_volatility > 0 else 0
        
        # Maximum drawdown
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.cummax()
        drawdowns = (cum_returns / running_max) - 1
        max_drawdown = abs(drawdowns.min())
        
        # Calmar ratio
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else np.inf
        
        # Win rate and profit metrics
        win_rate = (returns > 0).mean()
        avg_win = returns[returns > 0].mean() if len(returns[returns > 0]) > 0 else 0
        avg_loss = returns[returns < 0].mean() if len(returns[returns < 0]) > 0 else 0
        
        # Profit factor (gross profit / gross loss)
        gross_profit = (returns[returns > 0]).sum()
        gross_loss = abs((returns[returns < 0]).sum()) if len(returns[returns < 0]) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
        
        # Recovery factor
        recovery_factor = total_return / max_drawdown if max_drawdown > 0 else np.inf
        
        # Expected shortfall (ES) at 95% confidence level
        es_95 = returns.quantile(0.05)
        
        return {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "annualized_volatility": annualized_volatility,
            "sharpe_ratio": sharpe_ratio,
            "information_ratio": information_ratio,
            "sortino_ratio": sortino_ratio,
            "max_drawdown": max_drawdown,
            "calmar_ratio": calmar_ratio,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "recovery_factor": recovery_factor,
            "expected_shortfall_95": es_95,
            "trading_days": len(returns)
        }
    
    def run_transaction_cost_sensitivity(self, n_long=10, n_short=10, 
                                        cost_range=None,
                                        market_neutral=True):
        """
        Run transaction cost sensitivity analysis
        
        Args:
            n_long: Number of assets to long
            n_short: Number of assets to short
            cost_range: List of transaction costs to test
            market_neutral: If True, equal dollar long and short
            
        Returns:
            DataFrame with sensitivity results
        """
        if cost_range is None:
            cost_range = np.arange(0.0005, 0.031, 0.0005)  # 5bp to 300bp
        
        results = []
        
        for tc in tqdm(cost_range, desc="Transaction Cost Sensitivity"):
            backtest = self.backtest_long_short_portfolio(
                n_long=n_long,
                n_short=n_short,
                transaction_cost=tc,
                market_neutral=market_neutral
            )
            
            metrics = backtest["performance_metrics"]
            
            results.append({
                "transaction_cost_bps": tc * 10000,  # Convert to basis points
                "annualized_return": metrics["annualized_return"],
                "sharpe_ratio": metrics["sharpe_ratio"],
                "max_drawdown": metrics["max_drawdown"],
                "calmar_ratio": metrics["calmar_ratio"]
            })
        
        return pd.DataFrame(results)
    
    def plot_equity_curve(self, backtest_results):
        """
        Plot equity curve from backtest results
        
        Args:
            backtest_results: Results from backtest_long_short_portfolio
        """
        plt.figure(figsize=(12, 6))
        
        # Plot equity curve
        portfolio_values = backtest_results["portfolio_values"]
        plt.plot(portfolio_values, linewidth=2)
        
        # Calculate drawdowns for shading
        drawdowns = (portfolio_values / portfolio_values.cummax() - 1) * 100
        
        # Shade drawdowns
        for i in range(len(drawdowns) - 1):
            if drawdowns.iloc[i] < -5:  # Only shade significant drawdowns
                plt.axvspan(drawdowns.index[i], drawdowns.index[i+1], 
                           alpha=0.2, color='red')
        
        plt.title("Equity Curve", fontsize=14)
        plt.ylabel("Portfolio Value ($)")
        plt.grid(True, alpha=0.3)
        
        # Add performance metrics as text
        metrics = backtest_results["performance_metrics"]
        metrics_text = (
            f"Annual Return: {metrics['annualized_return']*100:.2f}%\n"
            f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n"
            f"Max Drawdown: {metrics['max_drawdown']*100:.2f}%\n"
            f"Calmar Ratio: {metrics['calmar_ratio']:.2f}"
        )
        
        plt.figtext(0.15, 0.15, metrics_text, fontsize=12, 
                   bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
    
    def plot_return_distribution(self, backtest_results):
        """
        Plot return distribution from backtest results
        
        Args:
            backtest_results: Results from backtest_long_short_portfolio
        """
        returns = backtest_results["portfolio_returns"]
        
        plt.figure(figsize=(12, 6))
        
        # Plot return distribution
        sns.histplot(returns, bins=50, kde=True)
        
        # Add vertical line for mean
        plt.axvline(returns.mean(), color='red', linestyle='--', 
                   label=f"Mean: {returns.mean()*100:.2f}%")
        
        # Add vertical lines for ES95
        es_95 = np.percentile(returns, 5)
        plt.axvline(es_95, color='darkred', linestyle='-', 
                   label=f"ES(95): {es_95*100:.2f}%")
        
        plt.title("Daily Return Distribution", fontsize=14)
        plt.xlabel("Return")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_monthly_returns(self, backtest_results):
        """
        Plot monthly returns heatmap
        
        Args:
            backtest_results: Results from backtest_long_short_portfolio
        """
        returns = backtest_results["portfolio_returns"]
        
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
        
        plt.figure(figsize=(12, 8))
        
        # Plot heatmap
        sns.heatmap(monthly_pivot * 100, 
                   annot=True, 
                   fmt=".2f", 
                   cmap="RdYlGn",
                   cbar_kws={'label': '%'},
                   center=0)
        
        plt.title("Monthly Returns (%)", fontsize=14)
        plt.ylabel("Year")
        plt.xlabel("Month")
        plt.tight_layout()
        plt.show()


class RiskAnalyzer:
    """
    Tools for analyzing risk attribution and factor exposures
    """
    
    def __init__(self, returns_series, factor_returns=None):
        """
        Initialize risk analyzer
        
        Args:
            returns_series: Series of portfolio returns
            factor_returns: DataFrame of factor returns (e.g., market, size, value)
        """
        self.returns = returns_series
        self.factor_returns = factor_returns
    
    def calculate_factor_exposures(self):
        """
        Calculate factor exposures (betas) using linear regression
        
        Returns:
            Dictionary with factor exposures and metrics
        """
        if self.factor_returns is None:
            raise ValueError("Factor returns are required for factor exposure analysis")
        
        # Align dates
        common_dates = sorted(set(self.returns.index) & set(self.factor_returns.index))
        y = self.returns.loc[common_dates]
        X = self.factor_returns.loc[common_dates]
        
        # Add constant for alpha
        X = sm.add_constant(X)
        
        # Fit linear regression
        model = sm.OLS(y, X).fit()
        
        # Extract factor exposures (betas)
        exposures = model.params[1:]  # Skip alpha (constant)
        
        # Calculate factor contributions
        factor_returns = self.factor_returns.loc[common_dates]
        factor_contributions = {}
        
        for factor in exposures.index:
            contribution = exposures[factor] * factor_returns[factor]
            factor_contributions[factor] = contribution.sum()
        
        # Calculate metrics
        total_contribution = sum(factor_contributions.values())
        alpha = model.params['const'] * len(common_dates)
        idiosyncratic = y.sum() - total_contribution - alpha
        
        return {
            "exposures": exposures,
            "factor_contributions": factor_contributions,
            "alpha": alpha,
            "idiosyncratic": idiosyncratic,
            "r_squared": model.rsquared,
            "model": model
        }
    
    def plot_factor_contributions(self, factor_exposures=None):
        """
        Plot factor contributions
        
        Args:
            factor_exposures: Results from calculate_factor_exposures
        """
        if factor_exposures is None:
            factor_exposures = self.calculate_factor_exposures()
        
        # Extract contributions
        contributions = factor_exposures["factor_contributions"]
        contributions["Alpha"] = factor_exposures["alpha"]
        contributions["Idiosyncratic"] = factor_exposures["idiosyncratic"]
        
        # Sort by absolute value
        sorted_contrib = pd.Series(contributions).sort_values(ascending=False)
        
        # Plot
        plt.figure(figsize=(12, 6))
        
        colors = ['green' if x > 0 else 'red' for x in sorted_contrib]
        bars = plt.bar(sorted_contrib.index, sorted_contrib, color=colors)
        
        plt.title("Factor Return Contributions", fontsize=14)
        plt.ylabel("Return Contribution")
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        # Add total return as text
        total_return = sum(sorted_contrib)
        plt.figtext(0.15, 0.85, f"Total Return: {total_return*100:.2f}%", 
                   fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
    
    def calculate_tail_risk_metrics(self):
        """
        Calculate tail risk metrics (Value at Risk, Expected Shortfall)
        
        Returns:
            Dictionary with tail risk metrics
        """
        # Value at Risk (VaR)
        var_95 = np.percentile(self.returns, 5)
        var_99 = np.percentile(self.returns, 1)
        
        # Expected Shortfall (ES)
        es_95 = self.returns[self.returns <= var_95].mean()
        es_99 = self.returns[self.returns <= var_99].mean()
        
        # Maximum drawdown
        cum_returns = (1 + self.returns).cumprod()
        running_max = cum_returns.cummax()
        drawdowns = (cum_returns / running_max) - 1
        max_drawdown = abs(drawdowns.min())
        
        # Conditional drawdown at risk (CDaR)
        cdar_95 = drawdowns[drawdowns <= np.percentile(drawdowns, 5)].mean()
        
        return {
            "var_95": var_95,
            "var_99": var_99,
            "es_95": es_95,
            "es_99": es_99,
            "max_drawdown": max_drawdown,
            "cdar_95": cdar_95
        }
    
    def compare_tail_risk_across_models(self, model_returns_dict):
        """
        Compare tail risk metrics across models
        
        Args:
            model_returns_dict: Dictionary mapping model names to return series
            
        Returns:
            DataFrame with tail risk metrics for each model
        """
        metrics = []
        
        for model_name, returns in model_returns_dict.items():
            analyzer = RiskAnalyzer(returns)
            tail_risk = analyzer.calculate_tail_risk_metrics()
            
            # Add model name
            tail_risk["Model"] = model_name
            metrics.append(tail_risk)
        
        # Combine into DataFrame
        return pd.DataFrame(metrics).set_index("Model")