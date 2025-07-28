import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr, ttest_1samp
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class ForecastingMetrics:
    """
    Metrics for evaluating forecasting accuracy of FinLLM and baseline models
    """
    
    @staticmethod
    def information_coefficient(predictions, actual):
        """
        Calculate Information Coefficient (Spearman rank correlation)
        
        Args:
            predictions: Array of predicted returns
            actual: Array of actual returns
            
        Returns:
            IC value
        """
        # Remove NaN values
        mask = ~(np.isnan(predictions) | np.isnan(actual))
        predictions = predictions[mask]
        actual = actual[mask]
        
        if len(predictions) < 2:
            return np.nan
        
        # Calculate Spearman rank correlation
        ic, _ = spearmanr(predictions, actual)
        return ic
    
    @staticmethod
    def information_ratio(ic_series):
        """
        Calculate Information Ratio (IR)
        
        Args:
            ic_series: Series of daily ICs
            
        Returns:
            Information Ratio
        """
        # Remove NaN values
        ic_series = ic_series.dropna()
        
        if len(ic_series) < 2:
            return np.nan
        
        # Calculate IR (mean IC / std IC)
        mean_ic = ic_series.mean()
        std_ic = ic_series.std()
        
        if std_ic == 0:
            return np.nan
            
        return mean_ic / std_ic
    
    @staticmethod
    def calc_daily_ic(predictions_df, returns_df):
        """
        Calculate daily Information Coefficient
        
        Args:
            predictions_df: DataFrame with predictions (index: date, columns: assets)
            returns_df: DataFrame with actual returns (index: date, columns: assets)
            
        Returns:
            Series of daily ICs
        """
        daily_ics = {}
        
        common_dates = sorted(set(predictions_df.index) & set(returns_df.index))
        
        for date in common_dates:
            preds = predictions_df.loc[date].values
            actual = returns_df.loc[date].values
            
            # Remove assets with NaN predictions or returns
            mask = ~(np.isnan(preds) | np.isnan(actual))
            preds = preds[mask]
            actual = actual[mask]
            
            if len(preds) < 2:
                daily_ics[date] = np.nan
                continue
                
            ic = ForecastingMetrics.information_coefficient(preds, actual)
            daily_ics[date] = ic
        
        return pd.Series(daily_ics)
    
    @staticmethod
    def calc_ic_metrics(daily_ics):
        """
        Calculate IC summary statistics
        
        Args:
            daily_ics: Series of daily ICs
        
        Returns:
            Dictionary of IC metrics
        """
        # Remove NaN values
        ic_series = daily_ics.dropna()
        
        if len(ic_series) == 0:
            return {
                "mean_ic": np.nan,
                "median_ic": np.nan,
                "std_ic": np.nan,
                "t_stat": np.nan,
                "p_value": np.nan,
                "ic_ir": np.nan
            }
        
        # Calculate metrics
        mean_ic = ic_series.mean()
        median_ic = ic_series.median()
        std_ic = ic_series.std()
        
        # T-test to determine if mean IC is significantly different from 0
        t_stat, p_value = ttest_1samp(ic_series, 0)
        
        # Calculate Information Ratio
        ic_ir = ForecastingMetrics.information_ratio(ic_series)
        
        return {
            "mean_ic": mean_ic,
            "median_ic": median_ic,
            "std_ic": std_ic,
            "t_stat": t_stat,
            "p_value": p_value,
            "ic_ir": ic_ir
        }
    
    @staticmethod
    def regression_metrics(predictions, actual):
        """
        Calculate regression metrics (MSE, RMSE, MAE, R2)
        
        Args:
            predictions: Array of predicted returns
            actual: Array of actual returns
            
        Returns:
            Dictionary of regression metrics
        """
        # Remove NaN values
        mask = ~(np.isnan(predictions) | np.isnan(actual))
        predictions = predictions[mask]
        actual = actual[mask]
        
        if len(predictions) < 2:
            return {
                "mse": np.nan,
                "rmse": np.nan,
                "mae": np.nan,
                "r2": np.nan
            }
        
        # Calculate metrics
        mse = mean_squared_error(actual, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual, predictions)
        r2 = r2_score(actual, predictions)
        
        return {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2": r2
        }
    
    @staticmethod
    def directional_accuracy(predictions, actual):
        """
        Calculate directional accuracy (% correct sign predictions)
        
        Args:
            predictions: Array of predicted returns
            actual: Array of actual returns
            
        Returns:
            Directional accuracy (0-1)
        """
        # Remove NaN values
        mask = ~(np.isnan(predictions) | np.isnan(actual))
        predictions = predictions[mask]
        actual = actual[mask]
        
        if len(predictions) < 1:
            return np.nan
        
        # Calculate directional accuracy
        correct_sign = np.sign(predictions) == np.sign(actual)
        return np.mean(correct_sign)
    
    @staticmethod
    def bootstrap_confidence_intervals(predictions, actual, metric_func, n_bootstrap=1000, ci=0.95):
        """
        Calculate bootstrap confidence intervals for a metric
        
        Args:
            predictions: Array of predicted returns
            actual: Array of actual returns
            metric_func: Function to calculate metric
            n_bootstrap: Number of bootstrap samples
            ci: Confidence interval level
            
        Returns:
            Dictionary with confidence interval bounds
        """
        # Remove NaN values
        mask = ~(np.isnan(predictions) | np.isnan(actual))
        predictions = predictions[mask]
        actual = actual[mask]
        
        if len(predictions) < 10:  # Need reasonable sample size for bootstrap
            return {
                "lower": np.nan,
                "upper": np.nan,
                "mean": np.nan
            }
        
        # Calculate bootstrap samples
        bootstrap_metrics = []
        indices = np.arange(len(predictions))
        
        for _ in range(n_bootstrap):
            # Sample with replacement
            bootstrap_indices = np.random.choice(indices, size=len(indices), replace=True)
            bootstrap_preds = predictions[bootstrap_indices]
            bootstrap_actual = actual[bootstrap_indices]
            
            # Calculate metric
            metric = metric_func(bootstrap_preds, bootstrap_actual)
            bootstrap_metrics.append(metric)
        
        # Calculate confidence interval
        lower_bound = np.percentile(bootstrap_metrics, (1 - ci) * 100 / 2)
        upper_bound = np.percentile(bootstrap_metrics, 100 - (1 - ci) * 100 / 2)
        mean_value = np.mean(bootstrap_metrics)
        
        return {
            "lower": lower_bound,
            "upper": upper_bound,
            "mean": mean_value
        }