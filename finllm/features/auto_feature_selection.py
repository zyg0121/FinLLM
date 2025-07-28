import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.inspection import permutation_importance
import shap
from tqdm import tqdm
import matplotlib.pyplot as plt

class FeatureSelector:
    """
    Automatic feature selection for FinLLM
    """
    def __init__(self, cut_threshold=0.15):
        """
        Initialize feature selector
        
        Args:
            cut_threshold: Proportion of features to cut (0-1)
        """
        self.cut_threshold = cut_threshold
        self.feature_importances = {}
        self.selected_features = {}
    
    def select_features_lgb_shap(self, X_train, y_train, feature_names, num_features=None):
        """
        Select features using LightGBM and SHAP values
        
        Args:
            X_train: Training features
            y_train: Training targets
            feature_names: List of feature names
            num_features: Number of features to select (if None, use cut_threshold)
            
        Returns:
            List of selected feature names
        """
        print("Running LightGBM-SHAP feature selection...")
        
        # Train a LightGBM model
        params = {
            'objective': 'regression',
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'n_estimators': 100,
            'max_depth': 5,
            'num_leaves': 31,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
        
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train)
        
        # Calculate SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_train)
        
        # Calculate feature importance
        feature_importance = np.mean(np.abs(shap_values), axis=0)
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        # Store feature importances
        self.feature_importances['lgb_shap'] = importance_df
        
        # Select top features
        if num_features is None:
            num_features = int((1 - self.cut_threshold) * len(feature_names))
        
        selected_features = importance_df.head(num_features)['feature'].tolist()
        self.selected_features['lgb_shap'] = selected_features
        
        print(f"Selected {len(selected_features)} features using LightGBM-SHAP")
        
        return selected_features
    
    def select_features_permutation(self, model, X_val, y_val, feature_names, num_features=None):
        """
        Select features using permutation importance
        
        Args:
            model: Trained model
            X_val: Validation features
            y_val: Validation targets
            feature_names: List of feature names
            num_features: Number of features to select (if None, use cut_threshold)
            
        Returns:
            List of selected feature names
        """
        print("Running permutation importance feature selection...")
        
        # Calculate permutation importance
        perm_importance = permutation_importance(
            model, X_val, y_val, 
            n_repeats=10, 
            random_state=42,
            n_jobs=-1
        )
        
        # Create importance DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': perm_importance.importances_mean
        }).sort_values('importance', ascending=False)
        
        # Store feature importances
        self.feature_importances['permutation'] = importance_df
        
        # Select top features
        if num_features is None:
            num_features = int((1 - self.cut_threshold) * len(feature_names))
        
        selected_features = importance_df.head(num_features)['feature'].tolist()
        self.selected_features['permutation'] = selected_features
        
        print(f"Selected {len(selected_features)} features using permutation importance")
        
        return selected_features
    
    def select_features_f_regression(self, X_train, y_train, feature_names, num_features=None):
        """
        Select features using F-regression
        
        Args:
            X_train: Training features
            y_train: Training targets
            feature_names: List of feature names
            num_features: Number of features to select (if None, use cut_threshold)
            
        Returns:
            List of selected feature names
        """
        print("Running F-regression feature selection...")
        
        # Calculate F-regression scores
        if num_features is None:
            num_features = int((1 - self.cut_threshold) * len(feature_names))
        
        selector = SelectKBest(f_regression, k=num_features)
        selector.fit(X_train, y_train)
        
        # Get feature scores
        scores = selector.scores_
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': scores
        }).sort_values('importance', ascending=False)
        
        # Store feature importances
        self.feature_importances['f_regression'] = importance_df
        
        # Get selected features
        mask = selector.get_support()
        selected_features = [feature_names[i] for i in range(len(feature_names)) if mask[i]]
        self.selected_features['f_regression'] = selected_features
        
        print(f"Selected {len(selected_features)} features using F-regression")
        
        return selected_features
    
    def select_features_ensemble(self, X_train, y_train, X_val, y_val, model, feature_names, num_features=None):
        """
        Ensemble multiple feature selection methods
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            model: Trained model
            feature_names: List of feature names
            num_features: Number of features to select (if None, use cut_threshold)
            
        Returns:
            List of selected feature names
        """
        print("Running ensemble feature selection...")
        
        # Run all selection methods
        lgb_shap_features = self.select_features_lgb_shap(X_train, y_train, feature_names)
        perm_features = self.select_features_permutation(model, X_val, y_val, feature_names)
        f_reg_features = self.select_features_f_regression(X_train, y_train, feature_names)
        
        # Count votes for each feature
        feature_votes = {}
        for feature in feature_names:
            votes = 0
            if feature in lgb_shap_features:
                votes += 1
            if feature in perm_features:
                votes += 1
            if feature in f_reg_features:
                votes += 1
            feature_votes[feature] = votes
        
        # Create votes DataFrame
        votes_df = pd.DataFrame({
            'feature': list(feature_votes.keys()),
            'votes': list(feature_votes.values())
        }).sort_values(['votes', 'feature'], ascending=[False, True])
        
        # Store feature votes
        self.feature_importances['ensemble_votes'] = votes_df
        
        # Select features with at least 2 votes
        selected_features = votes_df[votes_df['votes'] >= 2]['feature'].tolist()
        
        # If too few features selected, take top ones
        if num_features is not None and len(selected_features) < num_features:
            lgb_shap_df = self.feature_importances['lgb_shap']
            top_features = lgb_shap_df.head(num_features)['feature'].tolist()
            # Add any missing top features
            for feature in top_features:
                if feature not in selected_features:
                    selected_features.append(feature)
                if len(selected_features) >= num_features:
                    break
        
        self.selected_features['ensemble'] = selected_features
        
        print(f"Selected {len(selected_features)} features using ensemble method")
        
        return selected_features
    
    def plot_feature_importance(self, method='lgb_shap', top_n=20):
        """
        Plot feature importance
        
        Args:
            method: Feature selection method
            top_n: Number of top features to plot
        """
        if method not in self.feature_importances:
            print(f"No feature importance available for method '{method}'")
            return
        
        importance_df = self.feature_importances[method]
        top_features = importance_df.head(top_n)
        
        plt.figure(figsize=(12, 8))
        plt.barh(top_features['feature'][::-1], top_features['importance'][::-1])
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title(f'Top {top_n} Features ({method})')
        plt.tight_layout()
        plt.savefig(f'feature_importance_{method}.png')
        plt.show()