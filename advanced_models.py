import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class AdvancedModelEnsemble:
    """Advanced machine learning ensemble for ride-sharing price prediction"""
    
    def __init__(self):
        self.models = {
            'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=150, learning_rate=0.1, random_state=42),
            'Neural Network': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
            'Linear Regression': LinearRegression()
        }
        self.scaler = StandardScaler()
        self.trained_models = {}
        self.model_scores = {}
        self.best_model_name = None
        self.is_trained = False
        
    def train_models(self, X, y):
        """Train all models and select the best performing one"""
        X_scaled = self.scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        self.model_scores = {}
        
        for name, model in self.models.items():
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)
                
                # Calculate metrics
                train_r2 = r2_score(y_train, y_pred_train)
                test_r2 = r2_score(y_test, y_pred_test)
                test_mae = mean_absolute_error(y_test, y_pred_test)
                test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
                
                # Cross-validation score
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
                
                self.model_scores[name] = {
                    'train_r2': train_r2,
                    'test_r2': test_r2,
                    'test_mae': test_mae,
                    'test_rmse': test_rmse,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std()
                }
                
                self.trained_models[name] = model
                
            except Exception as e:
                print(f"Error training {name}: {str(e)}")
                continue
        
        # Select best model based on test R²
        if self.model_scores:
            self.best_model_name = max(self.model_scores.keys(), 
                                     key=lambda x: self.model_scores[x]['test_r2'])
            self.is_trained = True
        
        return self.model_scores
    
    def predict(self, X):
        """Make prediction using the best model"""
        if not self.is_trained or self.best_model_name not in self.trained_models:
            raise ValueError("Models not trained yet")
        
        X_scaled = self.scaler.transform(X)
        return self.trained_models[self.best_model_name].predict(X_scaled)
    
    def get_ensemble_prediction(self, X):
        """Get weighted ensemble prediction from all models"""
        if not self.is_trained:
            raise ValueError("Models not trained yet")
        
        X_scaled = self.scaler.transform(X)
        predictions = []
        weights = []
        
        for name, model in self.trained_models.items():
            pred = model.predict(X_scaled)
            predictions.append(pred)
            # Weight by test R² score
            weights.append(max(0, self.model_scores[name]['test_r2']))
        
        if sum(weights) == 0:
            weights = [1] * len(weights)  # Equal weights if all scores are 0
        
        # Weighted average
        weighted_pred = np.average(predictions, axis=0, weights=weights)
        return weighted_pred
    
    def get_feature_importance(self):
        """Get feature importance from tree-based models"""
        if not self.is_trained:
            return None
        
        importance_data = {}
        
        # Random Forest importance
        if 'Random Forest' in self.trained_models:
            rf_importance = self.trained_models['Random Forest'].feature_importances_
            importance_data['Random Forest'] = rf_importance
        
        # Gradient Boosting importance
        if 'Gradient Boosting' in self.trained_models:
            gb_importance = self.trained_models['Gradient Boosting'].feature_importances_
            importance_data['Gradient Boosting'] = gb_importance
        
        return importance_data
    
    def get_model_comparison(self):
        """Get comparison of all model performances"""
        return self.model_scores
    
    def get_best_model(self):
        """Get the best performing model"""
        if self.best_model_name:
            return self.best_model_name, self.trained_models[self.best_model_name]
        return None, None