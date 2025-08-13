import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

class PricePredictor:
    """Machine learning model for predicting ride-sharing prices"""
    
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.service_encoder = LabelEncoder()
        self.location_encoder = LabelEncoder()
        self.is_trained = False
        self.feature_columns = ['hour', 'day_of_week', 'month', 'is_weekend', 'service_encoded', 'location_encoded']
        self.average_price = 0
    
    def train_model(self, data):
        """Train the price prediction model"""
        if data is None or data.empty:
            raise ValueError("No data available for training")
        
        try:
            # Prepare features
            X = self._prepare_features(data)
            y = data['price']
            
            # Split data for training
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train model
            self.model.fit(X_train, y_train)
            
            # Calculate average price for comparison
            self.average_price = y.mean()
            
            # Mark as trained
            self.is_trained = True
            
            # Calculate training score
            train_score = self.model.score(X_train, y_train)
            test_score = self.model.score(X_test, y_test)
            
            print(f"Model trained successfully. Train R²: {train_score:.3f}, Test R²: {test_score:.3f}")
            
        except Exception as e:
            raise Exception(f"Error training model: {str(e)}")
    
    def _prepare_features(self, data):
        """Prepare feature matrix for training or prediction"""
        features = data.copy()
        
        # Encode categorical variables
        if 'service' in features.columns:
            if not hasattr(self.service_encoder, 'classes_'):
                features['service_encoded'] = self.service_encoder.fit_transform(features['service'])
            else:
                # Handle unseen categories during prediction
                features['service_encoded'] = features['service'].apply(
                    lambda x: self._safe_transform(self.service_encoder, x)
                )
        
        if 'location' in features.columns:
            if not hasattr(self.location_encoder, 'classes_'):
                features['location_encoded'] = self.location_encoder.fit_transform(features['location'])
            else:
                # Handle unseen categories during prediction
                features['location_encoded'] = features['location'].apply(
                    lambda x: self._safe_transform(self.location_encoder, x)
                )
        
        # Select only the features needed for the model
        return features[self.feature_columns]
    
    def _safe_transform(self, encoder, value):
        """Safely transform categorical value, handling unseen categories"""
        try:
            return encoder.transform([value])[0]
        except ValueError:
            # Return the most common class for unseen categories
            return encoder.transform([encoder.classes_[0]])[0]
    
    def predict_price(self, datetime_input, service, location):
        """Predict price for given parameters"""
        if not self.is_trained:
            raise ValueError("Model has not been trained yet")
        
        try:
            # Create feature vector for prediction
            features = {
                'hour': datetime_input.hour,
                'day_of_week': datetime_input.weekday(),
                'month': datetime_input.month,
                'is_weekend': 1 if datetime_input.weekday() >= 5 else 0,
                'service': service,
                'location': location
            }
            
            # Convert to DataFrame
            feature_df = pd.DataFrame([features])
            
            # Add encoded features
            feature_df['service_encoded'] = self._safe_transform(self.service_encoder, service)
            feature_df['location_encoded'] = self._safe_transform(self.location_encoder, location)
            
            # Select only model features
            X = feature_df[self.feature_columns]
            
            # Make prediction
            prediction = self.model.predict(X)[0]
            
            # Ensure prediction is positive
            prediction = max(prediction, 1.0)
            
            return prediction
            
        except Exception as e:
            print(f"Error making prediction: {str(e)}")
            return None
    
    def get_average_price(self):
        """Get the average price from training data"""
        return self.average_price
    
    def get_price_category(self, price):
        """Categorize price level"""
        if self.average_price == 0:
            return "Unknown"
        
        ratio = price / self.average_price
        
        if ratio < 0.8:
            return "Low"
        elif ratio < 1.2:
            return "Medium"
        elif ratio < 1.5:
            return "High"
        else:
            return "Peak"
    
    def get_feature_importance(self):
        """Get feature importance from trained model"""
        if not self.is_trained:
            return None
        
        try:
            importance = self.model.feature_importances_
            feature_importance = dict(zip(self.feature_columns, importance))
            return sorted(feature_importance.items(), key=lambda x: float(x[1]), reverse=True)
        except Exception as e:
            print(f"Error getting feature importance: {str(e)}")
            return None
