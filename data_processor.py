import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from advanced_models import AdvancedModelEnsemble
from surge_pricing import SurgePricingAnalyzer
from distance_optimizer import DistanceOptimizer

class DataProcessor:
    """Handles data loading and processing for ride-sharing price prediction"""
    
    def __init__(self):
        self.data = None
        self.services = ['Uber', 'Lyft', 'Taxi']
        self.locations = ['Downtown', 'Airport', 'Business District', 'Residential Area', 'Shopping Mall']
        self.surge_analyzer = SurgePricingAnalyzer()
        self.distance_optimizer = DistanceOptimizer()
        self.advanced_models = AdvancedModelEnsemble()
    
    def load_data(self):
        """Load ride-sharing data from available sources"""
        try:
            # First, try to load from external data source or API
            data = self._load_from_external_source()
            
            if data is not None:
                self.data = data
                return data
            else:
                # If no external data available, return None to trigger error handling
                return None
                
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return None
    
    def _load_from_external_source(self):
        """Attempt to load data from external sources"""
        # Try to load from environment variables or API endpoints
        api_key = os.getenv("RIDESHARE_API_KEY")
        data_url = os.getenv("RIDESHARE_DATA_URL")
        
        if api_key and data_url:
            try:
                # This would be implemented with actual API calls
                # For now, return None to indicate no data source available
                return None
            except Exception as e:
                print(f"Error loading from external source: {str(e)}")
                return None
        
        # Check if CSV file exists
        if os.path.exists("rideshare_data.csv"):
            try:
                data = pd.read_csv("rideshare_data.csv")
                return self._process_loaded_data(data)
            except Exception as e:
                print(f"Error loading CSV file: {str(e)}")
                return None
        
        return None
    
    def _process_loaded_data(self, data):
        """Process and clean loaded data"""
        try:
            # Check if this is Uber data format
            if 'fare_amount' in data.columns and 'pickup_datetime' in data.columns:
                return self._process_uber_data(data)
            
            # Otherwise, ensure required columns exist for standard format
            required_columns = ['datetime', 'service', 'location', 'price']
            for col in required_columns:
                if col not in data.columns:
                    raise ValueError(f"Missing required column: {col}")
            
            # Convert datetime column
            data['datetime'] = pd.to_datetime(data['datetime'])
            
            # Extract time features
            data['hour'] = data['datetime'].dt.hour
            data['day_of_week'] = data['datetime'].dt.dayofweek
            data['month'] = data['datetime'].dt.month
            data['is_weekend'] = data['day_of_week'].isin([5, 6]).astype(int)
            
            # Clean price data
            data['price'] = pd.to_numeric(data['price'], errors='coerce')
            data = data.dropna(subset=['price'])
            
            return data
            
        except Exception as e:
            print(f"Error processing data: {str(e)}")
            return None
    
    def _process_uber_data(self, data):
        """Process Uber dataset format"""
        try:
            # Clean the data first
            processed_data = data.copy()
            
            # Remove rows with invalid coordinates (0.0, 0.0) 
            processed_data = processed_data[
                (processed_data['pickup_longitude'] != 0.0) & 
                (processed_data['pickup_latitude'] != 0.0) &
                (processed_data['dropoff_longitude'] != 0.0) & 
                (processed_data['dropoff_latitude'] != 0.0)
            ]
            
            # Clean fare amounts - remove negative or extremely high fares
            processed_data = processed_data[
                (processed_data['fare_amount'] > 0) & 
                (processed_data['fare_amount'] < 200)
            ]
            
            # Convert pickup_datetime to datetime format
            processed_data['datetime'] = pd.to_datetime(processed_data['pickup_datetime'])
            
            # Extract time features
            processed_data['hour'] = processed_data['datetime'].dt.hour
            processed_data['day_of_week'] = processed_data['datetime'].dt.dayofweek
            processed_data['month'] = processed_data['datetime'].dt.month
            processed_data['is_weekend'] = processed_data['day_of_week'].isin([5, 6]).astype(int)
            
            # Set price from fare_amount
            processed_data['price'] = processed_data['fare_amount']
            
            # Add service and location based on coordinates
            processed_data['service'] = 'Uber'  # This is Uber data
            
            # Create location categories based on pickup coordinates
            processed_data['location'] = processed_data.apply(self._categorize_location, axis=1)
            
            # Keep coordinate columns for distance analysis
            final_columns = ['datetime', 'service', 'location', 'price', 'hour', 'day_of_week', 'month', 'is_weekend',
                           'pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']
            processed_data = processed_data[final_columns]
            
            # Add distance features
            processed_data = self.distance_optimizer.add_distance_features(processed_data)
            
            # Add surge pricing analysis
            processed_data = self.surge_analyzer.calculate_surge_multiplier(processed_data)
            
            # Remove any remaining null values
            processed_data = processed_data.dropna()
            
            return processed_data
            
        except Exception as e:
            print(f"Error processing Uber data: {str(e)}")
            return None
    
    def _categorize_location(self, row):
        """Categorize location based on pickup coordinates (NYC specific)"""
        lat = row['pickup_latitude']
        lon = row['pickup_longitude']
        
        # NYC coordinate ranges (approximate)
        # Manhattan: 40.7-40.8, -74.02 to -73.93
        # Brooklyn: 40.57-40.74, -74.05 to -73.83
        # Queens: 40.54-40.8, -73.96 to -73.7
        # Bronx: 40.79-40.92, -73.93 to -73.76
        # Staten Island: 40.49-40.65, -74.26 to -74.05
        
        if 40.7 <= lat <= 40.8 and -74.02 <= lon <= -73.93:
            if lat >= 40.75:
                return "Upper Manhattan"
            elif lat >= 40.73:
                return "Midtown Manhattan"
            else:
                return "Lower Manhattan"
        elif 40.57 <= lat <= 40.74 and -74.05 <= lon <= -73.83:
            return "Brooklyn"
        elif 40.54 <= lat <= 40.8 and -73.96 <= lon <= -73.7:
            return "Queens"
        elif 40.79 <= lat <= 40.92 and -73.93 <= lon <= -73.76:
            return "Bronx"
        elif 40.49 <= lat <= 40.65 and -74.26 <= lon <= -74.05:
            return "Staten Island"
        else:
            return "NYC Area"
    
    def get_available_services(self):
        """Get list of available ride services"""
        if self.data is not None and 'service' in self.data.columns:
            return sorted(self.data['service'].unique())
        return self.services
    
    def get_available_locations(self):
        """Get list of available locations"""
        if self.data is not None and 'location' in self.data.columns:
            return sorted(self.data['location'].unique())
        return self.locations
    
    def get_pricing_insights(self):
        """Generate pricing insights from the data"""
        if self.data is None:
            return ["No data available for insights generation."]
        
        try:
            insights = []
            
            # Peak hour insight
            hourly_avg = self.data.groupby('hour')['price'].mean()
            peak_hour = hourly_avg.idxmax()
            lowest_hour = hourly_avg.idxmin()
            
            insights.append(f"🔥 Prices are typically highest around {peak_hour}:00")
            insights.append(f"💰 Best deals are usually found around {lowest_hour}:00")
            
            # Weekend vs weekday insight
            weekend_avg = self.data[self.data['is_weekend'] == 1]['price'].mean()
            weekday_avg = self.data[self.data['is_weekend'] == 0]['price'].mean()
            
            if weekend_avg > weekday_avg:
                insights.append(f"📅 Weekend prices are {((weekend_avg/weekday_avg - 1) * 100):.1f}% higher than weekdays")
            else:
                insights.append(f"📅 Weekday prices are {((weekday_avg/weekend_avg - 1) * 100):.1f}% higher than weekends")
            
            # Service comparison insight
            service_avg = self.data.groupby('service')['price'].mean()
            cheapest_service = service_avg.idxmin()
            most_expensive_service = service_avg.idxmax()
            
            insights.append(f"🚗 {cheapest_service} tends to be the most affordable option")
            insights.append(f"💎 {most_expensive_service} typically has premium pricing")
            
            return insights
            
        except Exception as e:
            return [f"Error generating insights: {str(e)}"]
    
    def analyze_peak_hours(self):
        """Analyze peak pricing hours and patterns"""
        if self.data is None:
            return None
        
        try:
            # Hourly analysis
            hourly_stats = self.data.groupby('hour')['price'].agg(['mean', 'count'])
            peak_hour = hourly_stats['mean'].idxmax()
            lowest_hour = hourly_stats['mean'].idxmin()
            
            # Daily analysis
            daily_stats = self.data.groupby('day_of_week')['price'].mean()
            peak_day_num = daily_stats.idxmax()
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            peak_day = day_names[peak_day_num]
            
            # Peak multiplier
            avg_price = self.data['price'].mean()
            peak_price = hourly_stats['mean'].max()
            peak_multiplier = peak_price / avg_price
            
            return {
                'peak_hour': f"{peak_hour}:00",
                'lowest_hour': f"{lowest_hour}:00",
                'peak_day': peak_day,
                'peak_multiplier': peak_multiplier
            }
            
        except Exception as e:
            print(f"Error analyzing peak hours: {str(e)}")
            return None
