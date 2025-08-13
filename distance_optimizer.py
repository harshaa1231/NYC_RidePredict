import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

class DistanceOptimizer:
    """Analyze distance patterns and optimize route-based pricing"""
    
    def __init__(self):
        self.earth_radius_km = 6371  # Earth's radius in kilometers
    
    def calculate_haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calculate the great circle distance between two points on Earth"""
        # Convert latitude and longitude from degrees to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        # Distance in kilometers
        distance = self.earth_radius_km * c
        return distance
    
    def add_distance_features(self, data):
        """Add distance-related features to the dataset"""
        if data is None or data.empty:
            return data
        
        # Check if we have coordinate data
        required_cols = ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']
        if not all(col in data.columns for col in required_cols):
            return data
        
        data_with_distance = data.copy()
        
        # Calculate trip distance
        data_with_distance['trip_distance_km'] = self.calculate_haversine_distance(
            data_with_distance['pickup_latitude'],
            data_with_distance['pickup_longitude'],
            data_with_distance['dropoff_latitude'],
            data_with_distance['dropoff_longitude']
        )
        
        # Calculate price per kilometer
        data_with_distance['price_per_km'] = data_with_distance['price'] / (data_with_distance['trip_distance_km'] + 0.1)  # Avoid division by zero
        
        # Categorize trip types by distance
        data_with_distance['trip_type'] = pd.cut(
            data_with_distance['trip_distance_km'],
            bins=[0, 2, 5, 10, 20, float('inf')],
            labels=['Short (<2km)', 'Medium (2-5km)', 'Long (5-10km)', 'Very Long (10-20km)', 'Airport/Inter-city (>20km)'],
            include_lowest=True
        )
        
        # Calculate distance from city center (using Times Square as reference: 40.7580, -73.9855)
        times_square_lat, times_square_lon = 40.7580, -73.9855
        data_with_distance['distance_from_center'] = self.calculate_haversine_distance(
            data_with_distance['pickup_latitude'],
            data_with_distance['pickup_longitude'],
            times_square_lat,
            times_square_lon
        )
        
        return data_with_distance
    
    def analyze_distance_pricing(self, data):
        """Analyze pricing patterns by distance"""
        if data is None or data.empty or 'trip_distance_km' not in data.columns:
            return None
        
        # Distance-based analysis
        distance_analysis = data.groupby('trip_type').agg({
            'price': ['mean', 'median', 'count'],
            'price_per_km': ['mean', 'median'],
            'trip_distance_km': 'mean'
        }).round(2)
        
        distance_analysis.columns = ['avg_price', 'median_price', 'ride_count', 'avg_price_per_km', 'median_price_per_km', 'avg_distance']
        
        # Time-distance analysis
        time_distance = data.groupby(['hour', 'trip_type']).agg({
            'price': 'mean',
            'price_per_km': 'mean'
        }).reset_index()
        
        return {
            'distance_summary': distance_analysis,
            'time_distance_patterns': time_distance
        }
    
    def optimize_route_pricing(self, pickup_lat, pickup_lon, dropoff_lat, dropoff_lon, hour, day_of_week, data):
        """Suggest optimal pricing for a specific route"""
        if data is None or data.empty or 'trip_distance_km' not in data.columns:
            return None
        
        # Calculate trip distance
        trip_distance = self.calculate_haversine_distance(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)
        
        # Find similar trips
        similar_trips = data[
            (abs(data['trip_distance_km'] - trip_distance) <= 2) &  # Within 2km distance
            (data['hour'] == hour)
        ]
        
        if similar_trips.empty:
            # Fallback to distance-based pricing
            similar_trips = data[abs(data['trip_distance_km'] - trip_distance) <= 5]
        
        if similar_trips.empty:
            return None
        
        # Calculate pricing recommendations
        base_price = similar_trips['price'].median()
        price_range = {
            'min': similar_trips['price'].quantile(0.25),
            'median': base_price,
            'max': similar_trips['price'].quantile(0.75)
        }
        
        price_per_km = similar_trips['price_per_km'].median()
        estimated_price = price_per_km * trip_distance
        
        return {
            'trip_distance_km': trip_distance,
            'historical_price_range': price_range,
            'estimated_price': estimated_price,
            'price_per_km': price_per_km,
            'similar_trips_count': len(similar_trips)
        }
    
    def create_distance_analysis_chart(self, data):
        """Create distance-based pricing analysis chart"""
        if data is None or data.empty or 'trip_distance_km' not in data.columns:
            return None
        
        try:
            # Create subplot
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=['Price vs Distance', 'Price per KM Distribution', 
                               'Trip Type Distribution', 'Distance from City Center'],
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # 1. Price vs Distance scatter plot
            sample_data = data.sample(n=min(1000, len(data)), random_state=42)  # Sample for performance
            fig.add_trace(
                go.Scatter(
                    x=sample_data['trip_distance_km'],
                    y=sample_data['price'],
                    mode='markers',
                    name='Price vs Distance',
                    marker=dict(size=4, opacity=0.6),
                    showlegend=False
                ),
                row=1, col=1
            )
            
            # 2. Price per KM distribution
            fig.add_trace(
                go.Histogram(
                    x=data['price_per_km'].clip(0, 20),  # Cap extreme values
                    nbinsx=30,
                    name='Price per KM',
                    showlegend=False
                ),
                row=1, col=2
            )
            
            # 3. Trip type distribution
            trip_type_counts = data['trip_type'].value_counts()
            fig.add_trace(
                go.Bar(
                    x=trip_type_counts.index,
                    y=trip_type_counts.values,
                    name='Trip Types',
                    showlegend=False
                ),
                row=2, col=1
            )
            
            # 4. Distance from center analysis
            center_distance_price = data.groupby(pd.cut(data['distance_from_center'], bins=10))['price'].mean()
            fig.add_trace(
                go.Scatter(
                    x=[interval.mid for interval in center_distance_price.index],
                    y=center_distance_price.values,
                    mode='lines+markers',
                    name='Price by Distance from Center',
                    showlegend=False
                ),
                row=2, col=2
            )
            
            # Update layout
            fig.update_xaxes(title_text="Distance (km)", row=1, col=1)
            fig.update_yaxes(title_text="Price ($)", row=1, col=1)
            
            fig.update_xaxes(title_text="Price per KM ($)", row=1, col=2)
            fig.update_yaxes(title_text="Frequency", row=1, col=2)
            
            fig.update_xaxes(title_text="Trip Type", row=2, col=1)
            fig.update_yaxes(title_text="Count", row=2, col=1)
            
            fig.update_xaxes(title_text="Distance from Center (km)", row=2, col=2)
            fig.update_yaxes(title_text="Avg Price ($)", row=2, col=2)
            
            fig.update_layout(
                title_text="Distance-Based Pricing Analysis",
                height=800,
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating distance analysis chart: {str(e)}")
            return None
    
    def create_route_heatmap(self, data):
        """Create a heatmap of popular routes"""
        if data is None or data.empty:
            return None
        
        try:
            # Sample data for performance
            sample_data = data.sample(n=min(2000, len(data)), random_state=42)
            
            fig = go.Figure()
            
            # Add pickup points
            fig.add_trace(go.Scattermapbox(
                lat=sample_data['pickup_latitude'],
                lon=sample_data['pickup_longitude'],
                mode='markers',
                marker=dict(size=8, color='green', opacity=0.6),
                name='Pickup Points',
                text=sample_data['price'].apply(lambda x: f'${x:.2f}'),
                hovertemplate='Pickup<br>Price: %{text}<extra></extra>'
            ))
            
            # Add dropoff points
            fig.add_trace(go.Scattermapbox(
                lat=sample_data['dropoff_latitude'],
                lon=sample_data['dropoff_longitude'],
                mode='markers',
                marker=dict(size=8, color='red', opacity=0.6),
                name='Dropoff Points',
                text=sample_data['price'].apply(lambda x: f'${x:.2f}'),
                hovertemplate='Dropoff<br>Price: %{text}<extra></extra>'
            ))
            
            # Center map on NYC
            fig.update_layout(
                mapbox=dict(
                    style="open-street-map",
                    center=dict(lat=40.7580, lon=-73.9855),
                    zoom=10
                ),
                title="NYC Ride Pickup and Dropoff Locations",
                height=600
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating route heatmap: {str(e)}")
            return None
    
    def get_distance_insights(self, data):
        """Generate insights about distance and pricing patterns"""
        if data is None or data.empty:
            return ["No data available for distance analysis."]
        
        insights = []
        
        try:
            if 'trip_distance_km' in data.columns:
                # Average trip distance
                avg_distance = data['trip_distance_km'].mean()
                insights.append(f" Average trip distance: {avg_distance:.1f} km")
                
                # Price per kilometer analysis
                avg_price_per_km = data['price_per_km'].median()
                insights.append(f" Median rate: ${avg_price_per_km:.2f} per kilometer")
                
                # Most common trip type
                most_common_trip = data['trip_type'].mode().iloc[0]
                insights.append(f" Most common trips: {most_common_trip}")
                
                # Long distance premium
                short_trips = data[data['trip_distance_km'] < 2]['price_per_km'].median()
                long_trips = data[data['trip_distance_km'] > 10]['price_per_km'].median()
                
                if long_trips < short_trips:
                    discount = ((short_trips - long_trips) / short_trips) * 100
                    insights.append(f" Long trips get {discount:.1f}% better rate per km")
                
                # Distance from center impact
                if 'distance_from_center' in data.columns:
                    center_correlation = data['distance_from_center'].corr(data['price'])
                    if abs(center_correlation) > 0.1:
                        direction = "higher" if center_correlation > 0 else "lower"
                        insights.append(f" Trips farther from center tend to have {direction} prices")
            
            return insights
            
        except Exception as e:
            return [f"Error generating distance insights: {str(e)}"]
