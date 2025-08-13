import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

class LocationRouter:
    """Location-based route pricing optimizer for NYC areas"""
    
    def __init__(self):
        # NYC location coordinates (central points for each area)
        self.locations = {
            'Manhattan - Midtown': {'lat': 40.7589, 'lon': -73.9851},
            'Manhattan - Upper East Side': {'lat': 40.7736, 'lon': -73.9566},
            'Manhattan - Upper West Side': {'lat': 40.7873, 'lon': -73.9755},
            'Manhattan - Lower Manhattan': {'lat': 40.7074, 'lon': -74.0113},
            'Manhattan - Financial District': {'lat': 40.7074, 'lon': -74.0113},
            'Brooklyn - Williamsburg': {'lat': 40.7081, 'lon': -73.9571},
            'Brooklyn - Park Slope': {'lat': 40.6782, 'lon': -73.9442},
            'Brooklyn - DUMBO': {'lat': 40.7033, 'lon': -73.9888},
            'Queens - Long Island City': {'lat': 40.7505, 'lon': -73.9534},
            'Queens - Astoria': {'lat': 40.7698, 'lon': -73.9442},
            'Queens - Flushing': {'lat': 40.7675, 'lon': -73.8333},
            'Bronx - South Bronx': {'lat': 40.8176, 'lon': -73.9182},
            'Bronx - Bronx Zoo Area': {'lat': 40.8502, 'lon': -73.8772},
            'LaGuardia Airport': {'lat': 40.7769, 'lon': -73.8740},
            'JFK Airport': {'lat': 40.6413, 'lon': -73.7781},
            'Newark Airport': {'lat': 40.6895, 'lon': -74.1745}
        }
        
        self.earth_radius_km = 6371
        
        # Popular routes with typical characteristics
        self.popular_routes = {
            'Airport Routes': [
                ('Manhattan - Midtown', 'JFK Airport'),
                ('Manhattan - Midtown', 'LaGuardia Airport'),
                ('Brooklyn - DUMBO', 'JFK Airport'),
                ('Queens - Long Island City', 'LaGuardia Airport')
            ],
            'Cross-Borough Commutes': [
                ('Manhattan - Midtown', 'Brooklyn - Williamsburg'),
                ('Manhattan - Upper East Side', 'Queens - Astoria'),
                ('Brooklyn - Park Slope', 'Manhattan - Financial District'),
                ('Queens - Long Island City', 'Manhattan - Midtown')
            ],
            'Inner-Manhattan': [
                ('Manhattan - Midtown', 'Manhattan - Upper East Side'),
                ('Manhattan - Financial District', 'Manhattan - Midtown'),
                ('Manhattan - Upper West Side', 'Manhattan - Midtown')
            ]
        }
    
    def calculate_route_distance(self, origin, destination):
        """Calculate distance between two NYC locations"""
        if origin not in self.locations or destination not in self.locations:
            return None
        
        origin_coords = self.locations[origin]
        dest_coords = self.locations[destination]
        
        # Haversine formula
        lat1, lon1 = np.radians([origin_coords['lat'], origin_coords['lon']])
        lat2, lon2 = np.radians([dest_coords['lat'], dest_coords['lon']])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return self.earth_radius_km * c
    
    def get_route_category(self, origin, destination):
        """Categorize route type"""
        route = (origin, destination)
        
        for category, routes in self.popular_routes.items():
            if route in routes or (destination, origin) in routes:
                return category
        
        # Determine category based on location names
        if 'Airport' in origin or 'Airport' in destination:
            return 'Airport Routes'
        elif origin.split(' - ')[0] != destination.split(' - ')[0]:
            return 'Cross-Borough Commutes'
        else:
            return 'Inner-Borough'
    
    def predict_route_price(self, origin, destination, hour, day_of_week, data):
        """Predict price for a specific route"""
        distance = self.calculate_route_distance(origin, destination)
        if distance is None:
            return None
        
        # Map location names to the location categories in the data
        origin_mapped = self._map_to_data_location(origin)
        dest_mapped = self._map_to_data_location(destination)
        
        # Find similar trips in the historical data
        similar_trips = data[
            (data['hour'] == hour) &
            (data['day_of_week'] == day_of_week)
        ]
        
        if similar_trips.empty:
            similar_trips = data[data['hour'] == hour]
        
        if similar_trips.empty:
            # Fallback to overall average with distance adjustment
            base_price_per_km = data['price'].mean() / data['trip_distance_km'].mean() if 'trip_distance_km' in data.columns else 2.5
            return base_price_per_km * distance
        
        # Calculate price based on distance and historical patterns
        if 'trip_distance_km' in data.columns:
            distance_factor = distance / data['trip_distance_km'].mean()
            base_price = similar_trips['price'].median()
            estimated_price = base_price * distance_factor
        else:
            # Fallback calculation
            avg_price_per_km = similar_trips['price'].median() / 5  # Assume 5km average
            estimated_price = avg_price_per_km * distance
        
        # Apply route category multipliers
        category = self.get_route_category(origin, destination)
        multipliers = {
            'Airport Routes': 1.3,
            'Cross-Borough Commutes': 1.1,
            'Inner-Manhattan': 1.2,
            'Inner-Borough': 1.0
        }
        
        final_price = estimated_price * multipliers.get(category, 1.0)
        
        return {
            'estimated_price': final_price,
            'base_price': estimated_price,
            'distance_km': distance,
            'route_category': category,
            'price_per_km': final_price / distance,
            'similar_trips': len(similar_trips)
        }
    
    def _map_to_data_location(self, location_name):
        """Map friendly location names to data location categories"""
        if 'Manhattan' in location_name:
            if 'Upper' in location_name:
                return 'Upper Manhattan'
            elif 'Lower' in location_name or 'Financial' in location_name:
                return 'Lower Manhattan'
            else:
                return 'Midtown Manhattan'
        elif 'Brooklyn' in location_name:
            return 'Brooklyn'
        elif 'Queens' in location_name:
            return 'Queens'
        elif 'Bronx' in location_name:
            return 'Bronx'
        elif 'Airport' in location_name:
            return 'NYC Area'
        else:
            return 'NYC Area'
    
    def analyze_route_patterns(self, data):
        """Analyze pricing patterns for different route types"""
        if data is None or data.empty:
            return None
        
        try:
            route_analysis = {}
            
            # Get base pricing data
            base_price_per_km = data['price'].mean() / 5.0  # Assume 5km average trip
            
            # Analyze by time of day for popular routes
            for category, routes in self.popular_routes.items():
                category_data = []
                
                for origin, destination in routes[:2]:  # Limit to 2 routes per category for performance
                    distance = self.calculate_route_distance(origin, destination)
                    if distance and distance > 0:
                        # Create hourly data for this route
                        for hour in [6, 9, 12, 15, 18, 21]:  # Sample key hours only
                            # Get pricing factor for this hour
                            hour_data = data[data['hour'] == hour]
                            if not hour_data.empty:
                                hour_factor = hour_data['price'].mean() / data['price'].mean()
                            else:
                                hour_factor = 1.0
                            
                            # Apply category multipliers
                            if category == 'Airport Routes':
                                multiplier = 1.3
                            elif category == 'Cross-Borough Commutes':
                                multiplier = 1.1
                            else:
                                multiplier = 1.0
                            
                            estimated_price = base_price_per_km * distance * multiplier * hour_factor
                            
                            category_data.append({
                                'hour': hour,
                                'route': f"{origin} → {destination}",
                                'estimated_price': estimated_price,
                                'distance': distance
                            })
                
                if category_data:
                    route_analysis[category] = pd.DataFrame(category_data)
            
            return route_analysis
            
        except Exception as e:
            print(f"Error in route pattern analysis: {str(e)}")
            return None
    
    def create_route_comparison_chart(self, route_analysis):
        """Create route category comparison chart"""
        if not route_analysis:
            return None
        
        try:
            # Create a simple single chart instead of complex subplots
            fig = go.Figure()
            
            colors = ['blue', 'red', 'green', 'orange']
            
            for i, (category, data) in enumerate(route_analysis.items()):
                if not data.empty and 'hour' in data.columns and 'estimated_price' in data.columns:
                    hourly_avg = data.groupby('hour')['estimated_price'].mean()
                    
                    fig.add_trace(
                        go.Scatter(
                            x=hourly_avg.index,
                            y=hourly_avg.values,
                            mode='lines+markers',
                            name=category,
                            line=dict(color=colors[i % len(colors)], width=3),
                            marker=dict(size=8)
                        )
                    )
            
            fig.update_layout(
                title="Route Category Pricing Patterns by Hour",
                xaxis_title="Hour of Day",
                yaxis_title="Average Price ($)",
                height=500,
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating route comparison chart: {str(e)}")
            return None
    
    def create_route_network_map(self):
        """Create an interactive map showing popular routes"""
        try:
            fig = go.Figure()
            
            # Add location points
            for location, coords in self.locations.items():
                fig.add_trace(go.Scattermapbox(
                    lat=[coords['lat']],
                    lon=[coords['lon']],
                    mode='markers+text',
                    marker=dict(size=10, color='blue'),
                    text=location.split(' - ')[-1],  # Show only the area name
                    textposition='top center',
                    name=location,
                    showlegend=False,
                    hovertemplate=f'<b>{location}</b><extra></extra>'
                ))
            
            # Add popular route lines
            colors = ['red', 'green', 'purple']
            for i, (category, routes) in enumerate(self.popular_routes.items()):
                for origin, destination in routes[:3]:  # Show first 3 routes per category
                    origin_coords = self.locations[origin]
                    dest_coords = self.locations[destination]
                    
                    fig.add_trace(go.Scattermapbox(
                        lat=[origin_coords['lat'], dest_coords['lat']],
                        lon=[origin_coords['lon'], dest_coords['lon']],
                        mode='lines',
                        line=dict(width=2, color=colors[i % len(colors)]),
                        name=category,
                        showlegend=(origin == routes[0][0]),  # Show legend only once per category
                        hovertemplate=f'<b>{category}</b><br>{origin} → {destination}<extra></extra>'
                    ))
            
            fig.update_layout(
                mapbox=dict(
                    style="open-street-map",
                    center=dict(lat=40.7580, lon=-73.9855),
                    zoom=10
                ),
                title="NYC Popular Route Network",
                height=600,
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating route network map: {str(e)}")
            return None
    
    def get_route_recommendations(self, origin, destination, hour, data):
        """Get pricing recommendations and insights for a route"""
        prediction = self.predict_route_price(origin, destination, hour, 0, data)
        
        if not prediction:
            return None
        
        recommendations = {
            'route_info': prediction,
            'best_times': [],
            'cost_saving_tips': [],
            'alternative_routes': []
        }
        
        # Find best times to travel (lowest prices)
        best_hours = []
        for h in range(24):
            hour_prediction = self.predict_route_price(origin, destination, h, 0, data)
            if hour_prediction:
                best_hours.append((h, hour_prediction['estimated_price']))
        
        if best_hours:
            best_hours.sort(key=lambda x: x[1])
            recommendations['best_times'] = best_hours[:3]
        
        # Generate cost-saving tips
        if prediction['route_category'] == 'Airport Routes':
            recommendations['cost_saving_tips'] = [
                "Consider taking subway + AirTrain for significant savings",
                "Airport routes are typically 30% more expensive",
                "Early morning (5-7 AM) usually has lower prices"
            ]
        elif prediction['route_category'] == 'Cross-Borough Commutes':
            recommendations['cost_saving_tips'] = [
                "Consider subway for regular commuting",
                "Prices are typically higher during rush hours",
                "Shared rides can reduce costs by 20-30%"
            ]
        else:
            recommendations['cost_saving_tips'] = [
                "Walking might be feasible for short distances",
                "Consider bike-sharing options",
                "Off-peak hours typically offer better rates"
            ]
        
        # Suggest alternative routes (nearby locations)
        alternatives = []
        for loc in self.locations.keys():
            if loc != destination and destination.split(' - ')[0] in loc:
                alt_prediction = self.predict_route_price(origin, loc, hour, 0, data)
                if alt_prediction:
                    alternatives.append((loc, alt_prediction['estimated_price']))
        
        if alternatives:
            alternatives.sort(key=lambda x: x[1])
            recommendations['alternative_routes'] = alternatives[:2]
        
        return recommendations