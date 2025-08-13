import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

class SurgePricingAnalyzer:
    """Analyze surge pricing patterns and demand fluctuations"""
    
    def __init__(self):
        self.surge_thresholds = {
            'no_surge': 1.0,
            'low_surge': 1.2,
            'medium_surge': 1.5,
            'high_surge': 2.0,
            'extreme_surge': 3.0
        }
    
    def calculate_surge_multiplier(self, data):
        """Calculate surge pricing multipliers based on demand patterns"""
        if data is None or data.empty:
            return data
        
        # Calculate baseline price for each location
        baseline_prices = data.groupby('location')['price'].quantile(0.25)  # 25th percentile as baseline
        
        # Calculate surge multiplier for each ride
        data_with_surge = data.copy()
        data_with_surge['baseline_price'] = data_with_surge['location'].map(baseline_prices)
        data_with_surge['surge_multiplier'] = data_with_surge['price'] / data_with_surge['baseline_price']
        
        # Cap extreme values
        data_with_surge['surge_multiplier'] = data_with_surge['surge_multiplier'].clip(0.5, 5.0)
        
        # Categorize surge levels
        data_with_surge['surge_level'] = pd.cut(
            data_with_surge['surge_multiplier'],
            bins=[0, 1.2, 1.5, 2.0, 3.0, float('inf')],
            labels=['No Surge', 'Low Surge', 'Medium Surge', 'High Surge', 'Extreme Surge'],
            include_lowest=True
        )
        
        return data_with_surge
    
    def analyze_demand_patterns(self, data):
        """Analyze demand patterns that drive surge pricing"""
        if data is None or data.empty:
            return None
        
        # Calculate rides per hour as demand proxy
        demand_analysis = data.groupby(['hour', 'location']).agg({
            'price': ['count', 'mean'],
            'surge_multiplier': 'mean' if 'surge_multiplier' in data.columns else 'first'
        }).reset_index()
        
        demand_analysis.columns = ['hour', 'location', 'ride_count', 'avg_price', 'avg_surge']
        
        # Find peak demand hours
        hourly_demand = data.groupby('hour').size()
        peak_hours = hourly_demand.nlargest(3).index.tolist()
        
        # Find surge hotspots
        if 'surge_multiplier' in data.columns:
            location_surge = data.groupby('location')['surge_multiplier'].mean().sort_values(ascending=False)
            surge_hotspots = location_surge.head(3).index.tolist()
        else:
            surge_hotspots = []
        
        return {
            'demand_by_hour_location': demand_analysis,
            'peak_hours': peak_hours,
            'surge_hotspots': surge_hotspots,
            'total_rides': len(data)
        }
    
    def predict_surge_probability(self, data, hour, location, day_of_week):
        """Predict probability of surge pricing for given conditions"""
        if data is None or data.empty or 'surge_multiplier' not in data.columns:
            return 0.5  # Default probability
        
        # Filter data for similar conditions
        similar_conditions = data[
            (data['hour'] == hour) & 
            (data['location'] == location) & 
            (data['day_of_week'] == day_of_week)
        ]
        
        if similar_conditions.empty:
            # Fallback to hour-based probability
            similar_conditions = data[data['hour'] == hour]
        
        if similar_conditions.empty:
            return 0.5
        
        # Calculate surge probability (surge_multiplier > 1.2)
        surge_rides = similar_conditions[similar_conditions['surge_multiplier'] > 1.2]
        surge_probability = len(surge_rides) / len(similar_conditions)
        
        return surge_probability
    
    def create_surge_heatmap(self, data):
        """Create surge pricing heatmap visualization"""
        if data is None or data.empty or 'surge_multiplier' not in data.columns:
            return None
        
        try:
            # Create hourly-location surge matrix
            surge_matrix = data.groupby(['hour', 'location'])['surge_multiplier'].mean().reset_index()
            surge_pivot = surge_matrix.pivot(index='hour', columns='location', values='surge_multiplier')
            
            # Fill missing values with 1.0 (no surge)
            surge_pivot = surge_pivot.fillna(1.0)
            
            fig = go.Figure(data=go.Heatmap(
                z=surge_pivot.values,
                x=surge_pivot.columns,
                y=surge_pivot.index,
                colorscale='Reds',
                colorbar=dict(title="Surge Multiplier"),
                hoverongaps=False,
                hovertemplate='<b>%{x}</b><br>' +
                             'Hour: %{y}:00<br>' +
                             'Avg Surge: %{z:.2f}x<br>' +
                             '<extra></extra>'
            ))
            
            fig.update_layout(
                title='Surge Pricing Heatmap by Hour and Location',
                xaxis_title='Location',
                yaxis_title='Hour of Day',
                height=600
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating surge heatmap: {str(e)}")
            return None
    
    def create_demand_supply_chart(self, data):
        """Create demand vs supply analysis chart"""
        if data is None or data.empty:
            return None
        
        try:
            # Calculate hourly metrics
            hourly_stats = data.groupby('hour').agg({
                'price': ['count', 'mean'],
                'surge_multiplier': 'mean' if 'surge_multiplier' in data.columns else 'first'
            }).reset_index()
            
            hourly_stats.columns = ['hour', 'ride_count', 'avg_price', 'avg_surge']
            
            # Create subplot with secondary y-axis
            fig = make_subplots(
                rows=1, cols=1,
                specs=[[{"secondary_y": True}]],
                subplot_titles=['Demand vs Pricing Patterns']
            )
            
            # Add ride count (demand proxy)
            fig.add_trace(
                go.Scatter(
                    x=hourly_stats['hour'],
                    y=hourly_stats['ride_count'],
                    name='Ride Count (Demand)',
                    line=dict(color='blue', width=3),
                    mode='lines+markers'
                ),
                secondary_y=False
            )
            
            # Add average price
            fig.add_trace(
                go.Scatter(
                    x=hourly_stats['hour'],
                    y=hourly_stats['avg_price'],
                    name='Average Price',
                    line=dict(color='red', width=3),
                    mode='lines+markers'
                ),
                secondary_y=True
            )
            
            # Add surge multiplier if available
            if 'surge_multiplier' in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=hourly_stats['hour'],
                        y=hourly_stats['avg_surge'],
                        name='Surge Multiplier',
                        line=dict(color='orange', width=3, dash='dash'),
                        mode='lines+markers'
                    ),
                    secondary_y=True
                )
            
            # Update axes
            fig.update_xaxes(title_text="Hour of Day")
            fig.update_yaxes(title_text="Number of Rides", secondary_y=False)
            fig.update_yaxes(title_text="Price ($) / Surge Multiplier", secondary_y=True)
            
            fig.update_layout(
                title='Demand vs Supply Analysis',
                height=500,
                hovermode='x unified'
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating demand-supply chart: {str(e)}")
            return None
    
    def get_surge_insights(self, data):
        """Generate insights about surge pricing patterns"""
        if data is None or data.empty:
            return ["No data available for surge analysis."]
        
        insights = []
        
        try:
            if 'surge_multiplier' in data.columns:
                # Surge frequency
                surge_rides = data[data['surge_multiplier'] > 1.2]
                surge_percentage = (len(surge_rides) / len(data)) * 100
                insights.append(f"🔥 Surge pricing occurs {surge_percentage:.1f}% of the time")
                
                # Peak surge hours
                hourly_surge = data.groupby('hour')['surge_multiplier'].mean()
                peak_surge_hour = hourly_surge.idxmax()
                peak_surge_value = hourly_surge.max()
                insights.append(f"📈 Highest surge at {peak_surge_hour}:00 (avg {peak_surge_value:.1f}x)")
                
                # Surge by location
                location_surge = data.groupby('location')['surge_multiplier'].mean()
                highest_surge_location = location_surge.idxmax()
                insights.append(f"🏙️ {highest_surge_location} has highest average surge pricing")
                
                # Weekend vs weekday surge
                weekend_surge = data[data['is_weekend'] == 1]['surge_multiplier'].mean()
                weekday_surge = data[data['is_weekend'] == 0]['surge_multiplier'].mean()
                
                if weekend_surge > weekday_surge:
                    insights.append(f"📅 Weekend surge is {((weekend_surge/weekday_surge - 1) * 100):.1f}% higher")
                else:
                    insights.append(f"📅 Weekday surge is {((weekday_surge/weekend_surge - 1) * 100):.1f}% higher")
            
            # Demand patterns
            hourly_demand = data.groupby('hour').size()
            peak_demand_hour = hourly_demand.idxmax()
            insights.append(f"🚗 Peak demand at {peak_demand_hour}:00 ({hourly_demand.max()} rides)")
            
            return insights
            
        except Exception as e:
            return [f"Error generating surge insights: {str(e)}"]