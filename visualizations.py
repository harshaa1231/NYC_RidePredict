import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots

class Visualizations:
    """Create interactive visualizations for ride-sharing price analysis"""
    
    def __init__(self):
        self.color_palette = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 
            '#9467bd', '#8c564b', '#e377c2', '#7f7f7f'
        ]
    
    def create_hourly_trends(self, data):
        """Create hourly price trends visualization"""
        if data is None or data.empty:
            return None
        
        try:
            # Calculate hourly statistics
            hourly_stats = data.groupby(['hour', 'service'])['price'].agg(['mean', 'std', 'count']).reset_index()
            hourly_stats.columns = ['hour', 'service', 'avg_price', 'price_std', 'count']
            
            fig = go.Figure()
            
            # Add line for each service
            services = hourly_stats['service'].unique()
            for i, service in enumerate(services):
                service_data = hourly_stats[hourly_stats['service'] == service]
                
                fig.add_trace(go.Scatter(
                    x=service_data['hour'],
                    y=service_data['avg_price'],
                    mode='lines+markers',
                    name=service,
                    line=dict(color=self.color_palette[i % len(self.color_palette)], width=3),
                    marker=dict(size=8),
                    hovertemplate=f'<b>{service}</b><br>' +
                                'Hour: %{x}:00<br>' +
                                'Avg Price: $%{y:.2f}<br>' +
                                '<extra></extra>'
                ))
            
            # Update layout
            fig.update_layout(
                title='Average Hourly Price Trends by Service',
                xaxis_title='Hour of Day',
                yaxis_title='Average Price ($)',
                hovermode='x unified',
                template='plotly_white',
                height=500,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            # Customize x-axis
            fig.update_xaxes(
                tickmode='linear',
                tick0=0,
                dtick=2,
                range=[-0.5, 23.5]
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating hourly trends: {str(e)}")
            return None
    
    def create_daily_patterns(self, data):
        """Create daily price patterns visualization"""
        if data is None or data.empty:
            return None
        
        try:
            # Calculate daily statistics
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            daily_stats = data.groupby(['day_of_week', 'service'])['price'].agg(['mean', 'count']).reset_index()
            daily_stats.columns = ['day_of_week', 'service', 'avg_price', 'count']
            daily_stats['day_name'] = daily_stats['day_of_week'].map(dict(enumerate(day_names)))
            
            # Create subplots
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Average Price by Day', 'Ride Count by Day'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Price chart
            services = daily_stats['service'].unique()
            for i, service in enumerate(services):
                service_data = daily_stats[daily_stats['service'] == service]
                
                fig.add_trace(
                    go.Bar(
                        x=service_data['day_name'],
                        y=service_data['avg_price'],
                        name=service,
                        marker_color=self.color_palette[i % len(self.color_palette)],
                        showlegend=True
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Bar(
                        x=service_data['day_name'],
                        y=service_data['count'],
                        name=f"{service} (Count)",
                        marker_color=self.color_palette[i % len(self.color_palette)],
                        opacity=0.7,
                        showlegend=False
                    ),
                    row=1, col=2
                )
            
            # Update layout
            fig.update_layout(
                title_text="Daily Pricing and Demand Patterns",
                height=500,
                template='plotly_white',
                barmode='group'
            )
            
            fig.update_xaxes(title_text="Day of Week", row=1, col=1)
            fig.update_xaxes(title_text="Day of Week", row=1, col=2)
            fig.update_yaxes(title_text="Average Price ($)", row=1, col=1)
            fig.update_yaxes(title_text="Number of Rides", row=1, col=2)
            
            return fig
            
        except Exception as e:
            print(f"Error creating daily patterns: {str(e)}")
            return None
    
    def create_service_comparison(self, data):
        """Create service price comparison visualization"""
        if data is None or data.empty:
            return None
        
        try:
            # Calculate service statistics
            service_stats = data.groupby('service')['price'].agg([
                'mean', 'median', 'std', 'min', 'max', 'count'
            ]).reset_index()
            
            # Create box plot
            fig = go.Figure()
            
            services = data['service'].unique()
            for i, service in enumerate(services):
                service_prices = data[data['service'] == service]['price']
                
                fig.add_trace(go.Box(
                    y=service_prices,
                    name=service,
                    marker_color=self.color_palette[i % len(self.color_palette)],
                    boxpoints='outliers',
                    jitter=0.3,
                    whiskerwidth=0.2,
                    fillcolor=self.color_palette[i % len(self.color_palette)],
                    line_width=2
                ))
            
            # Update layout
            fig.update_layout(
                title='Price Distribution Comparison by Service',
                yaxis_title='Price ($)',
                xaxis_title='Service',
                template='plotly_white',
                height=500,
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating service comparison: {str(e)}")
            return None
    
    def create_peak_hours_heatmap(self, data):
        """Create peak hours heatmap visualization"""
        if data is None or data.empty:
            return None
        
        try:
            # Create hourly-daily price matrix
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            
            # Group by hour and day of week
            heatmap_data = data.groupby(['hour', 'day_of_week'])['price'].mean().reset_index()
            
            # Pivot to create matrix
            price_matrix = heatmap_data.pivot(index='hour', columns='day_of_week', values='price')
            
            # Ensure all hours and days are present
            all_hours = list(range(24))
            all_days = list(range(7))
            price_matrix = price_matrix.reindex(index=all_hours, columns=all_days, fill_value=0)
            
            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=price_matrix.values,
                x=[day_names[i] for i in price_matrix.columns],
                y=price_matrix.index,
                colorscale='Viridis',
                hoverongaps=False,
                hovertemplate='<b>%{x}</b><br>' +
                             'Hour: %{y}:00<br>' +
                             'Avg Price: $%{z:.2f}<br>' +
                             '<extra></extra>'
            ))
            
            # Update layout
            fig.update_layout(
                title='Price Heatmap: Hour vs Day of Week',
                xaxis_title='Day of Week',
                yaxis_title='Hour of Day',
                template='plotly_white',
                height=600
            )
            
            # Customize y-axis
            fig.update_yaxes(
                tickmode='linear',
                tick0=0,
                dtick=2,
                autorange='reversed'
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating peak hours heatmap: {str(e)}")
            return None
    
    def create_location_comparison(self, data):
        """Create location-based price comparison"""
        if data is None or data.empty or 'location' not in data.columns:
            return None
        
        try:
            # Calculate location statistics
            location_stats = data.groupby('location')['price'].agg([
                'mean', 'count'
            ]).reset_index()
            
            # Create scatter plot
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=location_stats['count'],
                y=location_stats['mean'],
                mode='markers+text',
                text=location_stats['location'],
                textposition='top center',
                marker=dict(
                    size=location_stats['count'],
                    sizemode='diameter',
                    sizeref=2.*max(location_stats['count'])/(40.**2),
                    sizemin=4,
                    color=location_stats['mean'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Avg Price ($)")
                )
            ))
            
            # Update layout
            fig.update_layout(
                title='Location Analysis: Price vs Demand',
                xaxis_title='Number of Rides',
                yaxis_title='Average Price ($)',
                template='plotly_white',
                height=500,
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating location comparison: {str(e)}")
            return None
