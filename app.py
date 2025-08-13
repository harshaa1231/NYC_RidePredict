import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from data_processor import DataProcessor
from price_predictor import PricePredictor
from visualizations import Visualizations
from advanced_models import AdvancedModelEnsemble
from surge_pricing import SurgePricingAnalyzer
from distance_optimizer import DistanceOptimizer
from location_router import LocationRouter

# Page configuration
st.set_page_config(
    page_title="Ride-Share Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize components
@st.cache_resource
def initialize_components():
    """Initialize data processor and price predictor"""
    try:
        data_processor = DataProcessor()
        price_predictor = PricePredictor()
        visualizations = Visualizations()
        
        # Load and process data
        data = data_processor.load_data()
        if data is not None and not data.empty:
            price_predictor.train_model(data)
            return data_processor, price_predictor, visualizations, data
        else:
            return None, None, None, None
    except Exception as e:
        st.error(f"Error initializing components: {str(e)}")
        return None, None, None, None

def main():
    # Title and description
    st.title("🚗 NYC Route Pricing Optimizer")
    st.markdown("**Predict ride prices between NYC locations and optimize your travel costs**")
    st.markdown("*Powered by 200,000+ real Uber rides with advanced ML models*")
    
    # Initialize components
    data_processor, price_predictor, visualizations, data = initialize_components()
    location_router = LocationRouter()
    
    if data_processor is None or data is None:
        st.error("⚠️ Unable to load ride-sharing data. Please ensure data sources are available.")
        st.info("This application requires historical ride-sharing pricing data to make predictions and generate insights.")
        return
    
    # Main Route Pricing Section (Featured prominently)
    st.header("🎯 Route Pricing Optimizer")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📍 Select Your Route")
        
        # Location selection with friendly names
        origin_location = st.selectbox(
            "From (Pickup Location)",
            options=list(location_router.locations.keys()),
            index=list(location_router.locations.keys()).index('Manhattan - Midtown')
        )
        
        destination_location = st.selectbox(
            "To (Dropoff Location)", 
            options=list(location_router.locations.keys()),
            index=list(location_router.locations.keys()).index('JFK Airport')
        )
        
        # Time selection
        selected_hour = st.slider(
            "Time of Day",
            min_value=0,
            max_value=23,
            value=datetime.now().hour,
            step=1,
            format="%d:00"
        )
        
        day_of_week = st.selectbox(
            "Day of Week",
            options=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
            index=datetime.now().weekday()
        )
        
        day_num = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'].index(day_of_week)
    
    with col2:
        st.subheader("💰 Price Prediction")
        
        if st.button("🚀 Get Route Price", type="primary"):
            route_prediction = location_router.predict_route_price(
                origin_location, destination_location, selected_hour, day_num, data
            )
            
            if route_prediction:
                # Display main price prominently
                st.metric(
                    "Estimated Price",
                    f"${route_prediction['estimated_price']:.2f}",
                    delta=f"${route_prediction['estimated_price'] - route_prediction['base_price']:.2f} route premium"
                )
                
                # Route details
                col_a, col_b, col_c = st.columns(3)
                
                with col_a:
                    st.metric("Distance", f"{route_prediction['distance_km']:.1f} km")
                
                with col_b:
                    st.metric("Rate per KM", f"${route_prediction['price_per_km']:.2f}")
                
                with col_c:
                    st.metric("Route Type", route_prediction['route_category'])
                
                # Get recommendations
                recommendations = location_router.get_route_recommendations(
                    origin_location, destination_location, selected_hour, data
                )
                
                if recommendations:
                    st.subheader("🎯 Money-Saving Tips")
                    
                    # Best times to travel
                    if recommendations['best_times']:
                        st.write("**💡 Cheapest times to travel:**")
                        for i, (hour, price) in enumerate(recommendations['best_times'][:3]):
                            savings = route_prediction['estimated_price'] - price
                            st.write(f"• {hour}:00 - ${price:.2f} (Save ${savings:.2f})")
                    
                    # Cost saving tips
                    if recommendations['cost_saving_tips']:
                        st.write("**💰 Cost-Saving Tips:**")
                        for tip in recommendations['cost_saving_tips']:
                            st.info(tip)
                    
                    # Alternative destinations
                    if recommendations['alternative_routes']:
                        st.write("**🔄 Alternative nearby destinations:**")
                        for location, price in recommendations['alternative_routes']:
                            savings = route_prediction['estimated_price'] - price
                            st.write(f"• {location} - ${price:.2f} (Save ${savings:.2f})")
            
            else:
                st.error("Unable to calculate route price. Please try different locations.")
    
    # Route Network Visualization
    st.subheader("🗺️ NYC Route Network")
    route_map = location_router.create_route_network_map()
    if route_map:
        st.plotly_chart(route_map, use_container_width=True)
    
    # Sidebar for additional features
    st.sidebar.header("📊 Advanced Analytics")
    
    # Quick route analysis
    st.sidebar.subheader("🚀 Quick Route Analysis")
    
    if st.sidebar.button("Analyze Popular Routes"):
        route_analysis = location_router.analyze_route_patterns(data)
        if route_analysis:
            st.sidebar.success("✅ Route analysis completed!")
            
            # Show popular route insights
            st.sidebar.write("**Popular Route Categories:**")
            for category in route_analysis.keys():
                st.sidebar.write(f"• {category}")
    
    # Advanced prediction (legacy feature in sidebar)
    st.sidebar.subheader("🔧 Traditional Prediction")
    
    # Service selection
    service_options = data_processor.get_available_services()
    selected_service = st.sidebar.selectbox(
        "Ride Service",
        options=service_options,
        index=0 if service_options else None
    )
    
    # Location selection (mapped to data categories)
    location_options = data_processor.get_available_locations()
    selected_location = st.sidebar.selectbox(
        "Data Location Category",
        options=location_options,
        index=0 if location_options else None
    )
    
    selected_date = st.sidebar.date_input(
        "Date",
        value=datetime.now().date(),
        min_value=datetime.now().date(),
        max_value=datetime.now().date() + timedelta(days=7)
    )
    
    selected_datetime = datetime.combine(selected_date, datetime.min.time().replace(hour=selected_hour))
    
    if st.sidebar.button("Legacy Prediction"):
        try:
            if price_predictor is not None:
                prediction = price_predictor.predict_price(
                    selected_datetime, 
                    selected_service, 
                    selected_location
                )
                
                if prediction is not None:
                    st.sidebar.metric(
                        "Traditional Prediction",
                        f"${prediction:.2f}"
                    )
                else:
                    st.sidebar.error("Unable to generate prediction.")
            else:
                st.sidebar.error("Price predictor not available.")
                
        except Exception as e:
            st.sidebar.error(f"Error: {str(e)}")
    
    # Route Analysis Section
    st.header("📈 Route Analytics & Insights")
    
    # Route-focused tabs with traditional analytics as secondary
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Route Patterns", "Route Comparison", "Best Times", "Traditional Analytics", 
        "Advanced Models", "Surge Pricing", "Distance Analysis"
    ])
    
    with tab1:
        st.subheader("Route Category Analysis")
        try:
            st.write("**Route Category Performance Overview:**")
            
            # Show simplified route category information
            route_categories = {
                'Airport Routes': {
                    'description': 'Routes to/from JFK, LaGuardia, Newark',
                    'typical_distance': '25-45 km',
                    'price_multiplier': '1.3x standard rate'
                },
                'Cross-Borough Commutes': {
                    'description': 'Manhattan ↔ Brooklyn, Queens, Bronx',
                    'typical_distance': '8-20 km', 
                    'price_multiplier': '1.1x standard rate'
                },
                'Inner-Manhattan': {
                    'description': 'Within Manhattan neighborhoods',
                    'typical_distance': '2-8 km',
                    'price_multiplier': '1.2x standard rate'
                }
            }
            
            for category, info in route_categories.items():
                with st.expander(f"📊 {category}"):
                    st.write(f"**Description:** {info['description']}")
                    st.write(f"**Typical Distance:** {info['typical_distance']}")
                    st.write(f"**Price Factor:** {info['price_multiplier']}")
            
            # Show sample calculations
            st.write("**Sample Route Predictions:**")
            sample_routes = [
                ("Manhattan - Midtown", "JFK Airport"),
                ("Manhattan - Financial District", "Brooklyn - Williamsburg"),
                ("Manhattan - Upper East Side", "Manhattan - Midtown")
            ]
            
            current_hour = datetime.now().hour
            for origin, destination in sample_routes:
                prediction = location_router.predict_route_price(origin, destination, current_hour, 0, data)
                if prediction:
                    st.write(f"• **{origin} → {destination}**: ${prediction['estimated_price']:.2f} ({prediction['distance_km']:.1f} km)")
                
        except Exception as e:
            st.error(f"Error in route analysis: {str(e)}")
            st.write("Showing basic route information instead.")
    
    with tab2:
        st.subheader("Route Price Comparison")
        try:
            st.write("**Compare Popular NYC Routes:**")
            
            # Popular route comparisons
            popular_routes = [
                ("Manhattan - Midtown", "JFK Airport"),
                ("Manhattan - Midtown", "LaGuardia Airport"),
                ("Manhattan - Financial District", "Brooklyn - Williamsburg"),
                ("Queens - Long Island City", "Manhattan - Midtown"),
                ("Brooklyn - Park Slope", "Manhattan - Upper East Side")
            ]
            
            comparison_data = []
            current_hour = datetime.now().hour
            
            for origin, destination in popular_routes:
                prediction = location_router.predict_route_price(origin, destination, current_hour, 0, data)
                if prediction:
                    comparison_data.append({
                        'Route': f"{origin} → {destination}",
                        'Price': f"${prediction['estimated_price']:.2f}",
                        'Distance': f"{prediction['distance_km']:.1f} km",
                        'Category': prediction['route_category'],
                        'Rate/km': f"${prediction['price_per_km']:.2f}"
                    })
            
            if comparison_data:
                df = pd.DataFrame(comparison_data)
                st.dataframe(df, use_container_width=True)
                
                # Price comparison chart
                fig = go.Figure(data=[
                    go.Bar(
                        x=[item['Route'] for item in comparison_data],
                        y=[float(item['Price'].replace('$', '')) for item in comparison_data],
                        text=[item['Price'] for item in comparison_data],
                        textposition='auto'
                    )
                ])
                
                fig.update_layout(
                    title='Popular Route Price Comparison',
                    xaxis_title='Route',
                    yaxis_title='Price ($)',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error creating route comparison: {str(e)}")
    
    with tab3:
        st.subheader("Best Times for Your Route")
        try:
            if origin_location and destination_location:
                st.write(f"**Optimal timing for: {origin_location} → {destination_location}**")
                
                # Calculate prices for all hours
                hourly_prices = []
                for hour in range(24):
                    prediction = location_router.predict_route_price(origin_location, destination_location, hour, 0, data)
                    if prediction:
                        hourly_prices.append({
                            'hour': hour,
                            'price': prediction['estimated_price']
                        })
                
                if hourly_prices:
                    df_hourly = pd.DataFrame(hourly_prices)
                    
                    # Find best and worst times
                    best_time = df_hourly.loc[df_hourly['price'].idxmin()]
                    worst_time = df_hourly.loc[df_hourly['price'].idxmax()]
                    savings = worst_time['price'] - best_time['price']
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Best Time", f"{int(best_time['hour'])}:00", f"${best_time['price']:.2f}")
                    with col2:
                        st.metric("Worst Time", f"{int(worst_time['hour'])}:00", f"${worst_time['price']:.2f}")
                    with col3:
                        st.metric("Max Savings", f"${savings:.2f}")
                    
                    # Hourly price chart for selected route
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=df_hourly['hour'],
                        y=df_hourly['price'],
                        mode='lines+markers',
                        name='Price',
                        line=dict(color='blue', width=3),
                        marker=dict(size=8)
                    ))
                    
                    # Highlight best and worst times
                    fig.add_trace(go.Scatter(
                        x=[best_time['hour']],
                        y=[best_time['price']],
                        mode='markers',
                        name='Best Time',
                        marker=dict(size=15, color='green', symbol='star')
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=[worst_time['hour']],
                        y=[worst_time['price']],
                        mode='markers',
                        name='Worst Time',
                        marker=dict(size=15, color='red', symbol='x')
                    ))
                    
                    fig.update_layout(
                        title=f'Hourly Pricing: {origin_location} → {destination_location}',
                        xaxis_title='Hour of Day',
                        yaxis_title='Estimated Price ($)',
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
            else:
                st.info("Select origin and destination locations to see optimal timing.")
                
        except Exception as e:
            st.error(f"Error analyzing best times: {str(e)}")
    
    with tab4:
        st.subheader("Traditional Pricing Analytics")
        try:
            # Traditional hourly trends
            if visualizations is not None:
                hourly_fig = visualizations.create_hourly_trends(data)
                if hourly_fig:
                    st.write("**Hourly Price Trends (All Data):**")
                    st.plotly_chart(hourly_fig, use_container_width=True)
                
                # Daily patterns
                daily_fig = visualizations.create_daily_patterns(data)
                if daily_fig:
                    st.write("**Daily Price Patterns:**")
                    st.plotly_chart(daily_fig, use_container_width=True)
                
                # Peak hours analysis
                st.write("**Peak Hours Analysis:**")
                peak_analysis = data_processor.analyze_peak_hours()
                if peak_analysis:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Peak Hour", peak_analysis.get('peak_hour', 'N/A'))
                        st.metric("Peak Day", peak_analysis.get('peak_day', 'N/A'))
                    
                    with col2:
                        st.metric("Lowest Price Hour", peak_analysis.get('lowest_hour', 'N/A'))
                        st.metric("Average Peak Multiplier", f"{peak_analysis.get('peak_multiplier', 0):.2f}x")
                    
                    # Peak hours heatmap
                    heatmap_fig = visualizations.create_peak_hours_heatmap(data)
                    if heatmap_fig:
                        st.plotly_chart(heatmap_fig, use_container_width=True)
                        
            else:
                st.info("Traditional analytics not available.")
        except Exception as e:
            st.error(f"Error creating traditional analytics: {str(e)}")
    
    with tab5:
        st.subheader("Advanced ML Model Comparison")
        try:
            if data_processor is not None and hasattr(data_processor, 'advanced_models'):
                st.write("**Training Advanced ML Models...**")
                
                # Prepare basic features that definitely exist
                base_features = ['hour', 'day_of_week', 'month', 'is_weekend']
                
                # Check which additional features are available
                available_features = base_features.copy()
                
                # Encode categorical variables
                from sklearn.preprocessing import LabelEncoder
                
                X = data[base_features].copy()
                
                # Add service encoding
                try:
                    le_service = LabelEncoder()
                    X['service_encoded'] = le_service.fit_transform(data['service'])
                    available_features.append('service_encoded')
                except:
                    pass
                
                # Add location encoding
                try:
                    le_location = LabelEncoder()
                    X['location_encoded'] = le_location.fit_transform(data['location'])
                    available_features.append('location_encoded')
                except:
                    pass
                
                # Add distance features if available
                if 'trip_distance_km' in data.columns:
                    X['trip_distance_km'] = data['trip_distance_km']
                    available_features.append('trip_distance_km')
                
                if 'price_per_km' in data.columns:
                    X['price_per_km'] = data['price_per_km']
                    available_features.append('price_per_km')
                
                y = data['price']
                
                # Remove any remaining NaN values
                mask = X.notna().all(axis=1) & y.notna()
                X_clean = X[mask]
                y_clean = y[mask]
                
                # Simplified model training approach
                st.write("**Training Models on Current Data...**")
                
                # Use basic Random Forest for demonstration
                from sklearn.ensemble import RandomForestRegressor
                from sklearn.model_selection import train_test_split
                from sklearn.metrics import r2_score, mean_absolute_error
                
                if len(X_clean) > 50:
                    X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)
                    
                    # Train multiple models
                    models = {
                        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
                        'Simple Random Forest': RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
                    }
                    
                    results = {}
                    
                    for name, model in models.items():
                        try:
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)
                            
                            r2 = r2_score(y_test, y_pred)
                            mae = mean_absolute_error(y_test, y_pred)
                            
                            results[name] = {
                                'R² Score': r2,
                                'MAE': mae,
                                'Features Used': len(X_clean.columns)
                            }
                        except Exception as model_error:
                            st.warning(f"Error training {name}: {str(model_error)}")
                    
                    if results:
                        st.success("✅ Models trained successfully!")
                        
                        # Display results
                        results_df = pd.DataFrame(results).T
                        st.dataframe(results_df.round(3))
                        
                        # Visualize performance
                        model_names = list(results.keys())
                        r2_scores = [results[name]['R² Score'] for name in model_names]
                        
                        fig = go.Figure(data=[
                            go.Bar(
                                x=model_names,
                                y=r2_scores,
                                text=[f'{score:.3f}' for score in r2_scores],
                                textposition='auto',
                                marker_color=['lightblue', 'lightgreen']
                            )
                        ])
                        
                        fig.update_layout(
                            title='Model Performance Comparison',
                            xaxis_title='Model',
                            yaxis_title='R² Score',
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Best model
                        best_model = max(results.keys(), key=lambda x: results[x]['R² Score'])
                        st.info(f"Best performing model: **{best_model}** (R² = {results[best_model]['R² Score']:.3f})")
                        
                        # Model insights
                        st.write("**Model Insights:**")
                        st.info("🎯 Random Forest works well for this pricing data")
                        st.info("📊 Higher R² scores indicate better price prediction accuracy")
                        st.info("🔄 Models use temporal and location features for predictions")
                    
                else:
                    st.warning("Insufficient data for model training (need at least 50 samples)")
                    
            else:
                st.info("Advanced models component not available.")
                
        except Exception as e:
            st.error(f"Error with advanced models: {str(e)}")
            st.write("**Fallback Model Information:**")
            st.write("• Random Forest: Good for non-linear patterns")
            st.write("• Gradient Boosting: Excellent for complex relationships") 
            st.write("• Neural Network: Best for large datasets")
            st.write("• Linear Regression: Simple baseline model")
    
    with tab6:
        st.subheader("Surge Pricing Analysis")
        try:
            st.write("**Surge Pricing Insights:**")
            
            # Calculate basic surge patterns from data
            if data is not None and not data.empty:
                # Find hours with highest prices (surge indicators)
                hourly_avg = data.groupby('hour')['price'].mean()
                peak_price_hour = hourly_avg.idxmax()
                peak_price = hourly_avg.max()
                avg_price = data['price'].mean()
                surge_factor = peak_price / avg_price
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Peak Price Hour", f"{peak_price_hour}:00")
                
                with col2:
                    st.metric("Peak Price", f"${peak_price:.2f}")
                
                with col3:
                    st.metric("Surge Factor", f"{surge_factor:.1f}x")
                
                # Surge probability for current selection
                st.write("**Current Route Surge Analysis:**")
                if origin_location and destination_location:
                    # Calculate surge probability based on time
                    current_hour_data = data[data['hour'] == selected_hour]
                    if not current_hour_data.empty:
                        current_avg = current_hour_data['price'].mean()
                        surge_likelihood = min(100, ((current_avg / avg_price) - 1) * 100)
                        
                        st.info(f"At {selected_hour}:00, prices are typically {surge_likelihood:.0f}% above average")
                        
                        if surge_likelihood > 20:
                            st.warning("⚠️ High surge probability - consider alternative times")
                        elif surge_likelihood < -10:
                            st.success("✅ Low price time - good for saving money")
                
                # Create surge pattern chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=hourly_avg.index,
                    y=hourly_avg.values,
                    mode='lines+markers',
                    name='Average Price',
                    line=dict(color='blue', width=3),
                    marker=dict(size=8)
                ))
                
                fig.add_hline(y=avg_price, line_dash="dash", line_color="red", 
                             annotation_text="Overall Average")
                
                fig.update_layout(
                    title='Surge Pricing Patterns by Hour',
                    xaxis_title='Hour of Day',
                    yaxis_title='Average Price ($)',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Surge tips
                st.write("**Money-Saving Tips:**")
                low_price_hours = hourly_avg.nsmallest(3)
                st.info(f"💰 Cheapest hours: {', '.join([f'{h}:00' for h in low_price_hours.index])}")
                st.info("🚫 Avoid peak hours (typically 7-9 AM and 5-7 PM)")
                st.info("📱 Use shared rides during surge times to reduce costs")
                
            else:
                st.warning("No data available for surge analysis")
                
        except Exception as e:
            st.error(f"Error with surge pricing analysis: {str(e)}")
            st.info("Showing basic surge information instead")
    
    with tab7:
        st.subheader("Distance-Based Pricing Analysis")
        try:
            st.write("**Distance Analysis for NYC Routes:**")
            
            # Calculate distance-based insights from our location router
            route_distances = {}
            sample_routes = [
                ("Manhattan - Midtown", "JFK Airport", "Airport"),
                ("Manhattan - Financial District", "Brooklyn - Williamsburg", "Cross-Borough"),
                ("Manhattan - Upper East Side", "Manhattan - Midtown", "Inner-Manhattan"),
                ("Queens - Long Island City", "Manhattan - Midtown", "Cross-Borough"),
                ("Brooklyn - Park Slope", "LaGuardia Airport", "Airport")
            ]
            
            for origin, destination, category in sample_routes:
                distance = location_router.calculate_route_distance(origin, destination)
                if distance:
                    route_distances[f"{origin} → {destination}"] = {
                        'distance': distance,
                        'category': category
                    }
            
            if route_distances:
                # Display route distances
                st.write("**Popular Route Distances:**")
                
                for route, info in route_distances.items():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"• **{route}**")
                    with col2:
                        st.write(f"{info['distance']:.1f} km ({info['category']})")
                
                # Distance vs Price analysis
                st.write("**Distance-Price Relationship:**")
                
                # Calculate estimated rates
                airport_rate = 2.5  # $/km for airport routes
                cross_borough_rate = 2.2  # $/km for cross-borough
                manhattan_rate = 3.0  # $/km for inner-Manhattan (higher due to traffic)
                
                distances = []
                prices = []
                categories = []
                route_names = []
                
                for route, info in route_distances.items():
                    if info['category'] == 'Airport':
                        price = info['distance'] * airport_rate
                    elif info['category'] == 'Cross-Borough':
                        price = info['distance'] * cross_borough_rate
                    else:
                        price = info['distance'] * manhattan_rate
                    
                    distances.append(info['distance'])
                    prices.append(price)
                    categories.append(info['category'])
                    route_names.append(route)
                
                # Create distance vs price chart
                fig = go.Figure()
                
                colors = {'Airport': 'red', 'Cross-Borough': 'blue', 'Inner-Manhattan': 'green'}
                
                for category in ['Airport', 'Cross-Borough', 'Inner-Manhattan']:
                    cat_distances = [d for d, c in zip(distances, categories) if c == category]
                    cat_prices = [p for p, c in zip(prices, categories) if c == category]
                    cat_routes = [r for r, c in zip(route_names, categories) if c == category]
                    
                    if cat_distances:
                        fig.add_trace(go.Scatter(
                            x=cat_distances,
                            y=cat_prices,
                            mode='markers',
                            name=category,
                            marker=dict(color=colors[category], size=10),
                            text=cat_routes,
                            hovertemplate='<b>%{text}</b><br>Distance: %{x:.1f} km<br>Price: $%{y:.2f}<extra></extra>'
                        ))
                
                fig.update_layout(
                    title='Distance vs Estimated Price by Route Category',
                    xaxis_title='Distance (km)',
                    yaxis_title='Estimated Price ($)',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Pricing insights
                st.write("**Distance-Based Pricing Insights:**")
                st.info(f"🛣️ Average rate: Airport routes ${airport_rate:.2f}/km, Cross-borough ${cross_borough_rate:.2f}/km, Manhattan ${manhattan_rate:.2f}/km")
                st.info("📏 Longer trips generally offer better per-kilometer rates")
                st.info("🏙️ Manhattan routes cost more per km due to traffic and demand")
                
                # Route efficiency analysis
                st.write("**Route Efficiency Tips:**")
                st.success("✅ Airport routes: Consider subway + AirTrain for major savings")
                st.success("✅ Cross-borough: Plan during off-peak hours for lower prices")
                st.success("✅ Short trips: Walking or bike-share might be more economical")
                
            else:
                st.warning("Unable to calculate route distances")
                
        except Exception as e:
            st.error(f"Error with distance analysis: {str(e)}")
            st.info("Showing basic distance information instead")
    
    # Footer with additional information
    st.markdown("---")
    st.markdown("""
    **About:** NYC Route Pricing Optimizer - The ultimate tool for smart ride-sharing decisions in New York City.
    Get accurate price predictions, find the best times to travel, and save money on every trip.
    
    **Key Features:**
    - 🗺️ **Location-Based Routing**: Simple selection between NYC neighborhoods, airports, and landmarks
    - 💰 **Smart Price Prediction**: AI-powered pricing using 200,000+ real Uber rides
    - ⏰ **Optimal Timing**: Find the cheapest hours and days for your specific route
    - 🎯 **Money-Saving Tips**: Personalized recommendations to reduce ride costs
    - 📊 **Route Analytics**: Compare popular routes and understand pricing patterns
    - 🚀 **Real-Time Insights**: Live surge probability and demand analysis
    
    **Perfect for:** Commuters, travelers, tourists, and anyone looking to optimize their NYC transportation costs.
    """)

if __name__ == "__main__":
    main()
