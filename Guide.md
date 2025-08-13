# Overview

This is a **NYC Route Pricing Optimizer** - a location-based ride-sharing price prediction platform that helps users find the best routes and optimal timing for NYC travel. The application uses 200,000+ real Uber rides to provide accurate price predictions between specific NYC locations like "Manhattan to Queens" or "Brooklyn to JFK Airport".

The main focus is the **Route Pricing Optimizer** which allows users to:
- Select pickup and dropoff locations from friendly NYC neighborhood names
- Get accurate price predictions with distance and route category analysis  
- Discover the best times to travel for maximum savings
- Receive personalized money-saving tips and alternative route suggestions
- View interactive maps showing popular route networks across NYC

The system is designed as a consumer-focused tool for smart transportation decisions, with additional advanced analytics for deeper insights into NYC ride-sharing patterns.

# User Preferences

Preferred communication style: Simple, everyday language.

# Career Relevance

This project demonstrates skills directly applicable to roles at ride-sharing companies like Uber and Lyft:
- Real-world dataset processing (200K+ actual Uber rides)
- Advanced ML ensemble methods (Random Forest, Gradient Boosting, Neural Networks)
- Surge pricing analysis and demand forecasting
- Distance-based route optimization with Haversine calculations
- Geographic data analysis with NYC borough categorization
- Interactive data visualization and business intelligence
- Production-ready web application with 7 analytical modules
- Feature engineering for temporal, geographic, and distance-based variables

# Recent Enhancements (Aug 2025)

## Advanced Features Added:
- **Multiple ML Models**: Ensemble comparison of 4 different algorithms
- **Surge Pricing Engine**: Real-time surge probability prediction and analysis
- **Distance Optimization**: Route-based pricing with geographic calculations
- **Enhanced Analytics**: 7 comprehensive analytical tabs
- **Business Intelligence**: Professional insights and recommendations
- **Production Ready**: Deployment guide and GitHub portfolio optimization

## Technical Improvements:
- Model performance comparison with cross-validation
- Interactive heatmaps for surge pricing patterns
- Route optimization with pickup/dropoff coordinate analysis
- Feature importance analysis for business insights
- Comprehensive error handling and user experience optimization

# System Architecture

## Frontend Architecture
- **Framework**: Streamlit web framework for rapid prototyping and deployment
- **UI Components**: Wide layout with expandable sidebar for user inputs and controls
- **Visualization Engine**: Plotly for interactive charts and graphs with custom color palettes
- **Caching Strategy**: Uses Streamlit's `@st.cache_resource` decorator to optimize component initialization and data loading

## Backend Architecture
- **Modular Design**: Separated into three main components:
  - `DataProcessor`: Handles data loading, cleaning, and preprocessing
  - `PricePredictor`: Manages machine learning model training and predictions
  - `Visualizations`: Creates interactive charts and analytical displays
- **Machine Learning**: Random Forest Regressor with 100 estimators for price prediction
- **Feature Engineering**: Extracts temporal features (hour, day of week, month, weekend indicators) and encodes categorical variables

## Data Processing Pipeline
- **Data Sources**: Supports multiple input methods including CSV files, API endpoints, and environment variable configurations
- **Feature Preparation**: Automatic encoding of categorical variables (service type, location)
- **Data Validation**: Error handling for missing or corrupted data sources
- **Preprocessing**: Standard train-test split (80/20) for model validation

## Model Architecture
- **Algorithm**: Random Forest Regressor chosen for its robustness and interpretability
- **Features**: Time-based features, service type, and location encodings
- **Training Strategy**: Automated retraining when new data is available
- **Evaluation Metrics**: R-squared scores for both training and test sets

# External Dependencies

## Core Libraries
- **streamlit**: Web application framework for the user interface
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing and array operations
- **plotly**: Interactive visualization library (graph_objects and express modules)

## Machine Learning
- **scikit-learn**: 
  - `RandomForestRegressor` for price prediction
  - `train_test_split` for data splitting
  - `LabelEncoder` for categorical variable encoding

## Data Sources (Configurable)
- **Environment Variables**: 
  - `RIDESHARE_API_KEY`: API authentication for external ride-sharing data
  - `RIDESHARE_DATA_URL`: Endpoint for fetching real-time or historical data
- **File System**: Local CSV files (`rideshare_data.csv`) as fallback data source
- **API Integration**: Placeholder for external ride-sharing service APIs (Uber, Lyft)

## Visualization Dependencies
- **plotly.subplots**: For creating complex multi-panel visualizations
- **Custom Color Palette**: Predefined color scheme for consistent visual branding across charts