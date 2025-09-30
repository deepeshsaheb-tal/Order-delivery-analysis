"""
Sample usage script for the logistics insights engine.
"""
import os
import sys
import logging
from datetime import datetime, timedelta

# Add the parent directory to the path to import the src module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loading import DataLoader
from src.preprocessing import DataPreprocessor
from src.correlation_engine import DataCorrelator
from src.root_cause_analysis import RootCauseAnalyzer
from src.insights_generation import InsightsGenerator
from src.query_processor import QueryProcessor


def print_section(title):
    """Print a section title."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_insights(insights):
    """Print insights and recommendations."""
    if 'insights' in insights and insights['insights']:
        print("\nInsights:")
        for i, insight in enumerate(insights['insights'], 1):
            print(f"  {i}. {insight}")
    
    if 'recommendations' in insights and insights['recommendations']:
        print("\nRecommendations:")
        for i, recommendation in enumerate(insights['recommendations'], 1):
            print(f"  {i}. {recommendation}")


def main():
    """Main function."""
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Define data directory
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Dataset'))
    
    print_section("Loading and preprocessing data")
    
    # Load data
    print("Loading data...")
    data_loader = DataLoader(data_dir)
    data_loader.load_all_data()
    print(f"Loaded {len(data_loader.dataframes)} datasets")
    
    # Preprocess data
    print("Preprocessing data...")
    preprocessor = DataPreprocessor()
    preprocessed_data = preprocessor.preprocess_all_data(data_loader.dataframes)
    print(f"Preprocessed {len(preprocessed_data)} datasets")
    
    # Correlate data
    print("Correlating data...")
    correlator = DataCorrelator(preprocessed_data)
    correlated_data = correlator.correlate_all_data()
    print(f"Created {len(correlated_data)} correlated datasets")
    
    # Initialize analyzer
    print("Initializing analyzer...")
    analyzer = RootCauseAnalyzer(correlated_data)
    
    # Initialize insights generator
    insights_generator = InsightsGenerator()
    
    # Initialize query processor
    query_processor = QueryProcessor()
    
    # Example 1: Analyze delivery failures in a specific city
    print_section("Example 1: Analyze delivery failures in Mumbai")
    
    # Filter data for Mumbai
    mumbai_data = correlator.filter_by_city('Mumbai')
    
    # Analyze delivery failures
    failure_analysis = analyzer.analyze_delivery_failures(mumbai_data)
    
    # Generate insights
    city_insights = insights_generator.generate_city_insights({
        'city': 'Mumbai',
        'order_count': len(mumbai_data),
        'delivery_failures': failure_analysis
    })
    
    # Print insights
    print_insights(city_insights)
    
    # Example 2: Analyze delivery delays for a specific client
    print_section("Example 2: Analyze delivery delays for Client 123")
    
    # Filter data for client
    client_data = correlator.filter_by_client(123)
    
    # Analyze delivery delays
    delay_analysis = analyzer.analyze_delivery_delays(client_data)
    
    # Generate insights
    client_insights = insights_generator.generate_client_insights({
        'client_id': 123,
        'client_name': 'Client 123',
        'order_count': len(client_data),
        'delivery_delays': delay_analysis
    })
    
    # Print insights
    print_insights(client_insights)
    
    # Example 3: Compare cities
    print_section("Example 3: Compare Mumbai and Delhi")
    
    # Analyze each city
    mumbai_analysis = analyzer.analyze_city_performance('Mumbai')
    delhi_analysis = analyzer.analyze_city_performance('Delhi')
    
    # Compare cities
    comparison = {
        'cities': ['Mumbai', 'Delhi'],
        'order_counts': [
            mumbai_analysis.get('order_count', 0),
            delhi_analysis.get('order_count', 0)
        ],
        'failure_rates': [
            mumbai_analysis.get('delivery_failures', {}).get('failure_rate', 0),
            delhi_analysis.get('delivery_failures', {}).get('failure_rate', 0)
        ],
        'failure_rate_difference': (
            mumbai_analysis.get('delivery_failures', {}).get('failure_rate', 0) -
            delhi_analysis.get('delivery_failures', {}).get('failure_rate', 0)
        ),
        'delay_rates': [
            mumbai_analysis.get('delivery_delays', {}).get('delay_rate', 0),
            delhi_analysis.get('delivery_delays', {}).get('delay_rate', 0)
        ],
        'delay_rate_difference': (
            mumbai_analysis.get('delivery_delays', {}).get('delay_rate', 0) -
            delhi_analysis.get('delivery_delays', {}).get('delay_rate', 0)
        ),
        'city1_analysis': mumbai_analysis,
        'city2_analysis': delhi_analysis
    }
    
    # Generate insights
    comparison_insights = insights_generator.generate_comparison_insights(comparison)
    
    # Print insights
    print_insights(comparison_insights)
    
    # Example 4: Predict risks for additional orders
    print_section("Example 4: Predict risks for 20,000 additional orders")
    
    # Predict risks
    risk_prediction = analyzer.predict_risks(20000)
    
    # Generate insights
    prediction_insights = insights_generator.generate_risk_prediction_insights(risk_prediction)
    
    # Print insights
    print_insights(prediction_insights)
    
    # Example 5: Process a natural language query
    print_section("Example 5: Process a natural language query")
    
    # Define a query
    query = "Why were deliveries delayed in Mumbai yesterday?"
    print(f"Query: {query}")
    
    # Process the query
    parsed_query = query_processor.process_query(query)
    print(f"\nParsed query: {parsed_query}")
    
    # Get city from parsed query
    city = parsed_query.get('entities', {}).get('city')
    
    if city:
        # Filter data for the city
        city_data = correlator.filter_by_city(city)
        
        # Analyze delivery delays
        delay_analysis = analyzer.analyze_delivery_delays(city_data)
        
        # Generate insights
        query_insights = insights_generator.generate_city_insights({
            'city': city,
            'order_count': len(city_data),
            'delivery_delays': delay_analysis
        })
        
        # Print insights
        print_insights(query_insights)
    else:
        print("\nCould not determine city from query.")


if __name__ == '__main__':
    main()
