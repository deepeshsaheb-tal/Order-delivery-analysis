"""
Main application entry point for the logistics insights engine.
"""
import os
import sys
import logging
import pandas as pd
import json
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from openai import OpenAI
from dotenv import load_dotenv

# Add the parent directory to the Python path
import sys
sys.path.insert(0, '.')

from src.data_loading import DataLoader
from src.preprocessing import DataPreprocessor
from src.correlation_engine import DataCorrelator
from src.root_cause_analysis import RootCauseAnalyzer
from src.insights_generation import InsightsGenerator
from src.query_processor import QueryProcessor
from src.ui import ConsoleUI

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LogisticsInsightsEngine:
    """
    Main application class for the logistics insights engine.
    """
    
    def __init__(self, data_dir: str = 'Dataset', verbose: bool = False):
        """
        Initialize the logistics insights engine.
        
        Args:
            data_dir: Directory containing data files
            verbose: Whether to enable verbose output
        """
        self.data_dir = data_dir
        self.verbose = verbose
        
        # Set up logging level based on verbose flag
        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        else:
            logging.getLogger().setLevel(logging.INFO)
        
        # Load environment variables for API keys
        load_dotenv()
        
        # Initialize OpenAI client
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            logger.warning("OpenAI API key not found. Some functionality will be limited.")
            self.openai_client = None
        else:
            self.openai_client = OpenAI(api_key=self.api_key)
        
        # Initialize components
        self.ui = ConsoleUI()
        self.query_processor = QueryProcessor()
        self.insights_generator = InsightsGenerator()
        
        # These will be initialized when loading data
        self.data_loader = None
        self.preprocessor = None
        self.correlator = None
        self.analyzer = None
        
        # Track whether data is loaded
        self.data_loaded = False
    
    def load_data(self) -> bool:
        """
        Load and preprocess data.
        
        Returns:
            bool: True if data was loaded successfully, False otherwise
        """
        if self.verbose:
            self.ui.display_loading("Loading data...")
        
        try:
            # Initialize data loader
            self.data_loader = DataLoader(self.data_dir)
            
            # Load data
            if not self.data_loader.load_all_data():
                self.ui.display_error("Failed to load data.")
                return False
            
            if self.verbose:
                self.ui.display_loading("Preprocessing data...")
            
            # Initialize preprocessor
            self.preprocessor = DataPreprocessor()
            
            # Preprocess data
            preprocessed_data = self.preprocessor.preprocess_all_data(self.data_loader.dataframes)
            
            if self.verbose:
                self.ui.display_loading("Correlating data...")
            
            # Initialize correlator
            self.correlator = DataCorrelator(preprocessed_data)
            
            # Correlate data
            correlated_data = self.correlator.correlate_all_data()
            
            # Initialize analyzer
            self.analyzer = RootCauseAnalyzer(correlated_data)
            
            self.data_loaded = True
            return True
        
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            self.ui.display_error(f"Error loading data: {str(e)}")
            return False
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a user query.
        
        Args:
            query: Natural language query string
            
        Returns:
            Dict[str, Any]: Query results
        """
        # Make sure data is loaded
        if not self.data_loaded and not self.load_data():
            return {"error": "Failed to load data."}
        
        # Direct check for festival-related keywords
        query_lower = query.lower()
        if 'festival' in query_lower or 'holiday' in query_lower or 'peak period' in query_lower:
            # Generate festival insights directly using the original query
            insights = self._generate_festival_insights(query)
            return {
                "parsed_query": {"query_type": "festival_analysis"},
                "results": {"special_period": "festival_period"},
                "insights": insights
            }
        
        try:
            # Process the query
            if self.verbose:
                self.ui.display_loading("Processing query...")
            
            # Parse the query
            parsed_query = self.query_processor.process_query(query)
            
            if self.verbose:
                self.ui.display_loading("Analyzing data...")
                logger.info(f"Parsed query: {parsed_query}")
            
            # Process based on query type
            query_type = parsed_query.get('query_type')
            entities = parsed_query.get('entities', {})
            time_range = parsed_query.get('time_range', {})
            additional_params = parsed_query.get('additional_parameters', {})
            
            # Special handling for city list in comparison queries
            if query_type == 'comparison' and isinstance(entities.get('city'), list):
                # Convert city list to string for simpler handling
                if len(entities['city']) >= 2:
                    # Create a detailed comparison directly
                    city1, city2 = entities['city'][0], entities['city'][1]
                    
                    # Get data for each city
                    city1_data = self._get_filtered_data(city1)
                    city2_data = self._get_filtered_data(city2)
                    
                    # Check data availability
                    city1_available = city1_data is not None and not city1_data.empty
                    city2_available = city2_data is not None and not city2_data.empty
                    
                    # Calculate metrics
                    city1_metrics = self._calculate_city_metrics(city1, city1_data) if city1_available else {}
                    city2_metrics = self._calculate_city_metrics(city2, city2_data) if city2_available else {}
                    
                    # Create detailed comparison
                    results = {
                        'cities': [city1, city2],
                        'order_counts': [len(city1_data) if city1_available else 0, 
                                        len(city2_data) if city2_available else 0],
                        'city1_data_available': city1_available,
                        'city2_data_available': city2_available,
                        'city1_metrics': city1_metrics,
                        'city2_metrics': city2_metrics,
                        'comparison_type': 'detailed'
                    }
                    
                    # Add comparison metrics if both cities have data
                    if city1_available and city2_available:
                        results['comparison_metrics'] = self._compare_city_metrics(city1_metrics, city2_metrics)
                    
                    # Generate insights
                    insights = self._generate_insights(query_type, results)
                    
                    return {
                        "parsed_query": parsed_query,
                        "results": results,
                        "insights": insights
                    }
                else:
                    # Not enough cities in the list
                    entities['city'] = entities['city'][0] if entities['city'] else None
            
            # Get results based on query type
            try:
                results = self._get_analysis_results(query_type, entities, time_range, additional_params)
                if self.verbose:
                    logger.info(f"Analysis results structure: {list(results.keys()) if isinstance(results, dict) else 'Not a dict'}")
            except Exception as analysis_error:
                logger.error(f"Error in analysis: {str(analysis_error)}")
                return {"error": f"Error in analysis: {str(analysis_error)}"}
            
            if self.verbose:
                self.ui.display_loading("Generating insights...")
            
            # Log the query for debugging
            logger.info(f"Processing query: {query}")
            
            # Special handling for festival period queries
            if query_type == 'festival_analysis' or additional_params.get('special_period') == 'festival_period':
                insights = self._generate_festival_insights(query)
            else:
                # Generate insights for other query types
                try:
                    insights = self._generate_insights(query_type, results, additional_params)
                except Exception as insights_error:
                    logger.error(f"Error generating insights: {str(insights_error)}")
                    return {"error": f"Error generating insights: {str(insights_error)}"}
            
            return {
                "parsed_query": parsed_query,
                "results": results,
                "insights": insights
            }
        
        except Exception as e:
            import traceback
            logger.error(f"Error processing query: {str(e)}")
            logger.error(traceback.format_exc())
            return {"error": f"Error processing query: {str(e)}"}
    
    def _get_analysis_results(
        self, 
        query_type: str, 
        entities: Dict[str, Any], 
        time_range: Dict[str, str],
        additional_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Get analysis results based on query type and parameters.
        
        Args:
            query_type: Type of query
            entities: Dictionary of entities
            time_range: Dictionary of time range parameters
            additional_params: Dictionary of additional parameters
            
        Returns:
            Dict[str, Any]: Analysis results
        """
        # Extract entities
        city = entities.get('city')
        client = entities.get('client')
        warehouse = entities.get('warehouse')
        
        # Extract time range
        start_date = None
        end_date = None
        
        # Extract date ranges safely
        try:
            if time_range.get('start_date'):
                start_date = datetime.strptime(time_range['start_date'], '%Y-%m-%d')
        except ValueError as e:
            logger.warning(f"Could not parse start date: {e}")
            # Use a default date range if parsing fails
            start_date = datetime.now() - timedelta(days=30)
        
        try:
            if time_range.get('end_date'):
                end_date = datetime.strptime(time_range['end_date'], '%Y-%m-%d')
        except ValueError as e:
            logger.warning(f"Could not parse end date: {e}")
            # Use current date if parsing fails
            end_date = datetime.now()
            
        # Handle special periods
        special_period = additional_params.get('special_period')
        
        # If this is a festival period query, adjust the query type and parameters
        if special_period == 'festival_period':
            # For festival period, we want to focus on failure analysis
            if query_type in ['general_analysis', 'explanation']:
                query_type = 'failure_analysis'
            
            # Add specific reason for the analysis
            additional_params['specific_reason'] = 'festival_period'
        
        # No special handling for specific months
        
        # Get filtered data based on entities and time range
        filtered_data = self._get_filtered_data(city, client, warehouse, start_date, end_date)
        
        # Process based on query type
        if query_type == 'comparison' and city and isinstance(city, list) and len(city) >= 2:
            # Create a detailed comparison
            city1, city2 = city[0], city[1]
            return self._create_city_comparison(city1, city2)
        
        elif query_type == 'prediction':
            # Extract order volume
            order_volume = additional_params.get('order_volume', 20000)  # Default to 20,000
            
            # Get client ID if available
            client_id = None
            if client and client.isdigit():
                client_id = int(client)
            
            return self.analyzer.predict_risks(order_volume, client_id)
        
        elif query_type == 'explanation':
            if warehouse:
                # Try to get warehouse ID
                warehouse_id = None
                if warehouse.isdigit():
                    warehouse_id = int(warehouse)
                
                # Pass the filtered data to the warehouse analysis
                return self.analyzer.analyze_warehouse_performance(warehouse_id, warehouse, filtered_data)
            
            elif client:
                # Try to get client ID
                client_id = None
                if client.isdigit():
                    client_id = int(client)
                
                # Pass the filtered data to the client analysis
                return self.analyzer.analyze_client_performance(client_id, client, filtered_data)
            
            elif city:
                # Pass the filtered data to the city analysis
                return self.analyzer.analyze_city_performance(city, filtered_data)
            
            else:
                # General analysis
                return {
                    "delivery_failures": self.analyzer.analyze_delivery_failures(),
                    "delivery_delays": self.analyzer.analyze_delivery_delays()
                }
        
        else:
            # Default to general analysis
            failure_analysis = self.analyzer.analyze_delivery_failures(filtered_data)
            delay_analysis = self.analyzer.analyze_delivery_delays(filtered_data)
            
            if city:
                results = {
                    'city': city,
                    'order_count': len(filtered_data) if filtered_data is not None else 0,
                    'delivery_failures': failure_analysis,
                    'delivery_delays': delay_analysis
                }
            elif client:
                results = {
                    'client_id': client if client.isdigit() else None,
                    'client_name': client if not client.isdigit() else f"Client {client}",
                    'order_count': len(filtered_data) if filtered_data is not None else 0,
                    'delivery_failures': failure_analysis,
                    'delivery_delays': delay_analysis
                }
            else:
                results = {
                    'delivery_failures': failure_analysis,
                    'delivery_delays': delay_analysis
                }
            
            return results
        
        # Get filtered data based on parameters
        filtered_data = self._get_filtered_data(city, client, warehouse, start_date, end_date)
        
        # Store analysis results
        results = {}
        
        # Get results based on query type
        if query_type == 'festival_analysis' or special_period == 'festival_period':
            # For festival period analysis, we want to analyze delivery failures
            failure_analysis = self.analyzer.analyze_delivery_failures(filtered_data)
            
            # Create a custom result structure for festival period
            results = {
                'delivery_failures': failure_analysis,
                'festival_period': True,
                'special_period': 'festival_period'
            }
            
            return results
            
        elif query_type == 'failure_analysis':
            # Analyze delivery failures
            failure_analysis = self.analyzer.analyze_delivery_failures(filtered_data)
            
            # Add city/client information for better insights
            if city:
                results = {
                    'city': city,
                    'order_count': len(filtered_data) if filtered_data is not None else 0,
                    'delivery_failures': failure_analysis
                }
            elif client:
                results = {
                    'client_id': client if client.isdigit() else None,
                    'client_name': client if not client.isdigit() else f"Client {client}",
                    'order_count': len(filtered_data) if filtered_data is not None else 0,
                    'delivery_failures': failure_analysis
                }
            else:
                results = failure_analysis
            
            return results
        
        elif query_type == 'delay_analysis':
            # Analyze delivery delays
            delay_analysis = self.analyzer.analyze_delivery_delays(filtered_data)
            
            # Add city/client information for better insights
            if city:
                results = {
                    'city': city,
                    'order_count': len(filtered_data) if filtered_data is not None else 0,
                    'delivery_delays': delay_analysis
                }
            elif client:
                results = {
                    'client_id': client if client.isdigit() else None,
                    'client_name': client if not client.isdigit() else f"Client {client}",
                    'order_count': len(filtered_data) if filtered_data is not None else 0,
                    'delivery_delays': delay_analysis
                }
            else:
                results = delay_analysis
            
            return results
        
        elif query_type == 'comparison':
            # For comparison, we need two cities
            if isinstance(city, list) and len(city) >= 2:
                # If city is already a list with at least 2 cities
                city1, city2 = city[0], city[1]
            elif city and isinstance(city, str):
                # If city is a string, try to split it
                cities = city.split(' and ')
                if len(cities) == 2:
                    city1, city2 = cities[0], cities[1]
                else:
                    # Try to find another city in the query
                    city2 = entities.get('city2')
                    if not city2:
                        return {"error": "Need two cities for comparison."}
                    city1 = city
            else:
                return {"error": "Need two cities for comparison."}
            
            # Create a detailed comparison
            try:
                # Get data for each city
                city1_data = self._get_filtered_data(city1)
                city2_data = self._get_filtered_data(city2)
                
                # Check data availability
                city1_available = city1_data is not None and not city1_data.empty
                city2_available = city2_data is not None and not city2_data.empty
                
                # Calculate basic metrics for each city
                city1_metrics = self._calculate_city_metrics(city1, city1_data) if city1_available else {}
                city2_metrics = self._calculate_city_metrics(city2, city2_data) if city2_available else {}
                
                # Create detailed comparison
                comparison = {
                    'cities': [city1, city2],
                    'order_counts': [len(city1_data) if city1_available else 0, 
                                     len(city2_data) if city2_available else 0],
                    'city1_data_available': city1_available,
                    'city2_data_available': city2_available,
                    'city1_metrics': city1_metrics,
                    'city2_metrics': city2_metrics,
                    'comparison_type': 'detailed'
                }
                
                # Add comparison metrics if both cities have data
                if city1_available and city2_available:
                    comparison['comparison_metrics'] = self._compare_city_metrics(city1_metrics, city2_metrics)
                
                return comparison
            except Exception as e:
                logger.error(f"Error in city comparison: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                return {"error": f"Error comparing cities: {str(e)}"}
        
        elif query_type == 'prediction':
            # Extract order volume
            order_volume = additional_params.get('order_volume', 20000)  # Default to 20,000
            
            # Get client ID if available
            client_id = None
            if client and client.isdigit():
                client_id = int(client)
            
            return self.analyzer.predict_risks(order_volume, client_id)
        
        elif query_type == 'explanation':
            if warehouse:
                # Try to get warehouse ID
                warehouse_id = None
                if warehouse.isdigit():
                    warehouse_id = int(warehouse)
                
                # First get filtered data based on time range
                filtered_warehouse_data = self._get_filtered_data(None, None, warehouse, start_date, end_date)
                
                # Pass the filtered data to the warehouse analysis
                return self.analyzer.analyze_warehouse_performance(warehouse_id, warehouse, filtered_warehouse_data)
            
            elif client:
                # Try to get client ID
                client_id = None
                if client.isdigit():
                    client_id = int(client)
                
                # First get filtered data based on time range
                filtered_client_data = self._get_filtered_data(None, client, None, start_date, end_date)
                
                # Pass the filtered data to the client analysis
                return self.analyzer.analyze_client_performance(client_id, client, filtered_client_data)
            
            elif city:
                # First get filtered data based on time range
                filtered_city_data = self._get_filtered_data(city, None, None, start_date, end_date)
                
                # Pass the filtered data to the city analysis
                return self.analyzer.analyze_city_performance(city, filtered_city_data)
            
            else:
                # General analysis
                return {
                    "delivery_failures": self.analyzer.analyze_delivery_failures(),
                    "delivery_delays": self.analyzer.analyze_delivery_delays()
                }
        
        else:
            # Default to general analysis
            failure_analysis = self.analyzer.analyze_delivery_failures(filtered_data)
            delay_analysis = self.analyzer.analyze_delivery_delays(filtered_data)
            
            if city:
                results = {
                    'city': city,
                    'order_count': len(filtered_data) if filtered_data is not None else 0,
                    'delivery_failures': failure_analysis,
                    'delivery_delays': delay_analysis
                }
            elif client:
                results = {
                    'client_id': client if client.isdigit() else None,
                    'client_name': client if not client.isdigit() else f"Client {client}",
                    'order_count': len(filtered_data) if filtered_data is not None else 0,
                    'delivery_failures': failure_analysis,
                    'delivery_delays': delay_analysis
                }
            else:
                results = {
                    'delivery_failures': failure_analysis,
                    'delivery_delays': delay_analysis
                }
            
            return results
    
    def _get_filtered_data(
        self, 
        city: Optional[str] = None, 
        client: Optional[str] = None, 
        warehouse: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ):
        """
        Get filtered data based on parameters.
        
        Args:
            city: City name to filter by
            client: Client name or ID to filter by
            warehouse: Warehouse name or ID to filter by
            start_date: Start date to filter by
            end_date: End date to filter by
            
        Returns:
            pd.DataFrame: Filtered data
        """
        # Start with comprehensive data
        data = self.correlator.correlated_data.get('order_comprehensive')
        
        if data is None:
            return None
        
        # Apply filters
        if city:
            data = data[data['city'] == city]
        
        if client:
            if client.isdigit():
                # Filter by client ID
                data = data[data['client_id'] == int(client)]
                logger.info(f"Filtered by client ID: {client}, found {len(data)} rows")
            elif 'client_name' in data.columns:
                # Try exact match first
                exact_match = data[data['client_name'] == client]
                if not exact_match.empty:
                    data = exact_match
                    logger.info(f"Filtered by exact client name: {client}, found {len(data)} rows")
                else:
                    # Try case-insensitive match
                    case_insensitive_match = data[data['client_name'].str.lower() == client.lower()]
                    if not case_insensitive_match.empty:
                        data = case_insensitive_match
                        logger.info(f"Filtered by case-insensitive client name: {client}, found {len(data)} rows")
                    else:
                        # Try partial match (contains)
                        partial_match = data[data['client_name'].str.contains(client, case=False, na=False)]
                        if not partial_match.empty:
                            data = partial_match
                            logger.info(f"Filtered by partial client name: {client}, found {len(data)} rows")
                        else:
                            # Try removing special characters and matching
                            clean_client = re.sub(r'[^a-zA-Z0-9]', '', client)
                            clean_data_names = data['client_name'].apply(lambda x: re.sub(r'[^a-zA-Z0-9]', '', str(x)))
                            clean_match = data[clean_data_names.str.contains(clean_client, case=False, na=False)]
                            if not clean_match.empty:
                                data = clean_match
                                logger.info(f"Filtered by cleaned client name: {client}, found {len(data)} rows")
                            else:
                                logger.warning(f"No matching client found for name: {client}")
                                # Don't return empty DataFrame yet, as there might be other filters
        
        if warehouse:
            # Get warehouse logs data
            warehouse_logs = self.correlator.correlated_data.get('warehouse_logs')
            logger.info(f"Filtering by warehouse: {warehouse}")
            
            if warehouse_logs is not None:
                # Filter warehouse logs by warehouse ID or name
                if warehouse.isdigit():
                    # Filter by warehouse ID
                    logger.info(f"Filtering warehouse logs by ID: {warehouse}")
                    filtered_logs = warehouse_logs[warehouse_logs['warehouse_id'] == int(warehouse)]
                    logger.info(f"Found {len(filtered_logs)} logs for warehouse ID {warehouse}")
                elif 'warehouse_name' in warehouse_logs.columns:
                    # Filter by warehouse name
                    logger.info(f"Filtering warehouse logs by name: {warehouse}")
                    filtered_logs = warehouse_logs[warehouse_logs['warehouse_name'] == warehouse]
                    logger.info(f"Found {len(filtered_logs)} logs for warehouse name {warehouse}")
                else:
                    # Can't filter by warehouse name, try to get warehouse ID from warehouses table
                    logger.info("Trying to get warehouse ID from warehouses table")
                    warehouses = self.correlator.correlated_data.get('warehouses')
                    if warehouses is not None and 'warehouse_name' in warehouses.columns:
                        warehouse_id = warehouses.loc[warehouses['warehouse_name'] == warehouse, 'warehouse_id'].values
                        if len(warehouse_id) > 0:
                            logger.info(f"Found warehouse ID {warehouse_id[0]} for name {warehouse}")
                            filtered_logs = warehouse_logs[warehouse_logs['warehouse_id'] == warehouse_id[0]]
                            logger.info(f"Found {len(filtered_logs)} logs for warehouse ID {warehouse_id[0]}")
                        else:
                            logger.warning(f"No warehouse found with name: {warehouse}")
                            filtered_logs = pd.DataFrame()
                    else:
                        logger.warning("Warehouses table not available or missing warehouse_name column")
                        filtered_logs = pd.DataFrame()
                
                # Join with orders data
                if not filtered_logs.empty:
                    # Get the order IDs from filtered logs
                    order_ids = filtered_logs['order_id'].unique()
                    logger.info(f"Found {len(order_ids)} unique order IDs in warehouse logs")
                    
                    # Filter orders by these IDs
                    original_count = len(data)
                    data = data[data['order_id'].isin(order_ids)]
                    logger.info(f"Filtered orders from {original_count} to {len(data)} based on warehouse logs")
                
                else:
                    # No matching warehouse logs, return empty DataFrame
                    logger.warning("No matching warehouse logs found")
                    data = pd.DataFrame()
        
        # Apply date filtering
        if start_date or end_date:
            logger.info(f"Applying date filtering: start_date={start_date}, end_date={end_date}")
            
            # Apply date filtering to all date ranges
            # Try different date columns
            date_columns = ['order_date', 'created_at']
            date_filtered = False
            
            for date_col in date_columns:
                if date_col in data.columns:
                    logger.info(f"Using {date_col} for date filtering")
                    
                    # Check if the column is a string or datetime type
                    sample_value = data[date_col].iloc[0] if not data.empty else None
                    is_string_date = isinstance(sample_value, str)
                    
                    if is_string_date:
                        logger.info(f"Column {date_col} contains string dates")
                        
                        if start_date:
                            start_date_str = start_date.strftime('%Y-%m-%d')
                            logger.info(f"Filtering by start date: {start_date_str}")
                            data = data[data[date_col] >= start_date_str]
                        
                        if end_date:
                            end_date_str = end_date.strftime('%Y-%m-%d')
                            logger.info(f"Filtering by end date: {end_date_str}")
                            data = data[data[date_col] <= end_date_str]
                    else:
                        logger.info(f"Column {date_col} contains datetime objects")
                        
                        if start_date:
                            logger.info(f"Filtering by start date: {start_date}")
                            data = data[data[date_col] >= start_date]
                        
                        if end_date:
                            logger.info(f"Filtering by end date: {end_date}")
                            data = data[data[date_col] <= end_date]
                    
                    date_filtered = True
                    break
                
                if not date_filtered:
                    logger.warning(f"No suitable date column found for filtering. Available columns: {data.columns.tolist()}")
        
        logger.info(f"Final filtered data has {len(data)} rows")
        if len(data) > 0 and 'status' in data.columns:
            logger.info(f"Status distribution: {data['status'].value_counts().to_dict()}")
            if 'Failed' in data['status'].values:
                failed_count = len(data[data['status'] == 'Failed'])
                logger.info(f"Found {failed_count} failed orders in filtered data")
        
        return data
    
    # Methods removed - no special case handling for Warehouse 5 in August

    def _calculate_city_metrics(self, city: str, city_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate metrics for a city.
        
        Args:
            city: City name
            city_data: DataFrame with city data
            
        Returns:
            Dict[str, Any]: City metrics
        """
        logger.info(f"Calculating metrics for city: {city}")
        
        try:
            # Check if data is available
            if city_data is None or city_data.empty:
                logger.warning(f"No data available for city: {city}")
                return {}
            
            # Calculate basic metrics
            order_count = len(city_data)
            
            # Calculate failure metrics
            failed_orders = city_data[city_data['status'] == 'Failed']
            failed_count = len(failed_orders)
            failure_rate = failed_count / order_count if order_count > 0 else 0
            
            # Calculate delay metrics
            delayed_orders = city_data[city_data['status'] == 'Delayed']
            delayed_count = len(delayed_orders)
            delay_rate = delayed_count / order_count if order_count > 0 else 0
            
            return {
                'city': city,
                'order_count': order_count,
                'delivery_failures': {
                    'failed_orders': failed_count,
                    'failure_rate': failure_rate
                },
                'delivery_delays': {
                    'delayed_orders': delayed_count,
                    'delay_rate': delay_rate
                }
            }
        except Exception as e:
            logger.error(f"Error calculating metrics for city {city}: {str(e)}")
            return {}
    
    def _create_city_comparison(self, city1: str, city2: str) -> Dict[str, Any]:
        """
        Create a comparison between two cities.
        
        Args:
            city1: First city to compare
            city2: Second city to compare
            
        Returns:
            Dict[str, Any]: Comparison results
        """
        try:
            # Get analysis for each city
            logger.info(f"Analyzing city1: {city1}")
            city1_analysis = self.analyzer.analyze_city_performance(city1)
            logger.info(f"City1 analysis keys: {list(city1_analysis.keys()) if isinstance(city1_analysis, dict) and city1_analysis else 'Empty dict'}")
            
            logger.info(f"Analyzing city2: {city2}")
            city2_analysis = self.analyzer.analyze_city_performance(city2)
            logger.info(f"City2 analysis keys: {list(city2_analysis.keys()) if isinstance(city2_analysis, dict) and city2_analysis else 'Empty dict'}")
            
            # Handle empty analysis results
            if not city1_analysis:
                city1_analysis = {'city': city1, 'order_count': 0, 'delivery_failures': {}, 'delivery_delays': {}}
                logger.warning(f"No data found for city: {city1}, using empty placeholder")
            
            if not city2_analysis:
                city2_analysis = {'city': city2, 'order_count': 0, 'delivery_failures': {}, 'delivery_delays': {}}
                logger.warning(f"No data found for city: {city2}, using empty placeholder")
            
            # Extract values safely
            c1_failure_rate = city1_analysis.get('delivery_failures', {}).get('failure_rate', 0)
            c2_failure_rate = city2_analysis.get('delivery_failures', {}).get('failure_rate', 0)
            logger.info(f"Failure rates: {c1_failure_rate}, {c2_failure_rate}")
            
            c1_delay_rate = city1_analysis.get('delivery_delays', {}).get('delay_rate', 0)
            c2_delay_rate = city2_analysis.get('delivery_delays', {}).get('delay_rate', 0)
            logger.info(f"Delay rates: {c1_delay_rate}, {c2_delay_rate}")
            
            # Create comparison dictionary
            comparison = {
                'cities': [city1, city2],
                'order_counts': [
                    city1_analysis.get('order_count', 0),
                    city2_analysis.get('order_count', 0)
                ],
                'failure_rates': [c1_failure_rate, c2_failure_rate],
                'failure_rate_difference': (c1_failure_rate - c2_failure_rate),
                'delay_rates': [c1_delay_rate, c2_delay_rate],
                'delay_rate_difference': (c1_delay_rate - c2_delay_rate),
                'city1_analysis': city1_analysis,
                'city2_analysis': city2_analysis,
                'comparison_type': 'detailed',
                'city1_data_available': bool(city1_analysis),
                'city2_data_available': bool(city2_analysis),
                'city1_metrics': city1_analysis,
                'city2_metrics': city2_analysis
            }
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error comparing cities: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {"error": f"Error comparing cities: {str(e)}"}
    
    def _calculate_city_metrics(self, city: str, city_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate detailed metrics for a city based on its data.
        
        Args:
            city: City name
            city_data: DataFrame containing city data
            
        Returns:
            Dict[str, Any]: City metrics
        """
        metrics = {
            'city': city,
            'order_count': len(city_data),
            'delivery_metrics': {},
            'failure_metrics': {},
            'external_factors': {}
        }
        
        # Calculate delivery metrics
        if 'status' in city_data.columns:
            # Order status distribution
            status_counts = city_data['status'].value_counts().to_dict()
            metrics['delivery_metrics']['status_distribution'] = status_counts
            
            # Calculate delivery rate
            delivered_count = status_counts.get('Delivered', 0)
            metrics['delivery_metrics']['delivery_rate'] = delivered_count / len(city_data) * 100 if len(city_data) > 0 else 0
            
            # Calculate failure rate
            failed_count = status_counts.get('Failed', 0)
            metrics['delivery_metrics']['failure_rate'] = failed_count / len(city_data) * 100 if len(city_data) > 0 else 0
            
            # Calculate pending rate
            pending_count = status_counts.get('Pending', 0)
            metrics['delivery_metrics']['pending_rate'] = pending_count / len(city_data) * 100 if len(city_data) > 0 else 0
        
        # Calculate delay metrics
        if 'is_late_delivery' in city_data.columns:
            delayed_orders = city_data[city_data['is_late_delivery'] == 1]
            metrics['delivery_metrics']['delay_rate'] = len(delayed_orders) / len(city_data) * 100 if len(city_data) > 0 else 0
            
            if 'delivery_delay_hours' in city_data.columns and not delayed_orders.empty:
                metrics['delivery_metrics']['avg_delay_hours'] = delayed_orders['delivery_delay_hours'].mean()
                metrics['delivery_metrics']['max_delay_hours'] = delayed_orders['delivery_delay_hours'].max()
        
        # Extract failure reasons if available
        if 'failure_reason' in city_data.columns and 'status' in city_data.columns:
            failed_orders = city_data[city_data['status'] == 'Failed']
            if not failed_orders.empty:
                failure_reasons = failed_orders['failure_reason'].value_counts().to_dict()
                metrics['failure_metrics']['reasons'] = failure_reasons
                metrics['failure_metrics']['total_failures'] = len(failed_orders)
                metrics['failure_metrics']['failure_rate'] = len(failed_orders) / len(city_data) * 100 if len(city_data) > 0 else 0
                
                # Categorize failure reasons
                categories = {
                    'Inventory': ['stockout', 'out of stock', 'inventory'],
                    'Address': ['address', 'location', 'gps', 'map'],
                    'Customer': ['customer', 'recipient', 'not available', 'refused'],
                    'Payment': ['payment', 'cod', 'cash', 'transaction'],
                    'Logistics': ['vehicle', 'transport', 'breakdown', 'accident'],
                    'Weather': ['weather', 'rain', 'storm', 'snow'],
                    'Other': []
                }
                
                category_counts = {category: 0 for category in categories}
                
                for reason, count in failure_reasons.items():
                    reason_lower = str(reason).lower()
                    matched = False
                    
                    for category, keywords in categories.items():
                        if any(keyword in reason_lower for keyword in keywords):
                            category_counts[category] += count
                            matched = True
                            break
                    
                    if not matched:
                        category_counts['Other'] += count
                
                metrics['failure_metrics']['categories'] = category_counts
                
                # Calculate category percentages
                total_failures = len(failed_orders)
                if total_failures > 0:
                    category_percentages = {category: (count / total_failures * 100) for category, count in category_counts.items()}
                    metrics['failure_metrics']['category_percentages'] = category_percentages
                    
                    # Identify top failure categories
                    top_categories = sorted([(cat, count) for cat, count in category_counts.items() if count > 0], 
                                           key=lambda x: x[1], reverse=True)
                    if top_categories:
                        metrics['failure_metrics']['top_category'] = top_categories[0][0]
                        metrics['failure_metrics']['top_category_count'] = top_categories[0][1]
                        metrics['failure_metrics']['top_category_percentage'] = category_percentages[top_categories[0][0]]
                
                # Extract specific failure reasons
                if 'failure_details' in failed_orders.columns:
                    details = failed_orders['failure_details'].value_counts().to_dict()
                    metrics['failure_metrics']['specific_reasons'] = details
                    
                    # Get top specific reasons
                    top_reasons = sorted([(reason, count) for reason, count in details.items()], 
                                        key=lambda x: x[1], reverse=True)[:3]
                    if top_reasons:
                        metrics['failure_metrics']['top_specific_reasons'] = top_reasons
        
        # Extract external factors if available
        external_factor_cols = ['has_traffic', 'has_bad_weather', 'has_event']
        if all(col in city_data.columns for col in external_factor_cols):
            metrics['external_factors']['traffic_impact'] = city_data['has_traffic'].mean() * 100
            metrics['external_factors']['weather_impact'] = city_data['has_bad_weather'].mean() * 100
            metrics['external_factors']['event_impact'] = city_data['has_event'].mean() * 100
        
        return metrics
    
    def _compare_city_metrics(self, city1_metrics: Dict[str, Any], city2_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare metrics between two cities.
        
        Args:
            city1_metrics: Metrics for the first city
            city2_metrics: Metrics for the second city
            
        Returns:
            Dict[str, Any]: Comparison metrics
        """
        comparison = {}
        
        # Compare delivery metrics
        if 'delivery_metrics' in city1_metrics and 'delivery_metrics' in city2_metrics:
            delivery_comparison = {}
            
            # Compare delivery rates
            if 'delivery_rate' in city1_metrics['delivery_metrics'] and 'delivery_rate' in city2_metrics['delivery_metrics']:
                c1_rate = city1_metrics['delivery_metrics']['delivery_rate']
                c2_rate = city2_metrics['delivery_metrics']['delivery_rate']
                delivery_comparison['delivery_rate_diff'] = c1_rate - c2_rate
                delivery_comparison['better_delivery_city'] = city1_metrics['city'] if c1_rate > c2_rate else city2_metrics['city']
            
            # Compare failure rates
            if 'failure_rate' in city1_metrics['delivery_metrics'] and 'failure_rate' in city2_metrics['delivery_metrics']:
                c1_rate = city1_metrics['delivery_metrics']['failure_rate']
                c2_rate = city2_metrics['delivery_metrics']['failure_rate']
                delivery_comparison['failure_rate_diff'] = c1_rate - c2_rate
                delivery_comparison['better_failure_city'] = city1_metrics['city'] if c1_rate < c2_rate else city2_metrics['city']
            
            # Compare delay rates
            if 'delay_rate' in city1_metrics['delivery_metrics'] and 'delay_rate' in city2_metrics['delivery_metrics']:
                c1_rate = city1_metrics['delivery_metrics']['delay_rate']
                c2_rate = city2_metrics['delivery_metrics']['delay_rate']
                delivery_comparison['delay_rate_diff'] = c1_rate - c2_rate
                delivery_comparison['better_delay_city'] = city1_metrics['city'] if c1_rate < c2_rate else city2_metrics['city']
            
            comparison['delivery_comparison'] = delivery_comparison
        
        # Compare external factors
        if 'external_factors' in city1_metrics and 'external_factors' in city2_metrics:
            factor_comparison = {}
            
            factors = ['traffic_impact', 'weather_impact', 'event_impact']
            for factor in factors:
                if factor in city1_metrics['external_factors'] and factor in city2_metrics['external_factors']:
                    c1_impact = city1_metrics['external_factors'][factor]
                    c2_impact = city2_metrics['external_factors'][factor]
                    factor_comparison[f'{factor}_diff'] = c1_impact - c2_impact
                    factor_comparison[f'higher_{factor}_city'] = city1_metrics['city'] if c1_impact > c2_impact else city2_metrics['city']
            
            comparison['external_factor_comparison'] = factor_comparison
        
        return comparison
    
    def _generate_festival_insights(self, query: str = None) -> Dict[str, Any]:
        """
        Generate insights for festival period queries using OpenAI.
        
        Args:
            query: The original user query (optional)
            
        Returns:
            Dict[str, Any]: Generated insights
        """
        # Check if OpenAI client is available
        if not self.openai_client:
            # Fall back to dynamic insights if OpenAI is not available
            return self._generate_dynamic_festival_insights()
        
        try:
            # Get relevant data for context
            context_data = self._get_festival_context_data()
            
            # Format the query for better results
            if not query:
                query = "What are the likely causes of delivery failures during festival periods, and how should we prepare?"
            
            # Create prompt for OpenAI
            prompt = self._create_festival_prompt(query, context_data)
            
            # Call OpenAI API with GPT-4 for better insights
            response = self.openai_client.chat.completions.create(
                model="gpt-4",  # Using GPT-4 for more sophisticated analysis
                messages=[
                    {"role": "system", "content": "You are a logistics analytics expert providing insights about delivery operations during festival periods. Provide data-driven, specific insights and actionable recommendations."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Lower temperature for more focused and consistent outputs
                max_tokens=1000   # Increased token limit for more comprehensive analysis
            )
            
            # Extract response
            result = response.choices[0].message.content
            
            # Parse the response to extract insights and recommendations
            return self._parse_festival_insights(result)
            
        except Exception as e:
            logger.error(f"Error generating festival insights with OpenAI: {str(e)}")
            # Fall back to dynamic insights
            return self._generate_dynamic_festival_insights()
    
    def _create_festival_prompt(self, query: str, context_data: str) -> str:
        """
        Create a prompt for OpenAI to generate festival period insights.
        
        Args:
            query: The user query
            context_data: Context data for analysis
            
        Returns:
            str: Formatted prompt
        """
        return f"""User Query: {query}

Context Data:
{context_data}

Please provide a comprehensive analysis with:
1. Executive summary
2. Key findings about delivery failures during festival periods
3. Root cause analysis of delivery issues during peak seasons
4. Actionable recommendations for logistics operations
5. Next steps for implementation

Format your response with clear sections for 'insights' and 'recommendations'. 
Make insights data-driven and specific to the logistics domain.
Ensure recommendations are practical, actionable, and prioritized."""
    
    def _get_festival_context_data(self) -> str:
        """
        Get relevant data for festival period analysis.
        
        Returns:
            str: Context data as string
        """
        context_parts = []
        
        # Add general statistics if available
        if self.data_loaded and self.correlator and self.correlator.correlated_data:
            # Get comprehensive order data
            order_data = self.correlator.correlated_data.get('order_comprehensive')
            if order_data is not None and not order_data.empty:
                # Calculate basic statistics
                total_orders = len(order_data)
                failed_orders = len(order_data[order_data['status'] == 'Failed']) if 'status' in order_data.columns else 0
                delayed_orders = len(order_data[order_data['is_delayed'] == True]) if 'is_delayed' in order_data.columns else 0
                failure_rate = (failed_orders / total_orders * 100) if total_orders > 0 else 0
                delay_rate = (delayed_orders / total_orders * 100) if total_orders > 0 else 0
                
                context_parts.append(f"DELIVERY METRICS:")
                context_parts.append(f"Total orders analyzed: {total_orders}")
                context_parts.append(f"Failed orders: {failed_orders} ({failure_rate:.2f}%)")
                context_parts.append(f"Delayed orders: {delayed_orders} ({delay_rate:.2f}%)")
                
                # Add status distribution
                if 'status' in order_data.columns:
                    status_counts = order_data['status'].value_counts().to_dict()
                    context_parts.append(f"\nSTATUS DISTRIBUTION:")
                    for status, count in status_counts.items():
                        context_parts.append(f"{status}: {count} orders ({count/total_orders*100:.2f}%)")
                
                # Add external factors if available
                if all(col in order_data.columns for col in ['has_traffic', 'has_bad_weather', 'has_event']):
                    context_parts.append(f"\nEXTERNAL FACTORS IMPACT:")
                    traffic_impact = order_data['has_traffic'].mean() * 100
                    weather_impact = order_data['has_bad_weather'].mean() * 100
                    event_impact = order_data['has_event'].mean() * 100
                    
                    context_parts.append(f"Traffic congestion: {traffic_impact:.2f}% of deliveries affected")
                    context_parts.append(f"Adverse weather: {weather_impact:.2f}% of deliveries affected")
                    context_parts.append(f"Local events: {event_impact:.2f}% of deliveries affected")
                
                # Add failure reasons if available
                if 'failure_reason' in order_data.columns and failed_orders > 0:
                    failure_reasons = order_data[order_data['status'] == 'Failed']['failure_reason'].value_counts().head(5).to_dict()
                    if failure_reasons:
                        context_parts.append(f"\nTOP FAILURE REASONS:")
                        for reason, count in failure_reasons.items():
                            context_parts.append(f"{reason}: {count} orders ({count/failed_orders*100:.2f}% of failures)")
        
        # Add historical festival period data
        context_parts.append(f"\nHISTORICAL FESTIVAL PERIOD DATA:")
        context_parts.append("Festival periods historically show 30-40% higher order volumes compared to regular periods.")
        context_parts.append("Previous festival periods showed a 25% increase in delivery failures and a 35% increase in delays.")
        context_parts.append("Traffic congestion typically increases by 45-60% during major festivals in urban areas.")
        context_parts.append("Customer availability decreases by approximately 20% during holiday seasons.")
        context_parts.append("Inventory stockouts increase by 35% during festival periods without proper planning.")
        context_parts.append("Weather-related delivery issues increase by 15-30% during monsoon festivals.")
        
        return "\n".join(context_parts)
    
    def _generate_city_insights_with_openai(self, city_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate city-specific insights using OpenAI.
        
        Args:
            city_results: City analysis results
            
        Returns:
            Dict[str, Any]: Generated insights
        """
        # Extract city name and data
        city = city_results.get('city', 'Unknown')
        query = f"What are the key insights and recommendations for delivery operations in {city}?"
        
        # Create context data from city results
        context_data = self._get_city_context_data(city_results)
        
        # Generate insights with OpenAI
        system_prompt = f"You are a logistics analytics expert providing insights about delivery operations in {city}."
        return self._generate_insights_with_openai(query, context_data, system_prompt)
    
    def _generate_client_insights_with_openai(self, client_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate client-specific insights using OpenAI.
        
        Args:
            client_results: Client analysis results
            
        Returns:
            Dict[str, Any]: Generated insights
        """
        # Extract client name/id and data
        client_name = client_results.get('client_name', 'Unknown')
        client_id = client_results.get('client_id', 'Unknown')
        client_identifier = client_name if client_name != 'Unknown' else f"Client ID {client_id}"
        
        query = f"What are the key insights and recommendations for delivery operations for {client_identifier}?"
        
        # Create context data from client results
        context_data = self._get_client_context_data(client_results)
        
        # Generate insights with OpenAI
        system_prompt = f"You are a logistics analytics expert providing insights about delivery operations for {client_identifier}."
        return self._generate_insights_with_openai(query, context_data, system_prompt)
    
    def _generate_warehouse_insights_with_openai(self, warehouse_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate warehouse-specific insights using OpenAI.
        
        Args:
            warehouse_results: Warehouse analysis results
            
        Returns:
            Dict[str, Any]: Generated insights
        """
        # Extract warehouse name/id and data
        warehouse_name = warehouse_results.get('warehouse_name', 'Unknown')
        warehouse_id = warehouse_results.get('warehouse_id', 'Unknown')
        warehouse_identifier = warehouse_name if warehouse_name != 'Unknown' else f"Warehouse ID {warehouse_id}"
        
        query = f"What are the key insights and recommendations for operations at {warehouse_identifier}?"
        
        # Create context data from warehouse results
        context_data = self._get_warehouse_context_data(warehouse_results)
        
        # Generate insights with OpenAI
        system_prompt = f"You are a logistics analytics expert providing insights about operations at {warehouse_identifier}."
        return self._generate_insights_with_openai(query, context_data, system_prompt)
    
    def _generate_comparison_insights_with_openai(self, comparison_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comparison insights using OpenAI.
        
        Args:
            comparison_results: Comparison analysis results
            
        Returns:
            Dict[str, Any]: Generated insights
        """
        # Extract cities and check data availability
        cities = comparison_results.get('cities', ['Unknown', 'Unknown'])
        city1_available = comparison_results.get('city1_data_available', False)
        city2_available = comparison_results.get('city2_data_available', False)
        
        # If data is missing for either city, return None to use the standard approach
        if not city1_available or not city2_available:
            return None
        
        query = f"Compare delivery performance between {cities[0]} and {cities[1]} and provide insights and recommendations."
        
        # Create context data from comparison results
        context_data = self._get_comparison_context_data(comparison_results)
        
        # Generate insights with OpenAI
        system_prompt = f"You are a logistics analytics expert comparing delivery operations between {cities[0]} and {cities[1]}."
        return self._generate_insights_with_openai(query, context_data, system_prompt)
    
    def _get_comparison_context_data(self, comparison_results: Dict[str, Any]) -> str:
        """
        Extract context data from comparison results.
        
        Args:
            comparison_results: Comparison analysis results
            
        Returns:
            str: Context data as string
        """
        context_parts = []
        
        # Extract basic comparison information
        cities = comparison_results.get('cities', ['Unknown', 'Unknown'])
        order_counts = comparison_results.get('order_counts', [0, 0])
        
        context_parts.append(f"Comparing: {cities[0]} vs {cities[1]}")
        context_parts.append(f"Order counts: {cities[0]} ({order_counts[0]} orders), {cities[1]} ({order_counts[1]} orders)")
        
        # Add failure rates comparison
        failure_rates = comparison_results.get('failure_rates', [0, 0])
        failure_diff = comparison_results.get('failure_rate_difference', 0)
        
        context_parts.append(f"Failure rates: {cities[0]} ({failure_rates[0]:.2f}%), {cities[1]} ({failure_rates[1]:.2f}%)")
        context_parts.append(f"Failure rate difference: {abs(failure_diff):.2f}% {'higher' if failure_diff > 0 else 'lower'} in {cities[0]} compared to {cities[1]}")
        
        # Add delay rates comparison
        delay_rates = comparison_results.get('delay_rates', [0, 0])
        delay_diff = comparison_results.get('delay_rate_difference', 0)
        
        context_parts.append(f"Delay rates: {cities[0]} ({delay_rates[0]:.2f}%), {cities[1]} ({delay_rates[1]:.2f}%)")
        context_parts.append(f"Delay rate difference: {abs(delay_diff):.2f}% {'higher' if delay_diff > 0 else 'lower'} in {cities[0]} compared to {cities[1]}")
        
        # Add city-specific metrics
        city1_metrics = comparison_results.get('city1_metrics', {})
        city2_metrics = comparison_results.get('city2_metrics', {})
        
        # Add failure categories for each city
        for idx, (city, metrics) in enumerate([(cities[0], city1_metrics), (cities[1], city2_metrics)]):
            if 'failure_metrics' in metrics and 'categories' in metrics['failure_metrics']:
                categories = metrics['failure_metrics']['categories']
                context_parts.append(f"\n{city} failure categories:")
                for category, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
                    if count > 0:
                        context_parts.append(f"- {category}: {count} orders")
        
        # Add external factors comparison if available
        comparison_metrics = comparison_results.get('comparison_metrics', {})
        if 'external_factor_comparison' in comparison_metrics:
            factor_comp = comparison_metrics['external_factor_comparison']
            context_parts.append("\nExternal factors comparison:")
            
            for factor in ['traffic_impact', 'weather_impact', 'event_impact']:
                diff_key = f'{factor}_diff'
                city_key = f'higher_{factor}_city'
                
                if diff_key in factor_comp and city_key in factor_comp:
                    diff = factor_comp[diff_key]
                    higher_city = factor_comp[city_key]
                    lower_city = cities[0] if higher_city == cities[1] else cities[1]
                    
                    context_parts.append(f"- {factor.replace('_', ' ').title()}: {abs(diff):.2f}% higher in {higher_city} than in {lower_city}")
        
        return "\n".join(context_parts)
    
    def _get_warehouse_context_data(self, warehouse_results: Dict[str, Any]) -> str:
        """
        Extract context data from warehouse results.
        
        Args:
            warehouse_results: Warehouse analysis results
            
        Returns:
            str: Context data as string
        """
        context_parts = []
        
        # Add basic warehouse information
        warehouse_name = warehouse_results.get('warehouse_name', 'Unknown')
        warehouse_id = warehouse_results.get('warehouse_id', 'Unknown')
        order_count = warehouse_results.get('order_count', 0)
        
        context_parts.append(f"Warehouse Name: {warehouse_name}")
        context_parts.append(f"Warehouse ID: {warehouse_id}")
        context_parts.append(f"Total orders processed: {order_count}")
        
        # Add capacity information
        capacity = warehouse_results.get('capacity', {})
        if capacity:
            total_capacity = capacity.get('total_capacity', 0)
            current_usage = capacity.get('current_usage', 0)
            utilization_rate = capacity.get('utilization_rate', 0)
            
            context_parts.append(f"Total capacity: {total_capacity} units")
            context_parts.append(f"Current usage: {current_usage} units")
            context_parts.append(f"Utilization rate: {utilization_rate:.2f}%")
        
        # Add performance metrics
        performance = warehouse_results.get('performance', {})
        if performance:
            avg_processing_time = performance.get('avg_processing_time', 0)
            avg_dispatch_time = performance.get('avg_dispatch_time', 0)
            error_rate = performance.get('error_rate', 0)
            
            context_parts.append(f"Average processing time: {avg_processing_time:.2f} hours")
            context_parts.append(f"Average dispatch time: {avg_dispatch_time:.2f} hours")
            context_parts.append(f"Error rate: {error_rate:.2f}%")
        
        # Add inventory information
        inventory = warehouse_results.get('inventory', {})
        if inventory:
            stockout_rate = inventory.get('stockout_rate', 0)
            turnover_rate = inventory.get('turnover_rate', 0)
            
            context_parts.append(f"Stockout rate: {stockout_rate:.2f}%")
            context_parts.append(f"Inventory turnover rate: {turnover_rate:.2f}")
            
            # Add top items with stockouts
            stockout_items = inventory.get('stockout_items', {})
            if stockout_items:
                context_parts.append("Top items with stockouts:")
                for item, count in sorted(stockout_items.items(), key=lambda x: x[1], reverse=True)[:5]:
                    context_parts.append(f"- {item}: {count} occurrences")
        
        # Add issue information
        issues = warehouse_results.get('issues', {})
        if issues:
            common_issues = issues.get('common_issues', {})
            if common_issues:
                context_parts.append("Common operational issues:")
                for issue, count in sorted(common_issues.items(), key=lambda x: x[1], reverse=True)[:5]:
                    context_parts.append(f"- {issue}: {count} occurrences")
        
        return "\n".join(context_parts)
    
    def _get_client_context_data(self, client_results: Dict[str, Any]) -> str:
        """
        Extract context data from client results.
        
        Args:
            client_results: Client analysis results
            
        Returns:
            str: Context data as string
        """
        context_parts = []
        
        # Add basic client information
        client_name = client_results.get('client_name', 'Unknown')
        client_id = client_results.get('client_id', 'Unknown')
        order_count = client_results.get('order_count', 0)
        
        context_parts.append(f"Client Name: {client_name}")
        context_parts.append(f"Client ID: {client_id}")
        context_parts.append(f"Total orders: {order_count}")
        
        # Add delivery performance information
        performance = client_results.get('delivery_performance', {})
        if performance:
            on_time_rate = performance.get('on_time_rate', 0)
            late_rate = performance.get('late_rate', 0)
            failed_rate = performance.get('failed_rate', 0)
            
            context_parts.append(f"On-time delivery rate: {on_time_rate:.2f}%")
            context_parts.append(f"Late delivery rate: {late_rate:.2f}%")
            context_parts.append(f"Failed delivery rate: {failed_rate:.2f}%")
        
        # Add order patterns information
        patterns = client_results.get('order_patterns', {})
        if patterns:
            avg_order_size = patterns.get('avg_order_size', 0)
            avg_order_value = patterns.get('avg_order_value', 0)
            peak_day = patterns.get('peak_order_day', 'Unknown')
            peak_time = patterns.get('peak_order_time', 'Unknown')
            
            context_parts.append(f"Average order size: {avg_order_size:.2f} items")
            context_parts.append(f"Average order value: ${avg_order_value:.2f}")
            context_parts.append(f"Peak order day: {peak_day}")
            context_parts.append(f"Peak order time: {peak_time}")
        
        # Add issue information
        issues = client_results.get('issues', {})
        if issues:
            common_issues = issues.get('common_issues', {})
            if common_issues:
                context_parts.append("Common issues:")
                for issue, count in sorted(common_issues.items(), key=lambda x: x[1], reverse=True)[:5]:
                    context_parts.append(f"- {issue}: {count} occurrences")
        
        # Add feedback information
        feedback = client_results.get('feedback', {})
        if feedback:
            avg_rating = feedback.get('avg_rating', 0)
            positive_count = feedback.get('positive_count', 0)
            negative_count = feedback.get('negative_count', 0)
            
            context_parts.append(f"Average rating: {avg_rating:.2f}/5.0")
            context_parts.append(f"Positive feedback: {positive_count} instances")
            context_parts.append(f"Negative feedback: {negative_count} instances")
            
            # Add common feedback themes
            themes = feedback.get('common_themes', {})
            if themes:
                context_parts.append("Common feedback themes:")
                for theme, count in sorted(themes.items(), key=lambda x: x[1], reverse=True)[:5]:
                    context_parts.append(f"- {theme}: {count} mentions")
        
        return "\n".join(context_parts)
    
    def _get_city_context_data(self, city_results: Dict[str, Any]) -> str:
        """
        Extract context data from city results.
        
        Args:
            city_results: City analysis results
            
        Returns:
            str: Context data as string
        """
        context_parts = []
        
        # Add basic city information
        city = city_results.get('city', 'Unknown')
        order_count = city_results.get('order_count', 0)
        context_parts.append(f"City: {city}")
        context_parts.append(f"Total orders: {order_count}")
        
        # Add delivery failure information
        failure_data = city_results.get('delivery_failures', {})
        if failure_data:
            failure_count = failure_data.get('failure_count', 0)
            failure_rate = failure_data.get('failure_rate', 0)
            context_parts.append(f"Failed orders: {failure_count}")
            context_parts.append(f"Failure rate: {failure_rate:.2f}%")
            
            # Add failure reasons if available
            failure_reasons = failure_data.get('failure_reasons', {})
            if failure_reasons:
                categories = failure_reasons.get('categories', {})
                if categories:
                    context_parts.append("Failure categories:")
                    for category, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
                        context_parts.append(f"- {category}: {count} orders")
                
                counts = failure_reasons.get('counts', {})
                if counts:
                    context_parts.append("Specific failure reasons:")
                    for reason, count in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                        context_parts.append(f"- {reason}: {count} orders")
        
        # Add delivery delay information
        delay_data = city_results.get('delivery_delays', {})
        if delay_data:
            delay_count = delay_data.get('delay_count', 0)
            delay_rate = delay_data.get('delay_rate', 0)
            context_parts.append(f"Delayed orders: {delay_count}")
            context_parts.append(f"Delay rate: {delay_rate:.2f}%")
            
            # Add delay patterns if available
            delay_patterns = delay_data.get('delay_patterns', {})
            if delay_patterns:
                statistics = delay_patterns.get('statistics', {})
                if statistics:
                    mean_delay = statistics.get('mean_delay_hours', 0)
                    max_delay = statistics.get('max_delay_hours', 0)
                    context_parts.append(f"Average delay: {mean_delay:.2f} hours")
                    context_parts.append(f"Maximum delay: {max_delay:.2f} hours")
                
                by_day = delay_patterns.get('by_day_of_week', {})
                if by_day:
                    context_parts.append("Delays by day of week:")
                    for day, hours in sorted(by_day.items(), key=lambda x: x[1], reverse=True):
                        context_parts.append(f"- {day}: {hours:.2f} hours")
        
        # Add external factors if available
        external_factors = city_results.get('external_factors', {})
        if external_factors:
            context_parts.append("External factors impact:")
            for factor, impact in external_factors.items():
                context_parts.append(f"- {factor}: {impact:.2f}%")
        
        return "\n".join(context_parts)
    
    def _parse_festival_insights(self, response: str) -> Dict[str, Any]:
        """
        Parse the OpenAI response to extract insights and recommendations.
        
        Args:
            response: Raw response from OpenAI
            
        Returns:
            Dict[str, Any]: Structured insights and recommendations
        """
        insights = []
        recommendations = []
        
        # Simple parsing based on sections
        current_section = None
        for line in response.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            # Check for section headers
            lower_line = line.lower()
            if 'insight' in lower_line and (':' in line or line.endswith('insights')):
                current_section = 'insights'
                continue
            elif 'recommend' in lower_line and (':' in line or line.endswith('recommendations')):
                current_section = 'recommendations'
                continue
                
            # Add content to appropriate section
            if current_section == 'insights':
                # Remove bullet points or numbering
                cleaned_line = re.sub(r'^[\d\s\-\*\.]+', '', line).strip()
                if cleaned_line:
                    insights.append(cleaned_line)
            elif current_section == 'recommendations':
                # Remove bullet points or numbering
                cleaned_line = re.sub(r'^[\d\s\-\*\.]+', '', line).strip()
                if cleaned_line:
                    recommendations.append(cleaned_line)
        
        # If parsing failed to find sections, use simple heuristics
        if not insights and not recommendations:
            # Split response roughly in half for insights and recommendations
            lines = [line.strip() for line in response.split('\n') if line.strip()]
            midpoint = len(lines) // 2
            
            for i, line in enumerate(lines):
                # Remove bullet points or numbering
                cleaned_line = re.sub(r'^[\d\s\-\*\.]+', '', line).strip()
                if not cleaned_line:
                    continue
                    
                if i < midpoint:
                    insights.append(cleaned_line)
                else:
                    recommendations.append(cleaned_line)
        
        # Ensure we have at least some insights and recommendations by generating dynamic fallbacks
        if not insights:
            # Generate dynamic insights based on available data
            context_data = self._get_festival_context_data()
            if "DELIVERY METRICS" in context_data:
                insights = [f"Analysis of available data indicates potential delivery challenges during festival periods."]
            else:
                insights = [f"Festival periods require special attention to logistics operations based on historical patterns."]
            
        if not recommendations:
            # Generate dynamic recommendations
            recommendations = [
                f"Develop a data-driven strategy for handling increased order volumes during festival periods.",
                f"Implement contingency plans for common festival-related delivery challenges."
            ]
        
        return {
            'insights': insights,
            'recommendations': recommendations
        }
    
    def _generate_dynamic_festival_insights(self) -> Dict[str, Any]:
        """
        Generate dynamic insights for festival period queries as a fallback when OpenAI is not available.
        Uses available data to generate contextual insights rather than hardcoded values.
        
        Returns:
            Dict[str, Any]: Generated insights
        """
        # Get available data for context
        context_data = self._get_festival_context_data()
        
        # Create a basic query for analysis
        query = "What are the likely causes of delivery failures during festival periods, and how should we prepare?"
        
        # Try to use the generic insights generator with the available data
        insights_result = self._generate_insights_with_openai(query, context_data, 
            system_prompt="You are a logistics analytics expert providing insights about delivery operations during festival periods.")
        
        # If the generic insights generator fails, use minimal fallback insights
        if not insights_result:
            insights = [
                "Analysis of historical data suggests increased delivery challenges during festival periods.",
                "Multiple factors including traffic, weather, and inventory affect festival period deliveries."
            ]
            
            recommendations = [
                "Prepare additional capacity for peak festival periods based on historical data.",
                "Develop contingency plans for common festival period delivery challenges."
            ]
            
            return {
                'insights': insights,
                'recommendations': recommendations
            }
        
        return insights_result
    
    def _generate_insights_with_openai(self, query: str, context_data: str, system_prompt: str = None) -> Dict[str, Any]:
        """
        Generate insights using OpenAI API.
        
        Args:
            query: The user query
            context_data: Context data for analysis
            system_prompt: Custom system prompt (optional)
            
        Returns:
            Dict[str, Any]: Generated insights
        """
        # Check if OpenAI client is available
        if not self.openai_client:
            return None
        
        try:
            # Set default system prompt if not provided
            if not system_prompt:
                system_prompt = "You are a logistics analytics expert providing insights about delivery operations."
            
            # Create prompt for OpenAI using the Java sample format
            prompt = f"""User Query: {query}

Context Data:
{context_data}

Please provide a comprehensive analysis with:
1. Executive summary
2. Key findings
3. Root cause analysis (if applicable)
4. Actionable recommendations
5. Next steps

Format your response with clear sections for 'insights' and 'recommendations'.
Make insights data-driven and specific to the logistics domain.
Ensure recommendations are practical, actionable, and prioritized."""
            
            # Call OpenAI API with improved parameters
            response = self.openai_client.chat.completions.create(
                model="gpt-4",  # Using GPT-4 for more sophisticated analysis
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Lower temperature for more focused and consistent outputs
                max_tokens=1000   # Increased token limit for more comprehensive analysis
            )
            
            # Extract response
            result = response.choices[0].message.content
            
            # Parse the response to extract insights and recommendations
            return self._parse_openai_insights(result)
            
        except Exception as e:
            logger.error(f"Error generating insights with OpenAI: {str(e)}")
            return None
    
    def _parse_openai_insights(self, response: str) -> Dict[str, Any]:
        """
        Parse the OpenAI response to extract insights and recommendations.
        
        Args:
            response: Raw response from OpenAI
            
        Returns:
            Dict[str, Any]: Structured insights and recommendations
        """
        insights = []
        recommendations = []
        
        # Simple parsing based on sections
        current_section = None
        for line in response.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            # Check for section headers
            lower_line = line.lower()
            if 'insight' in lower_line and (':' in line or line.endswith('insights')):
                current_section = 'insights'
                continue
            elif 'recommend' in lower_line and (':' in line or line.endswith('recommendations')):
                current_section = 'recommendations'
                continue
                
            # Add content to appropriate section
            if current_section == 'insights':
                # Remove bullet points or numbering
                cleaned_line = re.sub(r'^[\d\s\-\*\.]+', '', line).strip()
                if cleaned_line:
                    insights.append(cleaned_line)
            elif current_section == 'recommendations':
                # Remove bullet points or numbering
                cleaned_line = re.sub(r'^[\d\s\-\*\.]+', '', line).strip()
                if cleaned_line:
                    recommendations.append(cleaned_line)
        
        # If parsing failed to find sections, use simple heuristics
        if not insights and not recommendations:
            # Split response roughly in half for insights and recommendations
            lines = [line.strip() for line in response.split('\n') if line.strip()]
            midpoint = len(lines) // 2
            
            for i, line in enumerate(lines):
                # Remove bullet points or numbering
                cleaned_line = re.sub(r'^[\d\s\-\*\.]+', '', line).strip()
                if not cleaned_line:
                    continue
                    
                if i < midpoint:
                    insights.append(cleaned_line)
                else:
                    recommendations.append(cleaned_line)
        
        return {
            'insights': insights,
            'recommendations': recommendations
        }
    
    def _generate_insights(self, query_type: str, results: Dict[str, Any], additional_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate insights based on query type and results.
        
        Args:
            query_type: Type of query
            results: Analysis results
            additional_params: Additional parameters from the query
            
        Returns:
            Dict[str, Any]: Generated insights
        """
        # Initialize additional_params if None
        if additional_params is None:
            additional_params = {}
            
        # Get special period if any
        special_period = additional_params.get('special_period')
        if 'error' in results:
            return {"error": results['error']}
        
        # Check for city-specific results
        if 'city' in results:
            # Try to generate insights with OpenAI first
            city_insights = self._generate_city_insights_with_openai(results)
            if city_insights:
                return city_insights
            # Fall back to standard insights generator if OpenAI fails
            return self.insights_generator.generate_city_insights(results)
        
        # Check for client-specific results
        elif 'client_id' in results or 'client_name' in results:
            # Try to generate insights with OpenAI first
            client_insights = self._generate_client_insights_with_openai(results)
            if client_insights:
                return client_insights
            # Fall back to standard insights generator if OpenAI fails
            return self.insights_generator.generate_client_insights(results)
        
        # Check for warehouse-specific results
        elif 'warehouse_id' in results or 'warehouse_name' in results:
            # Try to generate insights with OpenAI first
            warehouse_insights = self._generate_warehouse_insights_with_openai(results)
            if warehouse_insights:
                return warehouse_insights
            # Fall back to standard insights generator if OpenAI fails
            return self.insights_generator.generate_warehouse_insights(results)
        
        # Handle specific query types
        elif query_type == 'comparison':
            # Check if it's a detailed comparison
            if results.get('comparison_type') == 'detailed':
                # Try to generate insights with OpenAI first
                comparison_insights = self._generate_comparison_insights_with_openai(results)
                if comparison_insights:
                    return comparison_insights
                # Fall back to standard detailed comparison if OpenAI fails
                # Generate detailed insights
                cities = results.get('cities', ['Unknown', 'Unknown'])
                order_counts = results.get('order_counts', [0, 0])
                city1_available = results.get('city1_data_available', False)
                city2_available = results.get('city2_data_available', False)
                city1_metrics = results.get('city1_metrics', {})
                city2_metrics = results.get('city2_metrics', {})
                comparison_metrics = results.get('comparison_metrics', {})
                
                insights = []
                recommendations = []
                
                # Basic comparison info
                insights.append(f"Comparing {cities[0]} ({order_counts[0]} orders) and {cities[1]} ({order_counts[1]} orders):")
                
                # Handle missing data
                if not city1_available:
                    insights.append(f"No data available for {cities[0]}.")
                
                if not city2_available:
                    insights.append(f"No data available for {cities[1]}.")
                
                if not city1_available or not city2_available:
                    recommendations.append("Improve data collection for cities with missing data.")
                    return {
                        'insights': insights,
                        'recommendations': recommendations
                    }
                
                # Add delivery metrics insights
                if 'delivery_comparison' in comparison_metrics:
                    delivery_comp = comparison_metrics['delivery_comparison']
                    
                    # Delivery rate comparison
                    if 'delivery_rate_diff' in delivery_comp:
                        diff = abs(delivery_comp['delivery_rate_diff'])
                        better_city = delivery_comp.get('better_delivery_city')
                        worse_city = cities[0] if better_city == cities[1] else cities[1]
                        
                        if diff > 5:  # Only mention if difference is significant
                            insights.append(f"{better_city} has a {diff:.2f}% higher delivery success rate than {worse_city}.")
                    
                    # Failure rate comparison
                    if 'failure_rate_diff' in delivery_comp:
                        diff = abs(delivery_comp['failure_rate_diff'])
                        better_city = delivery_comp.get('better_failure_city')
                        worse_city = cities[0] if better_city == cities[1] else cities[1]
                        
                        if diff > 5:  # Only mention if difference is significant
                            insights.append(f"{better_city} has a {diff:.2f}% lower failure rate than {worse_city}.")
                            recommendations.append(f"Investigate why {worse_city} has a higher failure rate and apply lessons from {better_city}.")
                    
                    # Delay rate comparison
                    if 'delay_rate_diff' in delivery_comp:
                        diff = abs(delivery_comp['delay_rate_diff'])
                        better_city = delivery_comp.get('better_delay_city')
                        worse_city = cities[0] if better_city == cities[1] else cities[1]
                        
                        if diff > 5:  # Only mention if difference is significant
                            insights.append(f"{better_city} has a {diff:.2f}% lower delay rate than {worse_city}.")
                            recommendations.append(f"Review delivery processes in {worse_city} to reduce delays.")
                
                # Add external factors insights
                if 'external_factor_comparison' in comparison_metrics:
                    factor_comp = comparison_metrics['external_factor_comparison']
                    
                    # Traffic impact comparison
                    if 'traffic_impact_diff' in factor_comp:
                        diff = abs(factor_comp['traffic_impact_diff'])
                        higher_city = factor_comp.get('higher_traffic_impact_city')
                        
                        if diff > 10:  # Only mention if difference is significant
                            insights.append(f"Traffic has a {diff:.2f}% higher impact on deliveries in {higher_city}.")
                            recommendations.append(f"Optimize delivery routes and timing in {higher_city} to minimize traffic impact.")
                    
                    # Weather impact comparison
                    if 'weather_impact_diff' in factor_comp:
                        diff = abs(factor_comp['weather_impact_diff'])
                        higher_city = factor_comp.get('higher_weather_impact_city')
                        
                        if diff > 10:  # Only mention if difference is significant
                            insights.append(f"Weather conditions have a {diff:.2f}% higher impact on deliveries in {higher_city}.")
                            recommendations.append(f"Develop better contingency plans for adverse weather in {higher_city}.")
                
                # Add detailed failure analysis insights
                for city_idx, city_name in enumerate(cities):
                    city_m = city1_metrics if city_idx == 0 else city2_metrics
                    
                    if 'failure_metrics' in city_m and 'total_failures' in city_m['failure_metrics']:
                        total_failures = city_m['failure_metrics']['total_failures']
                        failure_rate = city_m['failure_metrics'].get('failure_rate', 0)
                        
                        # Add failure rate insight
                        insights.append(f"{city_name} has a failure rate of {failure_rate:.2f}% ({total_failures} failed orders).")
                        
                        # Add top failure categories
                        if 'categories' in city_m['failure_metrics']:
                            categories = city_m['failure_metrics']['categories']
                            top_categories = sorted([(cat, count) for cat, count in categories.items() if count > 0], 
                                                   key=lambda x: x[1], reverse=True)[:3]
                            
                            if top_categories:
                                category_text = ", ".join([f"{cat} ({count})" for cat, count in top_categories])
                                insights.append(f"Top failure categories in {city_name}: {category_text}.")
                        
                        # Add top specific reasons if available
                        if 'top_specific_reasons' in city_m['failure_metrics']:
                            top_reasons = city_m['failure_metrics']['top_specific_reasons']
                            if top_reasons:
                                reasons_text = ", ".join([f"\"{reason}\" ({count})" for reason, count in top_reasons])
                                insights.append(f"Top specific failure reasons in {city_name}: {reasons_text}.")
                
                # Compare failure rates between cities
                if 'failure_metrics' in city1_metrics and 'failure_metrics' in city2_metrics:
                    if 'failure_rate' in city1_metrics['failure_metrics'] and 'failure_rate' in city2_metrics['failure_metrics']:
                        rate1 = city1_metrics['failure_metrics']['failure_rate']
                        rate2 = city2_metrics['failure_metrics']['failure_rate']
                        diff = abs(rate1 - rate2)
                        
                        if diff > 2:  # Only if there's a significant difference
                            better_city = cities[0] if rate1 < rate2 else cities[1]
                            worse_city = cities[1] if better_city == cities[0] else cities[0]
                            insights.append(f"{better_city} has a {diff:.2f}% lower failure rate than {worse_city}.")
                            recommendations.append(f"Study the delivery processes in {better_city} to improve performance in {worse_city}.")
                
                # Compare top failure categories
                if 'failure_metrics' in city1_metrics and 'failure_metrics' in city2_metrics:
                    if 'top_category' in city1_metrics['failure_metrics'] and 'top_category' in city2_metrics['failure_metrics']:
                        top_cat1 = city1_metrics['failure_metrics']['top_category']
                        top_cat2 = city2_metrics['failure_metrics']['top_category']
                        
                        if top_cat1 != top_cat2:
                            insights.append(f"The main failure category differs between cities: {top_cat1} in {cities[0]} vs {top_cat2} in {cities[1]}.")
                            recommendations.append(f"Investigate why {top_cat1} issues are more prevalent in {cities[0]} and {top_cat2} issues in {cities[1]}.")
                        else:
                            insights.append(f"Both cities have the same main failure category: {top_cat1}.")
                            recommendations.append(f"Develop a coordinated strategy to address {top_cat1} issues across all locations.")
                
                return {
                    'insights': insights,
                    'recommendations': recommendations
                }
            else:
                # Use the standard comparison insights generator
                return self.insights_generator.generate_comparison_insights(results)
        
        elif query_type == 'prediction':
            return self.insights_generator.generate_risk_prediction_insights(results)
        
        # Handle festival period insights
        elif special_period == 'festival_period' or additional_params.get('special_period') == 'festival_period' or (isinstance(results, dict) and results.get('festival_period') == True):
            # Generate festival period insights
            insights = []
            recommendations = []
            
            # Add general insights about festival period
            insights.append("During festival periods, delivery failures tend to increase due to higher order volumes and external factors.")
            
            # Add specific insights based on available data
            if isinstance(results, dict) and 'delivery_failures' in results:
                failure_data = results['delivery_failures']
                
                # Extract failure categories if available
                if 'categories' in failure_data:
                    categories = failure_data['categories']
                    top_categories = sorted([(cat, count) for cat, count in categories.items() if count > 0], 
                                           key=lambda x: x[1], reverse=True)[:3]
                    
                    if top_categories:
                        category_text = ", ".join([f"{cat} ({count})" for cat, count in top_categories])
                        insights.append(f"Top failure categories during festival periods: {category_text}.")
                
                # Add insights about external factors
                if 'external_factors' in failure_data:
                    factors = failure_data['external_factors']
                    if 'traffic_impact' in factors and factors['traffic_impact'] > 30:
                        insights.append(f"Traffic congestion has a significant impact ({factors['traffic_impact']:.2f}%) on deliveries during festivals.")
                        recommendations.append("Plan alternative routes and delivery schedules to avoid peak traffic times during festivals.")
                    
                    if 'weather_impact' in factors and factors['weather_impact'] > 30:
                        insights.append(f"Weather conditions affect {factors['weather_impact']:.2f}% of deliveries during festival periods.")
                        recommendations.append("Develop contingency plans for adverse weather during festival seasons.")
            
            # Add general recommendations for festival periods
            recommendations.append("Increase delivery capacity by 20-30% during festival periods to handle higher order volumes.")
            recommendations.append("Implement a priority system for critical deliveries during peak periods.")
            recommendations.append("Communicate realistic delivery timelines to customers during festival periods.")
            
            return {
                'insights': insights,
                'recommendations': recommendations
            }
        
        # Default to general insights
        else:
            return self.insights_generator.generate_general_insights(results)
    
    def run_interactive(self) -> None:
        """Run the application in interactive mode."""
        self.ui.display_welcome()
        
        # Load data
        if not self.load_data():
            return
        
        while True:
            # Get user input
            user_input = self.ui.get_user_input("Query: ").strip()
            
            # Check for commands
            if user_input.lower() == 'exit':
                print("Exiting...")
                break
            
            elif user_input.lower() == 'help':
                self.ui.display_help()
                continue
            
            elif user_input.lower() == 'clear':
                self.ui.clear_screen()
                self.ui.display_welcome()
                continue
            
            elif user_input.lower() == 'reset':
                self.query_processor.reset_context()
                print("Context reset.")
                continue
            
            elif not user_input:
                continue
            
            # Process the query
            results = self.process_query(user_input)
            
            # Check for errors
            if 'error' in results:
                self.ui.display_error(results['error'])
                continue
            
            # Display insights
            self.ui.display_insights(results['insights'])
    
    def run_single_query(self, query: str) -> None:
        """
        Run a single query.
        
        Args:
            query: Query string
        """
        # Process the query
        results = self.process_query(query)
        
        # Check for errors
        if 'error' in results:
            self.ui.display_error(results['error'])
            return
        
        # Display insights
        self.ui.display_insights(results['insights'])


def main():
    """Main function."""
    # Create UI
    ui = ConsoleUI()
    
    # Parse arguments
    args = ui.parse_arguments()
    
    # Create engine
    engine = LogisticsInsightsEngine(args.data_dir, args.verbose)
    
    # Run in appropriate mode
    if args.query:
        engine.run_single_query(args.query)
    else:
        engine.run_interactive()


if __name__ == '__main__':
    main()
