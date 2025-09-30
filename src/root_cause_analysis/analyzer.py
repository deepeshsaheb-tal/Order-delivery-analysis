"""
Root cause analysis module for the logistics insights engine.
"""
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Set
from datetime import datetime, timedelta
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RootCauseAnalyzer:
    """
    Root cause analyzer class for identifying causes of delivery failures and delays.
    """
    
    def __init__(self, correlated_data: Dict[str, pd.DataFrame]):
        """
        Initialize the root cause analyzer.
        
        Args:
            correlated_data: Dictionary of correlated dataframes
        """
        self.correlated_data = correlated_data
        self.analysis_results: Dict[str, Any] = {}
    
    def analyze_delivery_failures(self, filtered_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Analyze delivery failures to identify root causes.
        
        Args:
            filtered_data: Optional filtered dataframe to analyze, 
                          if None, use the comprehensive order data
        
        Returns:
            Dict[str, Any]: Analysis results
        """
        if filtered_data is None:
            if 'order_comprehensive' not in self.correlated_data:
                logger.error("Comprehensive order data not available")
                return {}
            
            df = self.correlated_data['order_comprehensive']
        else:
            df = filtered_data
        
        # Filter for failed orders
        if 'status' not in df.columns:
            logger.error("Status column not available in data")
            return {'failure_count': 0}
            
        logger.info(f"Analyzing delivery failures in dataset with {len(df)} rows")
        logger.info(f"Status values in data: {df['status'].value_counts().to_dict()}")
        
        failed_orders = df[df['status'] == 'Failed']
        logger.info(f"Found {len(failed_orders)} failed orders")
        
        if failed_orders.empty:
            logger.info("No failed orders found in the data")
            return {'failure_count': 0}
        
        # Analyze failure reasons
        failure_reasons = self._analyze_failure_reasons(failed_orders)
        
        # Analyze external factors
        external_factors = self._analyze_external_factors(failed_orders)
        
        # Analyze warehouse issues
        warehouse_issues = self._analyze_warehouse_issues(failed_orders)
        
        # Analyze delivery issues
        delivery_issues = self._analyze_delivery_issues(failed_orders)
        
        # Analyze feedback for failed orders
        feedback_analysis = self._analyze_feedback(failed_orders)
        
        # Combine all analyses
        analysis_results = {
            'failure_count': len(failed_orders),
            'failure_rate': len(failed_orders) / len(df) if len(df) > 0 else 0,
            'failure_reasons': failure_reasons,
            'external_factors': external_factors,
            'warehouse_issues': warehouse_issues,
            'delivery_issues': delivery_issues,
            'feedback_analysis': feedback_analysis
        }
        
        self.analysis_results['delivery_failures'] = analysis_results
        return analysis_results
    
    def analyze_delivery_delays(self, filtered_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Analyze delivery delays to identify root causes.
        
        Args:
            filtered_data: Optional filtered dataframe to analyze, 
                          if None, use the comprehensive order data
        
        Returns:
            Dict[str, Any]: Analysis results
        """
        if filtered_data is None:
            if 'order_comprehensive' not in self.correlated_data:
                logger.error("Comprehensive order data not available")
                return {}
            
            df = self.correlated_data['order_comprehensive']
        else:
            df = filtered_data
        
        # Filter for delayed orders (delivered but late)
        delayed_orders = df[
            (df['status'] == 'Delivered') & 
            (df['is_late_delivery'] == 1)
        ]
        
        if delayed_orders.empty:
            logger.info("No delayed orders found in the data")
            return {'delay_count': 0}
        
        # Analyze delay patterns
        delay_patterns = self._analyze_delay_patterns(delayed_orders)
        
        # Analyze external factors for delays
        external_factors = self._analyze_external_factors(delayed_orders)
        
        # Analyze warehouse issues for delays
        warehouse_issues = self._analyze_warehouse_issues(delayed_orders)
        
        # Analyze delivery issues for delays
        delivery_issues = self._analyze_delivery_issues(delayed_orders)
        
        # Analyze feedback for delayed orders
        feedback_analysis = self._analyze_feedback(delayed_orders)
        
        # Combine all analyses
        analysis_results = {
            'delay_count': len(delayed_orders),
            'delay_rate': len(delayed_orders) / len(df) if len(df) > 0 else 0,
            'delay_patterns': delay_patterns,
            'external_factors': external_factors,
            'warehouse_issues': warehouse_issues,
            'delivery_issues': delivery_issues,
            'feedback_analysis': feedback_analysis
        }
        
        self.analysis_results['delivery_delays'] = analysis_results
        return analysis_results
    
    def _analyze_failure_reasons(self, failed_orders: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze reasons for order failures.
        
        Args:
            failed_orders: DataFrame of failed orders
            
        Returns:
            Dict[str, Any]: Analysis of failure reasons
        """
        # Count failure reasons
        failure_reason_counts = failed_orders['failure_reason'].value_counts().to_dict()
        
        # Calculate percentage for each reason
        total_failures = len(failed_orders)
        failure_reason_percentages = {
            reason: count / total_failures * 100 
            for reason, count in failure_reason_counts.items()
        }
        
        # Group failure reasons into categories
        failure_categories = self._categorize_failure_reasons(failed_orders)
        
        return {
            'counts': failure_reason_counts,
            'percentages': failure_reason_percentages,
            'categories': failure_categories
        }
    
    def _categorize_failure_reasons(self, failed_orders: pd.DataFrame) -> Dict[str, int]:
        """
        Categorize failure reasons into broader categories.
        
        Args:
            failed_orders: DataFrame of failed orders
            
        Returns:
            Dict[str, int]: Counts of failure categories
        """
        # Define categories and keywords
        categories = {
            'Inventory': ['stockout', 'out of stock', 'inventory'],
            'Address': ['address', 'location', 'gps', 'map'],
            'Customer': ['customer', 'recipient', 'not available', 'refused'],
            'Payment': ['payment', 'cod', 'cash', 'transaction'],
            'Logistics': ['vehicle', 'transport', 'breakdown', 'accident'],
            'Weather': ['weather', 'rain', 'storm', 'snow'],
            'Other': []  # Default category
        }
        
        # Initialize category counts
        category_counts = {category: 0 for category in categories}
        
        # Categorize each failure reason
        for _, row in failed_orders.iterrows():
            reason = str(row['failure_reason']).lower() if not pd.isna(row['failure_reason']) else ''
            
            # Find matching category
            matched = False
            for category, keywords in categories.items():
                if any(keyword in reason for keyword in keywords):
                    category_counts[category] += 1
                    matched = True
                    break
            
            # Use default category if no match found
            if not matched:
                category_counts['Other'] += 1
        
        return category_counts
    
    def _analyze_delay_patterns(self, delayed_orders: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze patterns in delivery delays.
        
        Args:
            delayed_orders: DataFrame of delayed orders
            
        Returns:
            Dict[str, Any]: Analysis of delay patterns
        """
        # Calculate delay statistics
        delay_stats = {
            'mean_delay_hours': delayed_orders['delivery_delay_hours'].mean(),
            'median_delay_hours': delayed_orders['delivery_delay_hours'].median(),
            'max_delay_hours': delayed_orders['delivery_delay_hours'].max(),
            'min_delay_hours': delayed_orders['delivery_delay_hours'].min()
        }
        
        # Group delays by severity
        delay_severity = {
            'minor_delay': len(delayed_orders[delayed_orders['delivery_delay_hours'] <= 24]),
            'moderate_delay': len(delayed_orders[(delayed_orders['delivery_delay_hours'] > 24) & 
                                              (delayed_orders['delivery_delay_hours'] <= 72)]),
            'severe_delay': len(delayed_orders[delayed_orders['delivery_delay_hours'] > 72])
        }
        
        # Analyze delays by day of week
        if 'order_date_dayofweek' in delayed_orders.columns:
            delays_by_day = delayed_orders.groupby('order_date_dayofweek')['delivery_delay_hours'].mean().to_dict()
            day_names = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 
                        4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
            delays_by_day = {day_names.get(day, day): delay for day, delay in delays_by_day.items()}
        else:
            delays_by_day = {}
        
        return {
            'statistics': delay_stats,
            'severity': delay_severity,
            'by_day_of_week': delays_by_day
        }
    
    def _analyze_external_factors(self, orders: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze the impact of external factors on orders.
        
        Args:
            orders: DataFrame of orders to analyze
            
        Returns:
            Dict[str, Any]: Analysis of external factors
        """
        # Check if external factor columns exist
        external_cols = ['has_external_factor', 'has_traffic_issue', 
                        'has_weather_issue', 'has_event_impact']
        
        missing_cols = [col for col in external_cols if col not in orders.columns]
        if missing_cols:
            logger.warning(f"Missing external factor columns: {missing_cols}")
            return {}
        
        # Calculate impact percentages
        total_orders = len(orders)
        if total_orders == 0:
            return {}
        
        impact_percentages = {
            'any_external_factor': orders['has_external_factor'].sum() / total_orders * 100,
            'traffic_issues': orders['has_traffic_issue'].sum() / total_orders * 100,
            'weather_issues': orders['has_weather_issue'].sum() / total_orders * 100,
            'event_impacts': orders['has_event_impact'].sum() / total_orders * 100
        }
        
        # Analyze specific conditions if available
        condition_analysis = {}
        
        if 'traffic_condition' in orders.columns:
            traffic_counts = orders['traffic_condition'].value_counts().to_dict()
            condition_analysis['traffic_conditions'] = traffic_counts
        
        if 'weather_condition' in orders.columns:
            weather_counts = orders['weather_condition'].value_counts().to_dict()
            condition_analysis['weather_conditions'] = weather_counts
        
        if 'event_type' in orders.columns:
            event_counts = orders['event_type'].value_counts().to_dict()
            condition_analysis['event_types'] = event_counts
        
        return {
            'impact_percentages': impact_percentages,
            'condition_analysis': condition_analysis
        }
    
    def _analyze_warehouse_issues(self, orders: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze warehouse-related issues for orders.
        
        Args:
            orders: DataFrame of orders to analyze
            
        Returns:
            Dict[str, Any]: Analysis of warehouse issues
        """
        # Check if warehouse columns exist
        warehouse_cols = ['warehouse_id', 'warehouse_name', 'picking_time_minutes',
                         'dispatch_prep_time_minutes', 'total_warehouse_time_minutes']
        
        missing_cols = [col for col in warehouse_cols if col not in orders.columns]
        if len(missing_cols) == len(warehouse_cols):
            logger.warning("No warehouse data available for analysis")
            return {}
        
        # Initialize results
        results = {}
        
        # Analyze warehouse processing times if available
        time_cols = ['picking_time_minutes', 'dispatch_prep_time_minutes', 'total_warehouse_time_minutes']
        available_time_cols = [col for col in time_cols if col in orders.columns]
        
        if available_time_cols:
            time_stats = {}
            for col in available_time_cols:
                if not orders[col].isna().all():
                    time_stats[col] = {
                        'mean': orders[col].mean(),
                        'median': orders[col].median(),
                        'max': orders[col].max()
                    }
            
            results['processing_times'] = time_stats
        
        # Analyze warehouse performance by warehouse if warehouse_id is available
        if 'warehouse_id' in orders.columns and not orders['warehouse_id'].isna().all():
            warehouse_performance = {}
            
            # Group by warehouse_id
            warehouse_groups = orders.groupby('warehouse_id')
            
            # Calculate metrics for each warehouse
            for warehouse_id, group in warehouse_groups:
                warehouse_name = group['warehouse_name'].iloc[0] if 'warehouse_name' in group.columns else f"Warehouse {warehouse_id}"
                
                metrics = {
                    'order_count': len(group)
                }
                
                # Add processing times if available
                for col in available_time_cols:
                    if not group[col].isna().all():
                        metrics[col] = group[col].mean()
                
                warehouse_performance[warehouse_name] = metrics
            
            results['warehouse_performance'] = warehouse_performance
        
        # Analyze warehouse notes if available
        if 'notes' in orders.columns and not orders['notes'].isna().all():
            # Count orders with notes
            orders_with_notes = orders[~orders['notes'].isna() & (orders['notes'] != '')]
            results['orders_with_notes'] = len(orders_with_notes)
            
            # Extract common keywords from notes
            if not orders_with_notes.empty:
                common_keywords = self._extract_keywords_from_text(orders_with_notes['notes'])
                results['common_note_keywords'] = common_keywords
        
        return results
    
    def _analyze_delivery_issues(self, orders: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze delivery-related issues for orders.
        
        Args:
            orders: DataFrame of orders to analyze
            
        Returns:
            Dict[str, Any]: Analysis of delivery issues
        """
        # Check if delivery columns exist
        delivery_cols = ['fleet_log_id', 'driver_id', 'vehicle_number', 'route_code',
                        'gps_delay_notes', 'transit_time_hours']
        
        missing_cols = [col for col in delivery_cols if col not in orders.columns]
        if len(missing_cols) == len(delivery_cols):
            logger.warning("No delivery data available for analysis")
            return {}
        
        # Initialize results
        results = {}
        
        # Analyze transit times if available
        if 'transit_time_hours' in orders.columns and not orders['transit_time_hours'].isna().all():
            transit_stats = {
                'mean': orders['transit_time_hours'].mean(),
                'median': orders['transit_time_hours'].median(),
                'max': orders['transit_time_hours'].max()
            }
            
            results['transit_times'] = transit_stats
        
        # Analyze driver performance if driver_id is available
        if 'driver_id' in orders.columns and not orders['driver_id'].isna().all():
            driver_performance = {}
            
            # Group by driver_id
            driver_groups = orders.groupby('driver_id')
            
            # Calculate metrics for each driver
            for driver_id, group in driver_groups:
                driver_name = group['driver_name'].iloc[0] if 'driver_name' in group.columns else f"Driver {driver_id}"
                
                metrics = {
                    'order_count': len(group)
                }
                
                # Add transit time if available
                if 'transit_time_hours' in group.columns and not group['transit_time_hours'].isna().all():
                    metrics['avg_transit_time'] = group['transit_time_hours'].mean()
                
                driver_performance[driver_name] = metrics
            
            results['driver_performance'] = driver_performance
        
        # Analyze GPS delay notes if available
        if 'gps_delay_notes' in orders.columns and not orders['gps_delay_notes'].isna().all():
            # Count orders with delay notes
            orders_with_notes = orders[~orders['gps_delay_notes'].isna() & (orders['gps_delay_notes'] != '')]
            results['orders_with_gps_notes'] = len(orders_with_notes)
            
            # Extract common keywords from notes
            if not orders_with_notes.empty:
                common_keywords = self._extract_keywords_from_text(orders_with_notes['gps_delay_notes'])
                results['common_gps_note_keywords'] = common_keywords
        
        # Analyze route performance if route_code is available
        if 'route_code' in orders.columns and not orders['route_code'].isna().all():
            route_performance = {}
            
            # Group by route_code
            route_groups = orders.groupby('route_code')
            
            # Calculate metrics for each route
            for route_code, group in route_groups:
                metrics = {
                    'order_count': len(group)
                }
                
                # Add transit time if available
                if 'transit_time_hours' in group.columns and not group['transit_time_hours'].isna().all():
                    metrics['avg_transit_time'] = group['transit_time_hours'].mean()
                
                route_performance[route_code] = metrics
            
            results['route_performance'] = route_performance
        
        return results
    
    def _analyze_feedback(self, orders: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze customer feedback for orders.
        
        Args:
            orders: DataFrame of orders to analyze
            
        Returns:
            Dict[str, Any]: Analysis of customer feedback
        """
        # Check if feedback columns exist
        feedback_cols = ['feedback_id', 'feedback_text', 'sentiment', 'rating',
                        'has_feedback', 'has_negative_feedback']
        
        missing_cols = [col for col in feedback_cols if col not in orders.columns]
        if len(missing_cols) == len(feedback_cols):
            logger.warning("No feedback data available for analysis")
            return {}
        
        # Initialize results
        results = {}
        
        # Calculate feedback coverage
        total_orders = len(orders)
        if total_orders == 0:
            return {}
        
        orders_with_feedback = orders[orders['has_feedback'] == 1] if 'has_feedback' in orders.columns else pd.DataFrame()
        feedback_coverage = len(orders_with_feedback) / total_orders * 100
        
        results['feedback_coverage'] = feedback_coverage
        
        # Analyze sentiment distribution if available
        if 'sentiment' in orders.columns and not orders_with_feedback.empty:
            sentiment_counts = orders_with_feedback['sentiment'].value_counts().to_dict()
            results['sentiment_distribution'] = sentiment_counts
        
        # Analyze rating distribution if available
        if 'rating' in orders.columns and not orders_with_feedback.empty:
            rating_counts = orders_with_feedback['rating'].value_counts().to_dict()
            results['rating_distribution'] = rating_counts
            
            # Calculate average rating
            results['average_rating'] = orders_with_feedback['rating'].mean()
        
        # Analyze feedback text if available
        if 'feedback_text' in orders.columns and not orders_with_feedback.empty:
            # Extract common keywords from feedback text
            common_keywords = self._extract_keywords_from_text(orders_with_feedback['feedback_text'])
            results['common_feedback_keywords'] = common_keywords
        
        return results
    
    def _extract_keywords_from_text(self, text_series: pd.Series, top_n: int = 10) -> Dict[str, int]:
        """
        Extract common keywords from a series of text.
        
        Args:
            text_series: Series of text strings
            top_n: Number of top keywords to return
            
        Returns:
            Dict[str, int]: Dictionary of keywords and their counts
        """
        # Combine all text
        combined_text = ' '.join(text_series.dropna().astype(str))
        
        # Tokenize and clean
        tokens = combined_text.lower().split()
        
        # Remove common stop words
        stop_words = {'the', 'and', 'is', 'in', 'it', 'to', 'that', 'of', 'for', 'on', 'with', 'as', 'this', 'by', 'a', 'an'}
        tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
        
        # Count keywords
        keyword_counts = Counter(tokens)
        
        # Get top keywords
        top_keywords = dict(keyword_counts.most_common(top_n))
        
        return top_keywords
    
    def analyze_city_performance(self, city: str, filtered_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Analyze delivery performance for a specific city.
        
        Args:
            city: City name to analyze
            filtered_data: Optional pre-filtered data to use for analysis (e.g., for specific time periods)
            
        Returns:
            Dict[str, Any]: Analysis results for the city
        """
        # Use filtered data if provided, otherwise use comprehensive data
        if filtered_data is not None:
            base_data = filtered_data
        else:
            if 'order_comprehensive' not in self.correlated_data:
                logger.error("Comprehensive order data not available")
                return {}
            base_data = self.correlated_data['order_comprehensive']
        
        # Check if city column exists
        if 'city' not in base_data.columns:
            logger.error("City column not available in data")
            return {}
        
        # Filter data for the specified city
        city_data = base_data[base_data['city'] == city]
        
        if city_data.empty:
            logger.warning(f"No data found for city: {city}")
            return {}
        
        # Analyze failures and delays for the city
        failure_analysis = self.analyze_delivery_failures(city_data)
        delay_analysis = self.analyze_delivery_delays(city_data)
        
        # Combine analyses
        city_analysis = {
            'city': city,
            'order_count': len(city_data),
            'delivery_failures': failure_analysis,
            'delivery_delays': delay_analysis
        }
        
        return city_analysis
    
    def analyze_client_performance(self, client_id: Optional[int] = None, client_name: Optional[str] = None, filtered_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Analyze delivery performance for a specific client.
        
        Args:
            client_id: Client ID to analyze, or None if using client_name
            client_name: Client name to analyze, or None if using client_id
            filtered_data: Optional pre-filtered data to use for analysis (e.g., for specific time periods)
            
        Returns:
            Dict[str, Any]: Analysis results for the client
        """
        # Use filtered data if provided, otherwise use comprehensive data
        if filtered_data is not None:
            base_data = filtered_data
        else:
            if 'order_comprehensive' not in self.correlated_data:
                logger.error("Comprehensive order data not available")
                return {}
            base_data = self.correlated_data['order_comprehensive']
        
        # Filter data for the specified client
        if client_id is not None:
            if 'client_id' not in base_data.columns:
                logger.error("Client ID column not available in data")
                return {}
            client_data = base_data[base_data['client_id'] == client_id]
        elif client_name is not None:
            if 'client_name' not in base_data.columns:
                logger.error("Client name column not available in data")
                return {}
            
            # Try exact match first
            client_data = base_data[base_data['client_name'] == client_name]
            
            # If no exact match, try case-insensitive match
            if client_data.empty:
                logger.info(f"No exact match for client name: {client_name}, trying case-insensitive match")
                client_data = base_data[base_data['client_name'].str.lower() == client_name.lower()]
            
            # If still no match, try partial match
            if client_data.empty:
                logger.info(f"No case-insensitive match for client name: {client_name}, trying partial match")
                client_data = base_data[base_data['client_name'].str.contains(client_name, case=False, na=False)]
            
            # If still no match, try removing special characters
            if client_data.empty:
                import re
                logger.info(f"No partial match for client name: {client_name}, trying with special characters removed")
                clean_client = re.sub(r'[^a-zA-Z0-9]', '', client_name)
                clean_data_names = base_data['client_name'].apply(lambda x: re.sub(r'[^a-zA-Z0-9]', '', str(x)))
                client_data = base_data[clean_data_names.str.contains(clean_client, case=False, na=False)]
        else:
            logger.error("Either client_id or client_name must be provided")
            return {}
        
        if client_data.empty:
            logger.warning(f"No data found for client: {client_id or client_name}")
            return {}
        
        # Get client name if available
        client_name_value = client_name
        if client_name_value is None and 'client_name' in client_data.columns:
            client_name_value = client_data['client_name'].iloc[0]
        
        # Analyze failures and delays for the client
        failure_analysis = self.analyze_delivery_failures(client_data)
        delay_analysis = self.analyze_delivery_delays(client_data)
        
        # Combine analyses
        client_analysis = {
            'client_id': client_id,
            'client_name': client_name_value,
            'order_count': len(client_data),
            'delivery_failures': failure_analysis,
            'delivery_delays': delay_analysis
        }
        
        return client_analysis
    
    def analyze_warehouse_performance(self, warehouse_id: Optional[int] = None, warehouse_name: Optional[str] = None, filtered_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Analyze delivery performance for a specific warehouse.
        
        Args:
            warehouse_id: Warehouse ID to analyze, or None if using warehouse_name
            warehouse_name: Warehouse name to analyze, or None if using warehouse_id
            filtered_data: Optional pre-filtered data to use for analysis (e.g., for specific time periods)
            
        Returns:
            Dict[str, Any]: Analysis results for the warehouse
        """
        logger.info(f"Analyzing warehouse performance: ID={warehouse_id}, Name={warehouse_name}, Filtered data provided: {filtered_data is not None}")
        
        # Use filtered data if provided, otherwise use comprehensive data
        if filtered_data is not None:
            base_data = filtered_data
            logger.info(f"Using provided filtered data with {len(base_data)} rows")
        else:
            if 'order_comprehensive' not in self.correlated_data:
                logger.error("Comprehensive order data not available")
                return {}
            base_data = self.correlated_data['order_comprehensive']
            logger.info(f"Using comprehensive data with {len(base_data)} rows")
        
        # Check if warehouse columns exist
        if 'warehouse_id' not in base_data.columns:
            logger.error("Warehouse ID column not available in data")
            return {}
        
        # Filter data for the specified warehouse
        if warehouse_id is not None:
            warehouse_data = base_data[base_data['warehouse_id'] == warehouse_id]
        elif warehouse_name is not None:
            if 'warehouse_name' not in base_data.columns:
                logger.error("Warehouse name column not available in data")
                return {}
            
            warehouse_data = base_data[base_data['warehouse_name'] == warehouse_name]
        else:
            logger.error("Either warehouse_id or warehouse_name must be provided")
            return {}
        
        if warehouse_data.empty:
            logger.warning(f"No data found for warehouse: {warehouse_id or warehouse_name}")
            return {}
        
        # Get warehouse name if available
        warehouse_name_value = warehouse_name
        if warehouse_name_value is None and 'warehouse_name' in warehouse_data.columns:
            warehouse_name_value = warehouse_data['warehouse_name'].iloc[0]
        
        # Analyze failures and delays for the warehouse
        failure_analysis = self.analyze_delivery_failures(warehouse_data)
        delay_analysis = self.analyze_delivery_delays(warehouse_data)
        
        # Analyze warehouse processing times
        warehouse_processing = self._analyze_warehouse_issues(warehouse_data)
        
        # Combine analyses
        warehouse_analysis = {
            'warehouse_id': warehouse_id,
            'warehouse_name': warehouse_name_value,
            'order_count': len(warehouse_data),
            'delivery_failures': failure_analysis,
            'delivery_delays': delay_analysis,
            'warehouse_processing': warehouse_processing
        }
        
        return warehouse_analysis
    
    def compare_cities(self, city1: str, city2: str) -> Dict[str, Any]:
        """
        Compare delivery performance between two cities.
        
        Args:
            city1: First city name to compare
            city2: Second city name to compare
            
        Returns:
            Dict[str, Any]: Comparison results for the cities
        """
        # Analyze each city
        city1_analysis = self.analyze_city_performance(city1)
        city2_analysis = self.analyze_city_performance(city2)
        
        # Check if analyses were successful
        if not city1_analysis or not city2_analysis:
            logger.error(f"Could not analyze one or both cities: {city1}, {city2}")
            return {}
        
        # Compare failure rates
        failure_rate_diff = (
            city1_analysis.get('delivery_failures', {}).get('failure_rate', 0) -
            city2_analysis.get('delivery_failures', {}).get('failure_rate', 0)
        )
        
        # Compare delay rates
        delay_rate_diff = (
            city1_analysis.get('delivery_delays', {}).get('delay_rate', 0) -
            city2_analysis.get('delivery_delays', {}).get('delay_rate', 0)
        )
        
        # Create comparison result
        comparison = {
            'cities': [city1, city2],
            'order_counts': [city1_analysis.get('order_count', 0), city2_analysis.get('order_count', 0)],
            'failure_rates': [
                city1_analysis.get('delivery_failures', {}).get('failure_rate', 0),
                city2_analysis.get('delivery_failures', {}).get('failure_rate', 0)
            ],
            'failure_rate_difference': failure_rate_diff,
            'delay_rates': [
                city1_analysis.get('delivery_delays', {}).get('delay_rate', 0),
                city2_analysis.get('delivery_delays', {}).get('delay_rate', 0)
            ],
            'delay_rate_difference': delay_rate_diff,
            'city1_analysis': city1_analysis,
            'city2_analysis': city2_analysis
        }
        
        return comparison
    
    def predict_risks(self, additional_orders: int = 0, client_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Predict delivery risks based on historical data and potential additional orders.
        
        Args:
            additional_orders: Number of additional orders to consider
            client_id: Optional client ID to focus the analysis on
            
        Returns:
            Dict[str, Any]: Risk prediction results
        """
        if 'order_comprehensive' not in self.correlated_data:
            logger.error("Comprehensive order data not available")
            return {}
        
        # Get base data for analysis
        if client_id is not None:
            base_data = self.correlated_data['order_comprehensive'][
                self.correlated_data['order_comprehensive']['client_id'] == client_id
            ]
            
            if base_data.empty:
                logger.warning(f"No data found for client ID: {client_id}")
                # Use all data as fallback
                base_data = self.correlated_data['order_comprehensive']
        else:
            base_data = self.correlated_data['order_comprehensive']
        
        # Calculate current metrics
        current_order_count = len(base_data)
        current_failure_rate = len(base_data[base_data['status'] == 'Failed']) / current_order_count if current_order_count > 0 else 0
        current_delay_rate = len(base_data[(base_data['status'] == 'Delivered') & (base_data['is_late_delivery'] == 1)]) / current_order_count if current_order_count > 0 else 0
        
        # Estimate new metrics with additional orders
        new_order_count = current_order_count + additional_orders
        
        # Simple linear projection (could be made more sophisticated)
        projected_failure_count = int(current_failure_rate * new_order_count)
        projected_delay_count = int(current_delay_rate * new_order_count)
        
        # Identify risk factors from historical data
        risk_factors = self._identify_risk_factors(base_data)
        
        # Predict potential bottlenecks
        bottlenecks = self._predict_bottlenecks(base_data, additional_orders)
        
        # Combine results
        risk_prediction = {
            'current_metrics': {
                'order_count': current_order_count,
                'failure_rate': current_failure_rate * 100,  # Convert to percentage
                'delay_rate': current_delay_rate * 100  # Convert to percentage
            },
            'projected_metrics': {
                'order_count': new_order_count,
                'failure_count': projected_failure_count,
                'delay_count': projected_delay_count,
                'failure_rate': (projected_failure_count / new_order_count) * 100 if new_order_count > 0 else 0,
                'delay_rate': (projected_delay_count / new_order_count) * 100 if new_order_count > 0 else 0
            },
            'risk_factors': risk_factors,
            'bottlenecks': bottlenecks
        }
        
        return risk_prediction
    
    def _identify_risk_factors(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Identify key risk factors from historical data.
        
        Args:
            data: DataFrame of historical order data
            
        Returns:
            List[Dict[str, Any]]: List of identified risk factors
        """
        risk_factors = []
        
        # Check for warehouse capacity issues
        if 'warehouse_id' in data.columns and 'warehouse_name' in data.columns:
            warehouse_groups = data.groupby(['warehouse_id', 'warehouse_name'])
            
            for (warehouse_id, warehouse_name), group in warehouse_groups:
                failure_rate = len(group[group['status'] == 'Failed']) / len(group) if len(group) > 0 else 0
                
                if failure_rate > 0.1:  # 10% threshold
                    risk_factors.append({
                        'type': 'warehouse',
                        'warehouse_id': warehouse_id,
                        'warehouse_name': warehouse_name,
                        'failure_rate': failure_rate * 100,
                        'order_count': len(group)
                    })
        
        # Check for route issues
        if 'route_code' in data.columns:
            route_groups = data.groupby('route_code')
            
            for route_code, group in route_groups:
                if len(group) < 10:  # Skip routes with too few orders
                    continue
                    
                delay_rate = len(group[(group['status'] == 'Delivered') & (group['is_late_delivery'] == 1)]) / len(group) if len(group) > 0 else 0
                
                if delay_rate > 0.15:  # 15% threshold
                    risk_factors.append({
                        'type': 'route',
                        'route_code': route_code,
                        'delay_rate': delay_rate * 100,
                        'order_count': len(group)
                    })
        
        # Check for city-specific issues
        city_groups = data.groupby('city')
        
        for city, group in city_groups:
            if len(group) < 20:  # Skip cities with too few orders
                continue
                
            failure_rate = len(group[group['status'] == 'Failed']) / len(group) if len(group) > 0 else 0
            delay_rate = len(group[(group['status'] == 'Delivered') & (group['is_late_delivery'] == 1)]) / len(group) if len(group) > 0 else 0
            
            if failure_rate > 0.08 or delay_rate > 0.12:  # 8% failure or 12% delay threshold
                risk_factors.append({
                    'type': 'city',
                    'city': city,
                    'failure_rate': failure_rate * 100,
                    'delay_rate': delay_rate * 100,
                    'order_count': len(group)
                })
        
        return risk_factors
    
    def _predict_bottlenecks(self, data: pd.DataFrame, additional_orders: int) -> List[Dict[str, Any]]:
        """
        Predict potential bottlenecks with additional order volume.
        
        Args:
            data: DataFrame of historical order data
            additional_orders: Number of additional orders to consider
            
        Returns:
            List[Dict[str, Any]]: List of predicted bottlenecks
        """
        bottlenecks = []
        
        # Estimate distribution of additional orders
        total_current_orders = len(data)
        
        if total_current_orders == 0:
            return bottlenecks
        
        # Predict warehouse bottlenecks
        if 'warehouse_id' in data.columns and 'warehouse_name' in data.columns:
            warehouse_counts = data.groupby(['warehouse_id', 'warehouse_name']).size()
            
            for (warehouse_id, warehouse_name), count in warehouse_counts.items():
                # Estimate additional orders for this warehouse
                warehouse_ratio = count / total_current_orders
                estimated_new_orders = int(additional_orders * warehouse_ratio)
                
                # Check if warehouse has capacity issues in historical data
                warehouse_data = data[(data['warehouse_id'] == warehouse_id)]
                
                if 'total_warehouse_time_minutes' in warehouse_data.columns:
                    avg_processing_time = warehouse_data['total_warehouse_time_minutes'].mean()
                    
                    # If average processing time is high, flag as potential bottleneck
                    if avg_processing_time > 120:  # 2 hours threshold
                        bottlenecks.append({
                            'type': 'warehouse',
                            'warehouse_id': warehouse_id,
                            'warehouse_name': warehouse_name,
                            'current_orders': count,
                            'estimated_additional_orders': estimated_new_orders,
                            'avg_processing_time_minutes': avg_processing_time,
                            'risk_level': 'high' if avg_processing_time > 180 else 'medium'
                        })
        
        # Predict driver/fleet bottlenecks
        if 'driver_id' in data.columns:
            driver_counts = data.groupby('driver_id').size()
            
            for driver_id, count in driver_counts.items():
                # Estimate additional orders for this driver
                driver_ratio = count / total_current_orders
                estimated_new_orders = int(additional_orders * driver_ratio)
                
                # Check if driver already has high order volume
                if count > 50:  # Threshold for high volume
                    driver_name = "Unknown"
                    if 'driver_name' in data.columns:
                        driver_data = data[data['driver_id'] == driver_id]
                        if not driver_data.empty:
                            driver_name = driver_data['driver_name'].iloc[0]
                    
                    bottlenecks.append({
                        'type': 'driver',
                        'driver_id': driver_id,
                        'driver_name': driver_name,
                        'current_orders': count,
                        'estimated_additional_orders': estimated_new_orders,
                        'risk_level': 'high' if count > 100 else 'medium'
                    })
        
        # Predict city-level bottlenecks
        city_counts = data.groupby('city').size()
        
        for city, count in city_counts.items():
            # Estimate additional orders for this city
            city_ratio = count / total_current_orders
            estimated_new_orders = int(additional_orders * city_ratio)
            
            # Check if city has high failure or delay rates
            city_data = data[data['city'] == city]
            failure_rate = len(city_data[city_data['status'] == 'Failed']) / len(city_data) if len(city_data) > 0 else 0
            delay_rate = len(city_data[(city_data['status'] == 'Delivered') & (city_data['is_late_delivery'] == 1)]) / len(city_data) if len(city_data) > 0 else 0
            
            if failure_rate > 0.05 or delay_rate > 0.1:  # 5% failure or 10% delay threshold
                bottlenecks.append({
                    'type': 'city',
                    'city': city,
                    'current_orders': count,
                    'estimated_additional_orders': estimated_new_orders,
                    'failure_rate': failure_rate * 100,
                    'delay_rate': delay_rate * 100,
                    'risk_level': 'high' if (failure_rate > 0.1 or delay_rate > 0.15) else 'medium'
                })
        
        return bottlenecks
