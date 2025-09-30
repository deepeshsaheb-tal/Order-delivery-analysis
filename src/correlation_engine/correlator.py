"""
Data correlation engine for the logistics insights engine.
"""
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataCorrelator:
    """
    Data correlator class for linking data across different domains.
    """
    
    def __init__(self, dataframes: Dict[str, pd.DataFrame]):
        """
        Initialize the data correlator.
        
        Args:
            dataframes: Dictionary of preprocessed dataframes
        """
        self.dataframes = dataframes
        self.correlated_data: Dict[str, pd.DataFrame] = {}
    
    def correlate_all_data(self) -> Dict[str, pd.DataFrame]:
        """
        Correlate all data across different domains.
        
        Returns:
            Dict[str, pd.DataFrame]: Dictionary of correlated dataframes
        """
        # Correlate orders with fleet logs
        self.correlated_data['order_fleet'] = self.correlate_orders_with_fleet()
        
        # Correlate orders with warehouse logs
        self.correlated_data['order_warehouse'] = self.correlate_orders_with_warehouse()
        
        # Correlate orders with external factors
        self.correlated_data['order_external'] = self.correlate_orders_with_external_factors()
        
        # Correlate orders with feedback
        self.correlated_data['order_feedback'] = self.correlate_orders_with_feedback()
        
        # Create a comprehensive order data view
        self.correlated_data['order_comprehensive'] = self.create_comprehensive_order_view()
        
        return self.correlated_data
    
    def correlate_orders_with_fleet(self) -> pd.DataFrame:
        """
        Correlate orders with fleet logs.
        
        Returns:
            pd.DataFrame: Correlated orders and fleet logs
        """
        if 'orders' not in self.dataframes or 'fleet_logs' not in self.dataframes:
            logger.error("Missing required dataframes for order-fleet correlation")
            return pd.DataFrame()
        
        orders_df = self.dataframes['orders']
        fleet_df = self.dataframes['fleet_logs']
        
        # Merge orders with fleet logs
        correlated = pd.merge(
            orders_df,
            fleet_df,
            on='order_id',
            how='left',
            suffixes=('', '_fleet')
        )
        
        # Add driver information if available
        if 'drivers' in self.dataframes:
            drivers_df = self.dataframes['drivers']
            correlated = pd.merge(
                correlated,
                drivers_df,
                on='driver_id',
                how='left',
                suffixes=('', '_driver')
            )
        
        # Calculate delivery metrics
        correlated['fleet_delay_hours'] = np.nan
        mask = ~correlated['departure_time'].isna() & ~correlated['arrival_time'].isna()
        
        if mask.any():
            # Calculate time between promised delivery and actual arrival
            correlated.loc[mask, 'fleet_delay_hours'] = (
                (correlated.loc[mask, 'arrival_time'] - correlated.loc[mask, 'promised_delivery_date'])
                .dt.total_seconds() / 3600
            )
        
        return correlated
    
    def correlate_orders_with_warehouse(self) -> pd.DataFrame:
        """
        Correlate orders with warehouse logs.
        
        Returns:
            pd.DataFrame: Correlated orders and warehouse logs
        """
        if 'orders' not in self.dataframes or 'warehouse_logs' not in self.dataframes:
            logger.error("Missing required dataframes for order-warehouse correlation")
            return pd.DataFrame()
        
        orders_df = self.dataframes['orders']
        warehouse_logs_df = self.dataframes['warehouse_logs']
        
        # Merge orders with warehouse logs
        correlated = pd.merge(
            orders_df,
            warehouse_logs_df,
            on='order_id',
            how='left',
            suffixes=('', '_wh_log')
        )
        
        # Add warehouse information if available
        if 'warehouses' in self.dataframes:
            warehouses_df = self.dataframes['warehouses']
            correlated = pd.merge(
                correlated,
                warehouses_df,
                left_on='warehouse_id',
                right_on='warehouse_id',
                how='left',
                suffixes=('', '_warehouse')
            )
        
        # Calculate warehouse processing metrics
        correlated['warehouse_processing_delay'] = np.nan
        mask = ~correlated['dispatch_time'].isna() & ~correlated['order_date'].isna()
        
        if mask.any():
            # Calculate time between order date and dispatch time
            correlated.loc[mask, 'warehouse_processing_delay'] = (
                (correlated.loc[mask, 'dispatch_time'] - correlated.loc[mask, 'order_date'])
                .dt.total_seconds() / 3600
            )
        
        return correlated
    
    def correlate_orders_with_external_factors(self) -> pd.DataFrame:
        """
        Correlate orders with external factors.
        
        Returns:
            pd.DataFrame: Correlated orders and external factors
        """
        if 'orders' not in self.dataframes or 'external_factors' not in self.dataframes:
            logger.error("Missing required dataframes for order-external correlation")
            return pd.DataFrame()
        
        orders_df = self.dataframes['orders']
        external_df = self.dataframes['external_factors']
        
        # Merge orders with external factors
        correlated = pd.merge(
            orders_df,
            external_df,
            on='order_id',
            how='left',
            suffixes=('', '_external')
        )
        
        # Create binary flags for external factors
        correlated['has_external_factor'] = (~correlated['factor_id'].isna()).astype(int)
        correlated['has_traffic_issue'] = (
            (~correlated['traffic_condition'].isna()) & 
            (correlated['traffic_condition'] != 'Clear')
        ).astype(int)
        
        correlated['has_weather_issue'] = (
            (~correlated['weather_condition'].isna()) & 
            (correlated['weather_condition'] != 'Clear')
        ).astype(int)
        
        correlated['has_event_impact'] = (~correlated['event_type'].isna()).astype(int)
        
        return correlated
    
    def correlate_orders_with_feedback(self) -> pd.DataFrame:
        """
        Correlate orders with customer feedback.
        
        Returns:
            pd.DataFrame: Correlated orders and feedback
        """
        if 'orders' not in self.dataframes or 'feedback' not in self.dataframes:
            logger.error("Missing required dataframes for order-feedback correlation")
            return pd.DataFrame()
        
        orders_df = self.dataframes['orders']
        feedback_df = self.dataframes['feedback']
        
        # Merge orders with feedback
        correlated = pd.merge(
            orders_df,
            feedback_df,
            on='order_id',
            how='left',
            suffixes=('', '_feedback')
        )
        
        # Create binary flag for having feedback
        correlated['has_feedback'] = (~correlated['feedback_id'].isna()).astype(int)
        
        # Create binary flag for negative feedback
        correlated['has_negative_feedback'] = (
            (correlated['has_feedback'] == 1) & 
            (correlated['sentiment'] == 'Negative')
        ).astype(int)
        
        return correlated
    
    def create_comprehensive_order_view(self) -> pd.DataFrame:
        """
        Create a comprehensive view of orders with all correlated data.
        
        Returns:
            pd.DataFrame: Comprehensive order view
        """
        if 'orders' not in self.dataframes:
            logger.error("Missing orders dataframe for comprehensive view")
            return pd.DataFrame()
        
        # Start with the orders dataframe
        orders_df = self.dataframes['orders'].copy()
        
        # Add client information if available
        if 'clients' in self.dataframes:
            clients_df = self.dataframes['clients']
            orders_df = pd.merge(
                orders_df,
                clients_df[['client_id', 'client_name']],
                on='client_id',
                how='left'
            )
        
        # Add fleet information
        if 'order_fleet' in self.correlated_data:
            fleet_cols = [
                'order_id', 'fleet_log_id', 'driver_id', 'vehicle_number',
                'route_code', 'gps_delay_notes', 'departure_time', 'arrival_time',
                'transit_time_hours', 'fleet_delay_hours'
            ]
            
            # Add driver information if available
            if 'drivers' in self.dataframes:
                fleet_cols.extend(['driver_name', 'partner_company', 'status'])
            
            fleet_df = self.correlated_data['order_fleet'][fleet_cols].drop_duplicates(subset=['order_id'])
            
            orders_df = pd.merge(
                orders_df,
                fleet_df,
                on='order_id',
                how='left',
                suffixes=('', '_fleet')
            )
        
        # Add warehouse information
        if 'order_warehouse' in self.correlated_data:
            warehouse_cols = [
                'order_id', 'log_id', 'warehouse_id', 'picking_start', 'picking_end',
                'dispatch_time', 'notes', 'picking_time_minutes',
                'dispatch_prep_time_minutes', 'total_warehouse_time_minutes',
                'warehouse_processing_delay'
            ]
            
            # Add warehouse details if available
            if 'warehouses' in self.dataframes:
                warehouse_cols.extend(['warehouse_name', 'capacity', 'capacity_category'])
            
            warehouse_df = self.correlated_data['order_warehouse'][warehouse_cols].drop_duplicates(subset=['order_id'])
            
            orders_df = pd.merge(
                orders_df,
                warehouse_df,
                on='order_id',
                how='left',
                suffixes=('', '_wh')
            )
        
        # Add external factors
        if 'order_external' in self.correlated_data:
            external_cols = [
                'order_id', 'factor_id', 'traffic_condition', 'weather_condition',
                'event_type', 'recorded_at', 'has_external_factor',
                'has_traffic_issue', 'has_weather_issue', 'has_event_impact'
            ]
            
            external_df = self.correlated_data['order_external'][external_cols].drop_duplicates(subset=['order_id'])
            
            orders_df = pd.merge(
                orders_df,
                external_df,
                on='order_id',
                how='left',
                suffixes=('', '_ext')
            )
        
        # Add feedback information
        if 'order_feedback' in self.correlated_data:
            feedback_cols = [
                'order_id', 'feedback_id', 'feedback_text', 'sentiment',
                'rating', 'has_feedback', 'has_negative_feedback'
            ]
            
            feedback_df = self.correlated_data['order_feedback'][feedback_cols].drop_duplicates(subset=['order_id'])
            
            orders_df = pd.merge(
                orders_df,
                feedback_df,
                on='order_id',
                how='left',
                suffixes=('', '_fb')
            )
        
        # Add failure analysis columns
        orders_df['has_warehouse_delay'] = (
            (~orders_df['warehouse_processing_delay'].isna()) & 
            (orders_df['warehouse_processing_delay'] > 24)
        ).astype(int)
        
        orders_df['has_transit_delay'] = (
            (~orders_df['transit_time_hours'].isna()) & 
            (orders_df['transit_time_hours'] > 12)
        ).astype(int)
        
        # Fill NaN values in binary columns with 0
        binary_cols = [
            'has_external_factor', 'has_traffic_issue', 'has_weather_issue',
            'has_event_impact', 'has_feedback', 'has_negative_feedback',
            'has_warehouse_delay', 'has_transit_delay'
        ]
        
        for col in binary_cols:
            if col in orders_df.columns:
                orders_df[col] = orders_df[col].fillna(0)
        
        return orders_df
    
    def get_correlated_data(self, name: str) -> Optional[pd.DataFrame]:
        """
        Get a correlated dataframe by name.
        
        Args:
            name: Name of the correlated dataframe to get
            
        Returns:
            Optional[pd.DataFrame]: The correlated dataframe, or None if not found
        """
        return self.correlated_data.get(name)
    
    def filter_by_city(self, city: str) -> pd.DataFrame:
        """
        Filter comprehensive order data by city.
        
        Args:
            city: City name to filter by
            
        Returns:
            pd.DataFrame: Filtered comprehensive order data
        """
        if 'order_comprehensive' not in self.correlated_data:
            logger.error("Comprehensive order data not available")
            return pd.DataFrame()
        
        df = self.correlated_data['order_comprehensive']
        return df[df['city'] == city]
    
    def filter_by_client(self, client_id: int) -> pd.DataFrame:
        """
        Filter comprehensive order data by client ID.
        
        Args:
            client_id: Client ID to filter by
            
        Returns:
            pd.DataFrame: Filtered comprehensive order data
        """
        if 'order_comprehensive' not in self.correlated_data:
            logger.error("Comprehensive order data not available")
            return pd.DataFrame()
        
        df = self.correlated_data['order_comprehensive']
        return df[df['client_id'] == client_id]
    
    def filter_by_client_name(self, client_name: str) -> pd.DataFrame:
        """
        Filter comprehensive order data by client name.
        
        Args:
            client_name: Client name to filter by
            
        Returns:
            pd.DataFrame: Filtered comprehensive order data
        """
        if 'order_comprehensive' not in self.correlated_data:
            logger.error("Comprehensive order data not available")
            return pd.DataFrame()
        
        df = self.correlated_data['order_comprehensive']
        
        if 'client_name' not in df.columns:
            logger.error("Client name column not available in comprehensive data")
            return pd.DataFrame()
        
        return df[df['client_name'] == client_name]
    
    def filter_by_warehouse(self, warehouse_id: int) -> pd.DataFrame:
        """
        Filter comprehensive order data by warehouse ID.
        
        Args:
            warehouse_id: Warehouse ID to filter by
            
        Returns:
            pd.DataFrame: Filtered comprehensive order data
        """
        if 'order_comprehensive' not in self.correlated_data:
            logger.error("Comprehensive order data not available")
            return pd.DataFrame()
        
        df = self.correlated_data['order_comprehensive']
        
        if 'warehouse_id' not in df.columns:
            logger.error("Warehouse ID column not available in comprehensive data")
            return pd.DataFrame()
        
        return df[df['warehouse_id'] == warehouse_id]
    
    def filter_by_warehouse_name(self, warehouse_name: str) -> pd.DataFrame:
        """
        Filter comprehensive order data by warehouse name.
        
        Args:
            warehouse_name: Warehouse name to filter by
            
        Returns:
            pd.DataFrame: Filtered comprehensive order data
        """
        if 'order_comprehensive' not in self.correlated_data:
            logger.error("Comprehensive order data not available")
            return pd.DataFrame()
        
        df = self.correlated_data['order_comprehensive']
        
        if 'warehouse_name' not in df.columns:
            logger.error("Warehouse name column not available in comprehensive data")
            return pd.DataFrame()
        
        return df[df['warehouse_name'] == warehouse_name]
    
    def filter_by_date_range(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Filter comprehensive order data by date range.
        
        Args:
            start_date: Start date for filtering
            end_date: End date for filtering
            
        Returns:
            pd.DataFrame: Filtered comprehensive order data
        """
        if 'order_comprehensive' not in self.correlated_data:
            logger.error("Comprehensive order data not available")
            return pd.DataFrame()
        
        df = self.correlated_data['order_comprehensive']
        
        return df[
            (df['order_date'] >= start_date) & 
            (df['order_date'] <= end_date)
        ]
    
    def filter_by_status(self, status: str) -> pd.DataFrame:
        """
        Filter comprehensive order data by status.
        
        Args:
            status: Order status to filter by
            
        Returns:
            pd.DataFrame: Filtered comprehensive order data
        """
        if 'order_comprehensive' not in self.correlated_data:
            logger.error("Comprehensive order data not available")
            return pd.DataFrame()
        
        df = self.correlated_data['order_comprehensive']
        
        return df[df['status'] == status]
