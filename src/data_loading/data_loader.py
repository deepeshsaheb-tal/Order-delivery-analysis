"""
Data loader module for the logistics insights engine.
"""
import os
import logging
from typing import Dict, Any, List, Optional, TypeVar, Type, Callable
import pandas as pd
from datetime import datetime

from src.data_loading.models import (
    Client, Driver, ExternalFactor, Feedback, 
    FleetLog, Order, WarehouseLog, Warehouse
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Type variable for generic data model
T = TypeVar('T')


class DataLoader:
    """
    Data loader class for loading and validating CSV data files.
    """
    
    def __init__(self, data_dir: str):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Directory containing the data files
        """
        self.data_dir = data_dir
        self.dataframes: Dict[str, pd.DataFrame] = {}
        self.models: Dict[str, Any] = {}
    
    def load_all_data(self) -> bool:
        """
        Load all data files from the data directory.
        
        Returns:
            bool: True if all data was loaded successfully, False otherwise
        """
        try:
            # Load all CSV files
            self.load_clients()
            self.load_drivers()
            self.load_external_factors()
            self.load_feedback()
            self.load_fleet_logs()
            self.load_orders()
            self.load_warehouse_logs()
            self.load_warehouses()
            
            logger.info("All data loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return False
    
    def _load_csv(self, filename: str) -> pd.DataFrame:
        """
        Load a CSV file into a pandas DataFrame.
        
        Args:
            filename: Name of the CSV file to load
            
        Returns:
            pd.DataFrame: The loaded DataFrame
            
        Raises:
            FileNotFoundError: If the file does not exist
            pd.errors.ParserError: If the file cannot be parsed
        """
        file_path = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Loaded {len(df)} rows from {filename}")
            return df
        except pd.errors.ParserError as e:
            logger.error(f"Error parsing {filename}: {str(e)}")
            raise
    
    def _convert_to_models(
        self, 
        df: pd.DataFrame, 
        model_class: Type[T],
        date_columns: List[str] = None
    ) -> List[T]:
        """
        Convert a DataFrame to a list of model instances.
        
        Args:
            df: DataFrame to convert
            model_class: Model class to instantiate
            date_columns: List of column names that should be parsed as dates
            
        Returns:
            List[T]: List of model instances
        """
        if date_columns is None:
            date_columns = []
        
        # Convert date columns
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Convert DataFrame to list of dictionaries
        records = df.where(pd.notna(df), None).to_dict('records')
        
        # Create model instances
        models = []
        for record in records:
            try:
                model = model_class(**record)
                models.append(model)
            except Exception as e:
                logger.warning(f"Error creating {model_class.__name__} model: {str(e)}")
        
        return models
    
    def load_clients(self) -> List[Client]:
        """
        Load client data from CSV.
        
        Returns:
            List[Client]: List of Client models
        """
        df = self._load_csv('clients.csv')
        clients = self._convert_to_models(
            df, 
            Client, 
            date_columns=['created_at']
        )
        self.dataframes['clients'] = df
        self.models['clients'] = clients
        return clients
    
    def load_drivers(self) -> List[Driver]:
        """
        Load driver data from CSV.
        
        Returns:
            List[Driver]: List of Driver models
        """
        df = self._load_csv('drivers.csv')
        drivers = self._convert_to_models(
            df, 
            Driver, 
            date_columns=['created_at']
        )
        self.dataframes['drivers'] = df
        self.models['drivers'] = drivers
        return drivers
    
    def load_external_factors(self) -> List[ExternalFactor]:
        """
        Load external factors data from CSV.
        
        Returns:
            List[ExternalFactor]: List of ExternalFactor models
        """
        df = self._load_csv('external_factors.csv')
        factors = self._convert_to_models(
            df, 
            ExternalFactor, 
            date_columns=['recorded_at']
        )
        self.dataframes['external_factors'] = df
        self.models['external_factors'] = factors
        return factors
    
    def load_feedback(self) -> List[Feedback]:
        """
        Load feedback data from CSV.
        
        Returns:
            List[Feedback]: List of Feedback models
        """
        df = self._load_csv('feedback.csv')
        feedback = self._convert_to_models(
            df, 
            Feedback, 
            date_columns=['created_at']
        )
        self.dataframes['feedback'] = df
        self.models['feedback'] = feedback
        return feedback
    
    def load_fleet_logs(self) -> List[FleetLog]:
        """
        Load fleet logs data from CSV.
        
        Returns:
            List[FleetLog]: List of FleetLog models
        """
        df = self._load_csv('fleet_logs.csv')
        fleet_logs = self._convert_to_models(
            df, 
            FleetLog, 
            date_columns=['departure_time', 'arrival_time', 'created_at']
        )
        self.dataframes['fleet_logs'] = df
        self.models['fleet_logs'] = fleet_logs
        return fleet_logs
    
    def load_orders(self) -> List[Order]:
        """
        Load orders data from CSV.
        
        Returns:
            List[Order]: List of Order models
        """
        df = self._load_csv('orders.csv')
        orders = self._convert_to_models(
            df, 
            Order, 
            date_columns=[
                'order_date', 
                'promised_delivery_date', 
                'actual_delivery_date', 
                'created_at'
            ]
        )
        self.dataframes['orders'] = df
        self.models['orders'] = orders
        return orders
    
    def load_warehouse_logs(self) -> List[WarehouseLog]:
        """
        Load warehouse logs data from CSV.
        
        Returns:
            List[WarehouseLog]: List of WarehouseLog models
        """
        df = self._load_csv('warehouse_logs.csv')
        warehouse_logs = self._convert_to_models(
            df, 
            WarehouseLog, 
            date_columns=['picking_start', 'picking_end', 'dispatch_time']
        )
        self.dataframes['warehouse_logs'] = df
        self.models['warehouse_logs'] = warehouse_logs
        return warehouse_logs
    
    def load_warehouses(self) -> List[Warehouse]:
        """
        Load warehouses data from CSV.
        
        Returns:
            List[Warehouse]: List of Warehouse models
        """
        df = self._load_csv('warehouses.csv')
        warehouses = self._convert_to_models(
            df, 
            Warehouse, 
            date_columns=['created_at']
        )
        self.dataframes['warehouses'] = df
        self.models['warehouses'] = warehouses
        return warehouses
    
    def get_dataframe(self, name: str) -> Optional[pd.DataFrame]:
        """
        Get a DataFrame by name.
        
        Args:
            name: Name of the DataFrame to get
            
        Returns:
            Optional[pd.DataFrame]: The DataFrame, or None if not found
        """
        return self.dataframes.get(name)
    
    def get_models(self, name: str) -> Optional[List[Any]]:
        """
        Get a list of models by name.
        
        Args:
            name: Name of the models to get
            
        Returns:
            Optional[List[Any]]: The list of models, or None if not found
        """
        return self.models.get(name)
