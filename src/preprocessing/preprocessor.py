"""
Data preprocessing module for the logistics insights engine.
"""
import re
import logging
import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Any
from datetime import datetime, timedelta
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Data preprocessor class for cleaning and normalizing data.
    """
    
    def __init__(self):
        """Initialize the data preprocessor."""
        self.lemmatizer = WordNetLemmatizer()
        
        # Download NLTK resources if not already downloaded
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('tokenizers/punkt_tab')
            nltk.data.find('corpora/stopwords')
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('punkt')
            nltk.download('punkt_tab')
            nltk.download('stopwords')
            nltk.download('wordnet')
        
        self.stop_words = set(stopwords.words('english'))
    
    def preprocess_all_data(self, dataframes: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Preprocess all dataframes.
        
        Args:
            dataframes: Dictionary of dataframes to preprocess
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary of preprocessed dataframes
        """
        preprocessed = {}
        
        # Process each dataframe
        if 'clients' in dataframes:
            preprocessed['clients'] = self.preprocess_clients(dataframes['clients'])
        
        if 'drivers' in dataframes:
            preprocessed['drivers'] = self.preprocess_drivers(dataframes['drivers'])
        
        if 'external_factors' in dataframes:
            preprocessed['external_factors'] = self.preprocess_external_factors(dataframes['external_factors'])
        
        if 'feedback' in dataframes:
            preprocessed['feedback'] = self.preprocess_feedback(dataframes['feedback'])
        
        if 'fleet_logs' in dataframes:
            preprocessed['fleet_logs'] = self.preprocess_fleet_logs(dataframes['fleet_logs'])
        
        if 'orders' in dataframes:
            preprocessed['orders'] = self.preprocess_orders(dataframes['orders'])
        
        if 'warehouse_logs' in dataframes:
            preprocessed['warehouse_logs'] = self.preprocess_warehouse_logs(dataframes['warehouse_logs'])
        
        if 'warehouses' in dataframes:
            preprocessed['warehouses'] = self.preprocess_warehouses(dataframes['warehouses'])
        
        return preprocessed
    
    def normalize_datetime(self, df: pd.DataFrame, datetime_columns: List[str]) -> pd.DataFrame:
        """
        Normalize datetime columns in a dataframe.
        
        Args:
            df: DataFrame to normalize
            datetime_columns: List of datetime column names
            
        Returns:
            pd.DataFrame: Normalized DataFrame
        """
        df_copy = df.copy()
        
        for col in datetime_columns:
            if col in df_copy.columns:
                # Convert to datetime
                df_copy[col] = pd.to_datetime(df_copy[col], errors='coerce')
                
                # Add derived columns for analysis
                if not df_copy[col].isna().all():
                    df_copy[f'{col}_year'] = df_copy[col].dt.year
                    df_copy[f'{col}_month'] = df_copy[col].dt.month
                    df_copy[f'{col}_day'] = df_copy[col].dt.day
                    df_copy[f'{col}_hour'] = df_copy[col].dt.hour
                    df_copy[f'{col}_dayofweek'] = df_copy[col].dt.dayofweek
                    df_copy[f'{col}_is_weekend'] = df_copy[col].dt.dayofweek.isin([5, 6]).astype(int)
        
        return df_copy
    
    def standardize_address(self, df: pd.DataFrame, address_columns: List[str]) -> pd.DataFrame:
        """
        Standardize address columns in a dataframe.
        
        Args:
            df: DataFrame to standardize
            address_columns: List of address column names
            
        Returns:
            pd.DataFrame: Standardized DataFrame
        """
        df_copy = df.copy()
        
        for col in address_columns:
            if col in df_copy.columns:
                # Clean up addresses
                df_copy[col] = df_copy[col].astype(str)
                
                # Remove extra whitespace
                df_copy[col] = df_copy[col].str.strip()
                df_copy[col] = df_copy[col].str.replace(r'\s+', ' ', regex=True)
                
                # Standardize common abbreviations
                df_copy[col] = df_copy[col].str.replace(r'\bSt\b', 'Street', regex=True)
                df_copy[col] = df_copy[col].str.replace(r'\bRd\b', 'Road', regex=True)
                df_copy[col] = df_copy[col].str.replace(r'\bAve\b', 'Avenue', regex=True)
                df_copy[col] = df_copy[col].str.replace(r'\bBlvd\b', 'Boulevard', regex=True)
                
                # Remove special characters
                df_copy[col] = df_copy[col].str.replace(r'[^\w\s,.-]', '', regex=True)
        
        return df_copy
    
    def preprocess_text(self, df: pd.DataFrame, text_columns: List[str]) -> pd.DataFrame:
        """
        Preprocess text columns in a dataframe.
        
        Args:
            df: DataFrame to preprocess
            text_columns: List of text column names
            
        Returns:
            pd.DataFrame: Preprocessed DataFrame
        """
        df_copy = df.copy()
        
        for col in text_columns:
            if col in df_copy.columns:
                # Convert to string
                df_copy[col] = df_copy[col].astype(str)
                
                # Create a processed version of the text column
                processed_col = f'{col}_processed'
                df_copy[processed_col] = df_copy[col].apply(self._process_text)
                
                # Create a tokens column
                tokens_col = f'{col}_tokens'
                df_copy[tokens_col] = df_copy[processed_col].apply(lambda x: x.split())
        
        return df_copy
    
    def _process_text(self, text: str) -> str:
        """
        Process a text string for NLP analysis.
        
        Args:
            text: Text to process
            
        Returns:
            str: Processed text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stop words
        tokens = [token for token in tokens if token not in self.stop_words]
        
        # Lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        # Join tokens back into a string
        processed_text = ' '.join(tokens)
        
        return processed_text
    
    def add_delivery_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add delivery metrics to orders dataframe.
        
        Args:
            df: Orders DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with added metrics
        """
        df_copy = df.copy()
        
        # Calculate delivery time difference
        if 'promised_delivery_date' in df_copy.columns and 'actual_delivery_date' in df_copy.columns:
            # Calculate delivery delay in hours
            df_copy['delivery_delay_hours'] = (
                (df_copy['actual_delivery_date'] - df_copy['promised_delivery_date'])
                .dt.total_seconds() / 3600
            )
            
            # Flag for late delivery
            df_copy['is_late_delivery'] = (df_copy['delivery_delay_hours'] > 0).astype(int)
            
            # Flag for very late delivery (more than 24 hours)
            df_copy['is_very_late_delivery'] = (df_copy['delivery_delay_hours'] > 24).astype(int)
        
        return df_copy
    
    def preprocess_clients(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess clients dataframe.
        
        Args:
            df: Clients DataFrame
            
        Returns:
            pd.DataFrame: Preprocessed DataFrame
        """
        df_copy = df.copy()
        
        # Normalize datetime columns
        df_copy = self.normalize_datetime(df_copy, ['created_at'])
        
        # Standardize address columns
        df_copy = self.standardize_address(df_copy, ['address_line1', 'address_line2'])
        
        return df_copy
    
    def preprocess_drivers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess drivers dataframe.
        
        Args:
            df: Drivers DataFrame
            
        Returns:
            pd.DataFrame: Preprocessed DataFrame
        """
        df_copy = df.copy()
        
        # Normalize datetime columns
        df_copy = self.normalize_datetime(df_copy, ['created_at'])
        
        # Create binary status column
        df_copy['is_active'] = (df_copy['status'] == 'Active').astype(int)
        
        # Create binary column for in-house drivers
        df_copy['is_inhouse'] = (df_copy['partner_company'] == 'In-house').astype(int)
        
        return df_copy
    
    def preprocess_external_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess external factors dataframe.
        
        Args:
            df: External factors DataFrame
            
        Returns:
            pd.DataFrame: Preprocessed DataFrame
        """
        df_copy = df.copy()
        
        # Normalize datetime columns
        df_copy = self.normalize_datetime(df_copy, ['recorded_at'])
        
        # Create binary columns for conditions
        df_copy['has_traffic'] = (~df_copy['traffic_condition'].isin(['Clear', 'Light'])).astype(int)
        df_copy['has_bad_weather'] = (df_copy['weather_condition'].isin(['Rain', 'Snow', 'Storm'])).astype(int)
        df_copy['has_event'] = (~df_copy['event_type'].isna()).astype(int)
        
        return df_copy
    
    def preprocess_feedback(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess feedback dataframe.
        
        Args:
            df: Feedback DataFrame
            
        Returns:
            pd.DataFrame: Preprocessed DataFrame
        """
        df_copy = df.copy()
        
        # Normalize datetime columns
        df_copy = self.normalize_datetime(df_copy, ['created_at'])
        
        # Preprocess text columns
        df_copy = self.preprocess_text(df_copy, ['feedback_text'])
        
        # Create binary sentiment column
        df_copy['is_positive'] = (df_copy['sentiment'] == 'Positive').astype(int)
        
        # Create rating categories
        df_copy['rating_category'] = pd.cut(
            df_copy['rating'],
            bins=[0, 2, 3, 5],
            labels=['Poor', 'Average', 'Good']
        )
        
        return df_copy
    
    def preprocess_fleet_logs(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess fleet logs dataframe.
        
        Args:
            df: Fleet logs DataFrame
            
        Returns:
            pd.DataFrame: Preprocessed DataFrame
        """
        df_copy = df.copy()
        
        # Normalize datetime columns
        df_copy = self.normalize_datetime(df_copy, ['departure_time', 'arrival_time', 'created_at'])
        
        # Calculate transit time in hours
        df_copy['transit_time_hours'] = (
            (df_copy['arrival_time'] - df_copy['departure_time'])
            .dt.total_seconds() / 3600
        )
        
        # Preprocess delay notes
        df_copy = self.preprocess_text(df_copy, ['gps_delay_notes'])
        
        return df_copy
    
    def preprocess_orders(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess orders dataframe.
        
        Args:
            df: Orders DataFrame
            
        Returns:
            pd.DataFrame: Preprocessed DataFrame
        """
        df_copy = df.copy()
        
        # Normalize datetime columns
        df_copy = self.normalize_datetime(
            df_copy, 
            ['order_date', 'promised_delivery_date', 'actual_delivery_date', 'created_at']
        )
        
        # Standardize address columns
        df_copy = self.standardize_address(
            df_copy, 
            ['delivery_address_line1', 'delivery_address_line2']
        )
        
        # Add delivery metrics
        df_copy = self.add_delivery_metrics(df_copy)
        
        # Create binary status columns
        df_copy['is_delivered'] = (df_copy['status'] == 'Delivered').astype(int)
        df_copy['is_failed'] = (df_copy['status'] == 'Failed').astype(int)
        df_copy['is_pending'] = (df_copy['status'] == 'Pending').astype(int)
        
        # Create binary payment mode columns
        df_copy['is_cod'] = (df_copy['payment_mode'] == 'COD').astype(int)
        
        return df_copy
    
    def preprocess_warehouse_logs(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess warehouse logs dataframe.
        
        Args:
            df: Warehouse logs DataFrame
            
        Returns:
            pd.DataFrame: Preprocessed DataFrame
        """
        df_copy = df.copy()
        
        # Normalize datetime columns
        df_copy = self.normalize_datetime(
            df_copy, 
            ['picking_start', 'picking_end', 'dispatch_time']
        )
        
        # Calculate processing times
        df_copy['picking_time_minutes'] = (
            (df_copy['picking_end'] - df_copy['picking_start'])
            .dt.total_seconds() / 60
        )
        
        df_copy['dispatch_prep_time_minutes'] = (
            (df_copy['dispatch_time'] - df_copy['picking_end'])
            .dt.total_seconds() / 60
        )
        
        df_copy['total_warehouse_time_minutes'] = (
            (df_copy['dispatch_time'] - df_copy['picking_start'])
            .dt.total_seconds() / 60
        )
        
        # Preprocess notes
        df_copy = self.preprocess_text(df_copy, ['notes'])
        
        # Create binary column for delay notes
        df_copy['has_delay_note'] = (~df_copy['notes'].isna() & (df_copy['notes'] != '')).astype(int)
        
        return df_copy
    
    def preprocess_warehouses(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess warehouses dataframe.
        
        Args:
            df: Warehouses DataFrame
            
        Returns:
            pd.DataFrame: Preprocessed DataFrame
        """
        df_copy = df.copy()
        
        # Normalize datetime columns
        df_copy = self.normalize_datetime(df_copy, ['created_at'])
        
        # Categorize warehouse capacity
        df_copy['capacity_category'] = pd.cut(
            df_copy['capacity'],
            bins=[0, 800, 1200, float('inf')],
            labels=['Small', 'Medium', 'Large']
        )
        
        return df_copy
