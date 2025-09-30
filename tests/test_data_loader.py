"""
Unit tests for the data loading module.
"""
import os
import unittest
import pandas as pd
from unittest.mock import patch, mock_open

from src.data_loading.data_loader import DataLoader
from src.data_loading.models import Client, Driver, Order


class TestDataLoader(unittest.TestCase):
    """Test cases for the DataLoader class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_data_dir = "test_data"
        self.data_loader = DataLoader(self.test_data_dir)
    
    @patch('os.path.exists')
    @patch('pandas.read_csv')
    def test_load_csv(self, mock_read_csv, mock_exists):
        """Test loading a CSV file."""
        # Mock the file existence check
        mock_exists.return_value = True
        
        # Mock the pandas read_csv function
        mock_df = pd.DataFrame({
            'client_id': [1, 2],
            'client_name': ['Test Client 1', 'Test Client 2']
        })
        mock_read_csv.return_value = mock_df
        
        # Call the method
        result = self.data_loader._load_csv('clients.csv')
        
        # Verify the result
        self.assertEqual(len(result), 2)
        self.assertEqual(list(result.columns), ['client_id', 'client_name'])
        
        # Verify that the methods were called with the correct arguments
        mock_exists.assert_called_once_with(os.path.join(self.test_data_dir, 'clients.csv'))
        mock_read_csv.assert_called_once_with(os.path.join(self.test_data_dir, 'clients.csv'))
    
    @patch('os.path.exists')
    def test_load_csv_file_not_found(self, mock_exists):
        """Test loading a non-existent CSV file."""
        # Mock the file existence check
        mock_exists.return_value = False
        
        # Call the method and check for exception
        with self.assertRaises(FileNotFoundError):
            self.data_loader._load_csv('nonexistent.csv')
    
    @patch.object(DataLoader, '_load_csv')
    @patch.object(DataLoader, '_convert_to_models')
    def test_load_clients(self, mock_convert, mock_load_csv):
        """Test loading client data."""
        # Mock the load_csv method
        mock_df = pd.DataFrame({
            'client_id': [1, 2],
            'client_name': ['Test Client 1', 'Test Client 2'],
            'gst_number': ['GST1', 'GST2'],
            'contact_person': ['Contact 1', 'Contact 2'],
            'contact_phone': ['123456', '789012'],
            'contact_email': ['email1@test.com', 'email2@test.com'],
            'address_line1': ['Address 1', 'Address 2'],
            'address_line2': ['', ''],
            'city': ['City 1', 'City 2'],
            'state': ['State 1', 'State 2'],
            'pincode': ['123456', '789012'],
            'created_at': ['2023-01-01', '2023-01-02']
        })
        mock_load_csv.return_value = mock_df
        
        # Mock the convert_to_models method
        mock_clients = [
            Client(
                client_id=1,
                client_name='Test Client 1',
                gst_number='GST1',
                contact_person='Contact 1',
                contact_phone='123456',
                contact_email='email1@test.com',
                address_line1='Address 1',
                address_line2='',
                city='City 1',
                state='State 1',
                pincode='123456',
                created_at=pd.Timestamp('2023-01-01')
            ),
            Client(
                client_id=2,
                client_name='Test Client 2',
                gst_number='GST2',
                contact_person='Contact 2',
                contact_phone='789012',
                contact_email='email2@test.com',
                address_line1='Address 2',
                address_line2='',
                city='City 2',
                state='State 2',
                pincode='789012',
                created_at=pd.Timestamp('2023-01-02')
            )
        ]
        mock_convert.return_value = mock_clients
        
        # Call the method
        result = self.data_loader.load_clients()
        
        # Verify the result
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].client_name, 'Test Client 1')
        self.assertEqual(result[1].client_name, 'Test Client 2')
        
        # Verify that the methods were called with the correct arguments
        mock_load_csv.assert_called_once_with('clients.csv')
        mock_convert.assert_called_once()
        
        # Verify that the dataframes and models were stored
        self.assertIn('clients', self.data_loader.dataframes)
        self.assertIn('clients', self.data_loader.models)
        self.assertEqual(self.data_loader.dataframes['clients'], mock_df)
        self.assertEqual(self.data_loader.models['clients'], mock_clients)


if __name__ == '__main__':
    unittest.main()
