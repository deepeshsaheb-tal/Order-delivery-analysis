"""
Unit tests for the query processor module.
"""
import unittest
from unittest.mock import patch, MagicMock
import json
from datetime import datetime, timedelta

from src.query_processor.processor import QueryProcessor


class TestQueryProcessor(unittest.TestCase):
    """Test cases for the QueryProcessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.query_processor = QueryProcessor()
    
    def test_rule_based_parsing_city_query(self):
        """Test rule-based parsing of a city query."""
        query = "Why were deliveries delayed in Mumbai yesterday?"
        result = self.query_processor._rule_based_parsing(query)
        
        # Verify the result
        self.assertEqual(result["query_type"], "delay_analysis")
        self.assertEqual(result["entities"]["city"], "Mumbai")
        
        # Check that yesterday's date was set correctly
        yesterday = datetime.now() - timedelta(days=1)
        self.assertEqual(result["time_range"]["start_date"], yesterday.strftime("%Y-%m-%d"))
        self.assertEqual(result["time_range"]["end_date"], yesterday.strftime("%Y-%m-%d"))
    
    def test_rule_based_parsing_client_query(self):
        """Test rule-based parsing of a client query."""
        query = "Why did Client X's orders fail in the past week?"
        result = self.query_processor._rule_based_parsing(query)
        
        # Verify the result
        self.assertEqual(result["query_type"], "failure_analysis")
        self.assertEqual(result["entities"]["client"], "X")
        
        # Check that last week's date range was set correctly
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        self.assertEqual(result["time_range"]["start_date"], start_date.strftime("%Y-%m-%d"))
        self.assertEqual(result["time_range"]["end_date"], end_date.strftime("%Y-%m-%d"))
    
    def test_rule_based_parsing_warehouse_query(self):
        """Test rule-based parsing of a warehouse query."""
        query = "Explain the top reasons for delivery failures linked to Warehouse B in August?"
        result = self.query_processor._rule_based_parsing(query)
        
        # Verify the result
        self.assertEqual(result["query_type"], "explanation")
        self.assertEqual(result["entities"]["warehouse"], "B")
        
        # Check that August date range was set correctly
        self.assertEqual(result["time_range"]["start_date"], "2023-08-01")
        self.assertEqual(result["time_range"]["end_date"], "2023-08-31")
    
    def test_rule_based_parsing_comparison_query(self):
        """Test rule-based parsing of a comparison query."""
        query = "Compare delivery failure causes between City A and City B last month?"
        result = self.query_processor._rule_based_parsing(query)
        
        # Verify the result
        self.assertEqual(result["query_type"], "comparison")
        
        # Check that last month's date range was set correctly
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        self.assertEqual(result["time_range"]["start_date"], start_date.strftime("%Y-%m-%d"))
        self.assertEqual(result["time_range"]["end_date"], end_date.strftime("%Y-%m-%d"))
    
    def test_rule_based_parsing_prediction_query(self):
        """Test rule-based parsing of a prediction query."""
        query = "If we onboard Client Y with ~20,000 extra monthly orders, what new failure risks should we expect?"
        result = self.query_processor._rule_based_parsing(query)
        
        # Verify the result
        self.assertEqual(result["query_type"], "prediction")
        self.assertEqual(result["entities"]["client"], "Y")
        self.assertEqual(result["additional_parameters"]["order_volume"], 20000)
    
    def test_is_follow_up_question(self):
        """Test detection of follow-up questions."""
        # Test follow-up indicators
        self.assertTrue(self.query_processor.is_follow_up_question("What about Mumbai?"))
        self.assertTrue(self.query_processor.is_follow_up_question("Tell me more about them."))
        self.assertTrue(self.query_processor.is_follow_up_question("Why?"))
        
        # Test non-follow-up questions
        self.assertFalse(self.query_processor.is_follow_up_question("Why were deliveries delayed in Mumbai yesterday?"))
        self.assertFalse(self.query_processor.is_follow_up_question("Compare delivery failure causes between City A and City B."))
    
    def test_reset_context(self):
        """Test resetting the context."""
        # Set some context
        self.query_processor.context = {"test": "value"}
        
        # Reset the context
        self.query_processor.reset_context()
        
        # Verify that the context is empty
        self.assertEqual(self.query_processor.context, {})
    
    @patch('openai.ChatCompletion.create')
    def test_openai_parsing(self, mock_create):
        """Test parsing using OpenAI API."""
        # Set up the mock response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "query_type": "delay_analysis",
            "entities": {
                "city": "Mumbai",
                "client": None,
                "warehouse": None
            },
            "time_range": {
                "start_date": "2023-09-29",
                "end_date": "2023-09-30"
            },
            "additional_parameters": {
                "order_volume": None,
                "specific_reason": None
            }
        })
        mock_create.return_value = mock_response
        
        # Set API key
        self.query_processor.api_key = "test_key"
        
        # Call the method
        query = "Why were deliveries delayed in Mumbai yesterday?"
        result = self.query_processor._openai_parsing(query)
        
        # Verify the result
        self.assertEqual(result["query_type"], "delay_analysis")
        self.assertEqual(result["entities"]["city"], "Mumbai")
        self.assertEqual(result["time_range"]["start_date"], "2023-09-29")
        self.assertEqual(result["time_range"]["end_date"], "2023-09-30")
        
        # Verify that the API was called with the correct arguments
        mock_create.assert_called_once()
        args, kwargs = mock_create.call_args
        self.assertEqual(kwargs["model"], "gpt-3.5-turbo")
        self.assertEqual(len(kwargs["messages"]), 2)
        self.assertEqual(kwargs["messages"][1]["content"], "Parse the following logistics query: \"Why were deliveries delayed in Mumbai yesterday?\"")


if __name__ == '__main__':
    unittest.main()
