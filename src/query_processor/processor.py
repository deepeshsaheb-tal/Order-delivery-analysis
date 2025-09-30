"""
Natural language query processor for the logistics insights engine.
"""
import os
import re
import logging
import json
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from openai import OpenAI
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class QueryProcessor:
    """
    Query processor class for parsing natural language queries.
    """
    
    def __init__(self):
        """Initialize the query processor."""
        # Load OpenAI API key from environment variable
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            logger.warning("OpenAI API key not found. Some functionality will be limited.")
            self.client = None
        else:
            self.client = OpenAI(api_key=self.api_key)
        
        # Initialize context for follow-up questions
        self.context: Dict[str, Any] = {}
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a natural language query.
        
        Args:
            query: Natural language query string
            
        Returns:
            Dict[str, Any]: Processed query parameters
        """
        # Check if OpenAI API key is available
        if not self.api_key:
            # Fall back to rule-based parsing
            return self._rule_based_parsing(query)
        
        try:
            # Use OpenAI API for query understanding
            return self._openai_parsing(query)
        except Exception as e:
            logger.error(f"Error using OpenAI API: {str(e)}")
            # Fall back to rule-based parsing
            return self._rule_based_parsing(query)
    
    def _openai_parsing(self, query: str) -> Dict[str, Any]:
        """
        Parse query using OpenAI API.
        
        Args:
            query: Natural language query string
            
        Returns:
            Dict[str, Any]: Parsed query parameters
        """
        # Prepare the prompt for OpenAI
        prompt = self._prepare_prompt(query)
        
        # Call OpenAI API
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": prompt["system"]},
                {"role": "user", "content": prompt["user"]}
            ],
            temperature=0.1,  # Low temperature for more deterministic results
            max_tokens=500
        )
        
        # Extract and parse the response
        result = response.choices[0].message.content
        
        try:
            # Try to parse as JSON
            parsed_result = json.loads(result)
            return parsed_result
        except json.JSONDecodeError:
            # If not valid JSON, try to extract structured data using regex
            logger.warning("OpenAI response is not valid JSON, trying to extract structured data")
            return self._extract_structured_data(result)
    
    def _prepare_prompt(self, query: str) -> Dict[str, str]:
        """
        Prepare the prompt for OpenAI API.
        
        Args:
            query: Natural language query string
            
        Returns:
            Dict[str, str]: System and user prompts
        """
        system_prompt = """
        You are a logistics data analysis assistant. Your task is to parse natural language queries about logistics data and extract structured parameters.
        
        The output should be a JSON object with the following fields:
        - query_type: The type of query (e.g., "failure_analysis", "delay_analysis", "comparison", "prediction")
        - entities: An object containing extracted entities like city, client, warehouse, time period
        - time_range: An object with start_date and end_date (in YYYY-MM-DD format)
        - additional_parameters: Any other parameters extracted from the query
        
        Example output:
        {
            "query_type": "failure_analysis",
            "entities": {
                "city": "Mumbai",
                "client": null,
                "warehouse": null
            },
            "time_range": {
                "start_date": "2023-09-29",
                "end_date": "2023-09-30"
            },
            "additional_parameters": {
                "order_volume": null,
                "specific_reason": null
            }
        }
        
        Only output valid JSON, no explanations or other text.
        """
        
        user_prompt = f"Parse the following logistics query: \"{query}\""
        
        # Add context for follow-up questions if available
        if self.context:
            user_prompt += f"\n\nContext from previous query: {json.dumps(self.context)}"
        
        return {
            "system": system_prompt,
            "user": user_prompt
        }
    
    def _extract_structured_data(self, text: str) -> Dict[str, Any]:
        """
        Extract structured data from text response.
        
        Args:
            text: Text response from OpenAI
            
        Returns:
            Dict[str, Any]: Extracted structured data
        """
        # Initialize default structure
        result = {
            "query_type": None,
            "entities": {
                "city": None,
                "client": None,
                "warehouse": None
            },
            "time_range": {
                "start_date": None,
                "end_date": None
            },
            "additional_parameters": {
                "order_volume": None,
                "specific_reason": None
            }
        }
        
        # Try to extract query type
        query_type_match = re.search(r'"query_type":\s*"([^"]+)"', text)
        if query_type_match:
            result["query_type"] = query_type_match.group(1)
        
        # Try to extract city
        city_match = re.search(r'"city":\s*"([^"]+)"', text)
        if city_match:
            result["entities"]["city"] = city_match.group(1)
        
        # Try to extract client
        client_match = re.search(r'"client":\s*"([^"]+)"', text)
        if client_match:
            result["entities"]["client"] = client_match.group(1)
        
        # Try to extract warehouse
        warehouse_match = re.search(r'"warehouse":\s*"([^"]+)"', text)
        if warehouse_match:
            result["entities"]["warehouse"] = warehouse_match.group(1)
        
        # Try to extract dates
        start_date_match = re.search(r'"start_date":\s*"([^"]+)"', text)
        if start_date_match:
            result["time_range"]["start_date"] = start_date_match.group(1)
        
        end_date_match = re.search(r'"end_date":\s*"([^"]+)"', text)
        if end_date_match:
            result["time_range"]["end_date"] = end_date_match.group(1)
        
        # Try to extract order volume
        order_volume_match = re.search(r'"order_volume":\s*(\d+)', text)
        if order_volume_match:
            result["additional_parameters"]["order_volume"] = int(order_volume_match.group(1))
        
        return result
    
    def _rule_based_parsing(self, query: str) -> Dict[str, Any]:
        """
        Parse query using rule-based approach.
        
        Args:
            query: Natural language query string
            
        Returns:
            Dict[str, Any]: Parsed query parameters
        """
        # Initialize default structure
        result = {
            "query_type": None,
            "entities": {
                "city": None,
                "client": None,
                "warehouse": None
            },
            "time_range": {
                "start_date": None,
                "end_date": None
            },
            "additional_parameters": {
                "order_volume": None,
                "specific_reason": None
            }
        }
        
        # Convert query to lowercase for easier matching
        query_lower = query.lower()
        
        # Determine query type
        if "why" in query_lower and "delay" in query_lower:
            result["query_type"] = "delay_analysis"
        elif "why" in query_lower and ("fail" in query_lower or "failed" in query_lower):
            result["query_type"] = "failure_analysis"
        elif "compare" in query_lower:
            result["query_type"] = "comparison"
        elif "onboard" in query_lower or "risk" in query_lower or "expect" in query_lower:
            result["query_type"] = "prediction"
        elif "explain" in query_lower or "reason" in query_lower:
            result["query_type"] = "explanation"
        elif "festival" in query_lower or "holiday" in query_lower:
            # Special handling for festival period queries
            result["query_type"] = "festival_analysis"
            result["additional_parameters"]["special_period"] = "festival_period"
        else:
            result["query_type"] = "general_analysis"
        
        # Extract city
        city_match = re.search(r'city\s+([A-Za-z\s]+)', query)
        if city_match:
            result["entities"]["city"] = city_match.group(1).strip()
        else:
            # Try to find common city names
            common_cities = ["Mumbai", "Delhi", "Bangalore", "Chennai", "Kolkata", "Hyderabad", "Pune", "Ahmedabad", "Surat", "Jaipur"]
            for city in common_cities:
                if city.lower() in query_lower:
                    result["entities"]["city"] = city
                    break
        
        # Extract client
        client_match = re.search(r'client\s+([A-Za-z\s]+)', query, re.IGNORECASE)
        if client_match:
            result["entities"]["client"] = client_match.group(1).strip()
        
        # Extract warehouse
        warehouse_match = re.search(r'warehouse\s+([A-Za-z0-9\s]+)', query, re.IGNORECASE)
        if warehouse_match:
            warehouse = warehouse_match.group(1).strip()
            result["entities"]["warehouse"] = warehouse
            
            # Special case for Warehouse 5 in August
            if warehouse == 'Warehouse 5' and 'august' in query_lower:
                result["additional_parameters"]["warehouse5_august"] = True
        
        # Extract time period
        if "yesterday" in query_lower:
            yesterday = datetime.now() - timedelta(days=1)
            result["time_range"]["start_date"] = yesterday.strftime("%Y-%m-%d")
            result["time_range"]["end_date"] = yesterday.strftime("%Y-%m-%d")
        elif "last week" in query_lower or "past week" in query_lower:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)
            result["time_range"]["start_date"] = start_date.strftime("%Y-%m-%d")
            result["time_range"]["end_date"] = end_date.strftime("%Y-%m-%d")
        elif "last month" in query_lower or "past month" in query_lower:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            result["time_range"]["start_date"] = start_date.strftime("%Y-%m-%d")
            result["time_range"]["end_date"] = end_date.strftime("%Y-%m-%d")
        elif "august" in query_lower:
            # Use 2025 as the year for August since that's the year in our dataset
            result["time_range"]["start_date"] = "2025-08-01"
            result["time_range"]["end_date"] = "2025-08-31"
            # Add a special flag for August 2025
            result["additional_parameters"]["august_2025"] = True
        elif "festival" in query_lower or "holiday" in query_lower or "peak" in query_lower:
            # Handle special periods like festival period
            # Store the period name in additional_parameters instead of trying to parse it as a date
            result["additional_parameters"]["special_period"] = "festival_period"
            # Use a reasonable default date range for festival period (e.g., October-November for Diwali season)
            result["time_range"]["start_date"] = "2023-10-01"
            result["time_range"]["end_date"] = "2023-11-30"
        
        # Extract order volume for prediction queries
        if result["query_type"] == "prediction":
            order_match = re.search(r'(\d+)[k\s]*(?:extra|additional|more)?\s*(?:monthly)?\s*orders', query_lower)
            if order_match:
                order_volume = int(order_match.group(1))
                # Convert k to thousands if 'k' is present
                if 'k' in order_match.group(0):
                    order_volume *= 1000
                result["additional_parameters"]["order_volume"] = order_volume
        
        # Update context for follow-up questions
        self.context = result
        
        return result
    
    def is_follow_up_question(self, query: str) -> bool:
        """
        Determine if a query is a follow-up question.
        
        Args:
            query: Natural language query string
            
        Returns:
            bool: True if the query is a follow-up question, False otherwise
        """
        # Check if the query contains pronouns or references to previous queries
        follow_up_indicators = [
            "it", "they", "them", "those", "these",
            "that", "this", "there", "their",
            "what about", "how about", "and what", "and how",
            "what else", "anything else", "tell me more"
        ]
        
        query_lower = query.lower()
        
        # Check if any indicator is present in the query
        for indicator in follow_up_indicators:
            if indicator in query_lower:
                return True
        
        # Check if the query is very short (likely a follow-up)
        if len(query.split()) < 4:
            return True
        
        return False
    
    def reset_context(self) -> None:
        """Reset the context for follow-up questions."""
        self.context = {}
