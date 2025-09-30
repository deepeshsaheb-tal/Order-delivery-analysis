"""
Console user interface for the logistics insights engine.
"""
import sys
import logging
import argparse
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import textwrap

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ConsoleUI:
    """
    Console user interface class for the logistics insights engine.
    """
    
    def __init__(self):
        """Initialize the console UI."""
        self.terminal_width = 80
        try:
            # Try to get the terminal width
            import shutil
            self.terminal_width = shutil.get_terminal_size().columns
        except:
            # Fall back to default width
            pass
    
    def parse_arguments(self) -> argparse.Namespace:
        """
        Parse command line arguments.
        
        Returns:
            argparse.Namespace: Parsed arguments
        """
        parser = argparse.ArgumentParser(
            description='Logistics Insights Engine - Analyze logistics data to identify causes of delivery failures and delays'
        )
        
        parser.add_argument(
            'query',
            nargs='?',
            help='Natural language query (e.g., "Why were deliveries delayed in Mumbai yesterday?")',
            default=None
        )
        
        parser.add_argument(
            '--interactive',
            '-i',
            action='store_true',
            help='Start in interactive mode'
        )
        
        parser.add_argument(
            '--data-dir',
            '-d',
            help='Directory containing data files',
            default='Dataset'
        )
        
        parser.add_argument(
            '--verbose',
            '-v',
            action='store_true',
            help='Enable verbose output'
        )
        
        return parser.parse_args()
    
    def display_welcome(self) -> None:
        """Display welcome message."""
        self._print_header("Logistics Insights Engine")
        print("Analyze logistics data to identify causes of delivery failures and delays.")
        print()
        print("Type 'help' for available commands or 'exit' to quit.")
        print("Example query: Why were deliveries delayed in Mumbai yesterday?")
        print("=" * self.terminal_width)
        print()
    
    def display_help(self) -> None:
        """Display help information."""
        self._print_header("Available Commands")
        
        commands = [
            ("help", "Display this help information"),
            ("exit", "Exit the application"),
            ("clear", "Clear the screen"),
            ("reset", "Reset the conversation context"),
            ("Natural language query", "Ask a question about logistics data")
        ]
        
        for command, description in commands:
            print(f"  {command.ljust(20)} - {description}")
        
        print()
        self._print_header("Example Queries")
        
        examples = [
            "Why were deliveries delayed in Mumbai yesterday?",
            "Why did Client X's orders fail in the past week?",
            "Explain the top reasons for delivery failures linked to Warehouse B in August?",
            "Compare delivery failure causes between Mumbai and Delhi last month?",
            "What are the likely causes of delivery failures during the festival period, and how should we prepare?",
            "If we onboard Client Y with ~20,000 extra monthly orders, what new failure risks should we expect and how do we mitigate them?"
        ]
        
        for example in examples:
            print(f"  • {example}")
        
        print()
    
    def get_user_input(self, prompt: str = "Query: ") -> str:
        """
        Get input from the user.
        
        Args:
            prompt: Prompt to display
            
        Returns:
            str: User input
        """
        try:
            return input(prompt)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            sys.exit(0)
    
    def display_insights(self, insights: Dict[str, Any]) -> None:
        """
        Display insights to the user.
        
        Args:
            insights: Dictionary of insights
        """
        if not insights:
            print("No insights available.")
            return
        
        # Display insights
        if 'insights' in insights and insights['insights']:
            self._print_header("Insights")
            for i, insight in enumerate(insights['insights'], 1):
                print(f"  {i}. {self._wrap_text(insight)}")
            print()
        
        # Display recommendations
        if 'recommendations' in insights and insights['recommendations']:
            self._print_header("Recommendations")
            for i, recommendation in enumerate(insights['recommendations'], 1):
                print(f"  {i}. {self._wrap_text(recommendation)}")
            print()
    
    def display_error(self, message: str) -> None:
        """
        Display an error message.
        
        Args:
            message: Error message to display
        """
        print(f"\nERROR: {message}")
        print()
    
    def display_loading(self, message: str = "Processing your query...") -> None:
        """
        Display a loading message.
        
        Args:
            message: Loading message to display
        """
        print(f"\n{message}")
    
    def display_processing_steps(self, steps: List[str]) -> None:
        """
        Display processing steps.
        
        Args:
            steps: List of processing steps
        """
        print("\nProcessing steps:")
        for i, step in enumerate(steps, 1):
            print(f"  {i}. {step}")
        print()
    
    def clear_screen(self) -> None:
        """Clear the terminal screen."""
        print("\033c", end="")
    
    def _print_header(self, header: str) -> None:
        """
        Print a section header.
        
        Args:
            header: Header text
        """
        print(f"{header}")
        print("-" * len(header))
    
    def _wrap_text(self, text: str, indent: int = 4) -> str:
        """
        Wrap text to fit the terminal width.
        
        Args:
            text: Text to wrap
            indent: Number of spaces to indent wrapped lines
            
        Returns:
            str: Wrapped text
        """
        wrapper = textwrap.TextWrapper(
            width=self.terminal_width - indent,
            subsequent_indent=' ' * indent
        )
        return wrapper.fill(text)
