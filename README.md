# Logistics Insights Engine

## Project Overview
The Logistics Insights Engine is a console application that analyzes logistics data to identify causes of delivery failures and delays. The system aggregates multi-domain data, correlates events automatically, generates human-readable insights, and provides actionable recommendations.

## Features
- Data loading and preprocessing from multiple sources
- Correlation of events across different data domains
- Root cause analysis for delivery failures and delays
- Natural language insights generation
- Query processing for specific questions about logistics performance
- Interactive console interface for querying the system

## Data Sources
The system analyzes data from the following sources:
- Client information
- Driver details
- External factors (weather, traffic, events)
- Customer feedback
- Fleet logs
- Order data
- Warehouse logs
- Warehouse information

## Sample Use Cases
The system can answer questions such as:
- Why were deliveries delayed in city X yesterday?
- Why did Client X's orders fail in the past week?
- Explain the top reasons for delivery failures linked to Warehouse B in August?
- Compare delivery failure causes between City A and City B last month?
- What are the likely causes of delivery failures during the festival period, and how should we prepare?
- If we onboard Client Y with ~20,000 extra monthly orders, what new failure risks should we expect and how do we mitigate them?

## Installation

1. Clone the repository
2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Copy `.env.example` to `.env` and add your OpenAI API key:
   ```
   cp .env.example .env
   # Edit .env and add your OpenAI API key
   ```

## Usage
Run the main application:
```
python src/main.py
```

Or specify a query directly:
```
python src/main.py "Why were deliveries delayed in Mumbai yesterday?"
```

## Project Structure
```
logistics-insights-engine/
├── Dataset/                  # Data files
├── src/                      # Source code
│   ├── data_loading/         # Data loading modules
│   ├── preprocessing/        # Data preprocessing modules
│   ├── correlation_engine/   # Data correlation engine
│   ├── root_cause_analysis/  # Root cause analysis modules
│   ├── insights_generation/  # Insights generation modules
│   ├── query_processor/      # Natural language query processor
│   ├── ui/                   # Console user interface
│   └── main.py               # Main application entry point
├── tests/                    # Test files
├── .env.example              # Example environment variables
├── requirements.txt          # Project dependencies
└── README.md                 # Project documentation
```
