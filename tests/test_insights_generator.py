"""
Unit tests for the insights generation module.
"""
import unittest
from src.insights_generation.generator import InsightsGenerator


class TestInsightsGenerator(unittest.TestCase):
    """Test cases for the InsightsGenerator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.insights_generator = InsightsGenerator()
    
    def test_generate_city_insights_empty_analysis(self):
        """Test generating city insights with empty analysis."""
        insights = self.insights_generator.generate_city_insights({})
        
        # Verify the result
        self.assertEqual(insights, {'insights': [], 'recommendations': []})
    
    def test_generate_city_insights(self):
        """Test generating city insights."""
        # Create a sample analysis result
        analysis = {
            'city': 'Mumbai',
            'order_count': 100,
            'delivery_failures': {
                'failure_count': 10,
                'failure_rate': 10.0,
                'failure_reasons': {
                    'categories': {
                        'Inventory': 5,
                        'Address': 3,
                        'Weather': 2
                    },
                    'counts': {
                        'Stockout': 5,
                        'Address not found': 3,
                        'Rain': 2
                    }
                },
                'external_factors': {
                    'impact_percentages': {
                        'traffic_issues': 30.0,
                        'weather_issues': 20.0,
                        'event_impacts': 10.0
                    }
                }
            },
            'delivery_delays': {
                'delay_count': 20,
                'delay_rate': 20.0,
                'delay_patterns': {
                    'statistics': {
                        'mean_delay_hours': 5.0,
                        'max_delay_hours': 24.0
                    },
                    'by_day_of_week': {
                        'Monday': 4.0,
                        'Tuesday': 3.0,
                        'Wednesday': 6.0,
                        'Thursday': 5.0,
                        'Friday': 7.0,
                        'Saturday': 4.0,
                        'Sunday': 3.0
                    }
                }
            }
        }
        
        insights = self.insights_generator.generate_city_insights(analysis)
        
        # Verify the result
        self.assertIsInstance(insights, dict)
        self.assertIn('insights', insights)
        self.assertIn('recommendations', insights)
        self.assertIsInstance(insights['insights'], list)
        self.assertIsInstance(insights['recommendations'], list)
        
        # Check that insights were generated
        self.assertGreater(len(insights['insights']), 0)
        
        # Check that specific insights were generated
        insight_texts = ' '.join(insights['insights'])
        self.assertIn('Mumbai', insight_texts)
        self.assertIn('10 orders failed', insight_texts)
        self.assertIn('10.00%', insight_texts)
        self.assertIn('Friday', insight_texts)
        
        # Check that recommendations were generated
        self.assertGreater(len(insights['recommendations']), 0)
        
        # Check that specific recommendations were generated
        recommendation_texts = ' '.join(insights['recommendations'])
        self.assertIn('Mumbai', recommendation_texts)
    
    def test_generate_client_insights(self):
        """Test generating client insights."""
        # Create a sample analysis result
        analysis = {
            'client_id': 1,
            'client_name': 'Test Client',
            'order_count': 50,
            'delivery_failures': {
                'failure_count': 5,
                'failure_rate': 10.0,
                'failure_reasons': {
                    'categories': {
                        'Inventory': 3,
                        'Address': 2
                    }
                },
                'feedback_analysis': {
                    'feedback_coverage': 80.0,
                    'average_rating': 3.5,
                    'common_feedback_keywords': {
                        'late': 10,
                        'damaged': 5,
                        'missing': 3
                    }
                }
            },
            'delivery_delays': {
                'delay_count': 10,
                'delay_rate': 20.0,
                'delay_patterns': {
                    'statistics': {
                        'mean_delay_hours': 4.0
                    }
                }
            }
        }
        
        insights = self.insights_generator.generate_client_insights(analysis)
        
        # Verify the result
        self.assertIsInstance(insights, dict)
        self.assertIn('insights', insights)
        self.assertIn('recommendations', insights)
        
        # Check that insights were generated
        self.assertGreater(len(insights['insights']), 0)
        
        # Check that specific insights were generated
        insight_texts = ' '.join(insights['insights'])
        self.assertIn('Test Client', insight_texts)
        self.assertIn('5 orders failed', insight_texts)
        self.assertIn('10.00%', insight_texts)
        self.assertIn('average rating of 3.5', insight_texts)
        
        # Check that recommendations were generated
        self.assertGreater(len(insights['recommendations']), 0)
    
    def test_generate_comparison_insights(self):
        """Test generating comparison insights."""
        # Create a sample comparison result
        comparison = {
            'cities': ['Mumbai', 'Delhi'],
            'order_counts': [100, 150],
            'failure_rates': [10.0, 5.0],
            'failure_rate_difference': 5.0,
            'delay_rates': [20.0, 15.0],
            'delay_rate_difference': 5.0,
            'city1_analysis': {
                'city': 'Mumbai',
                'delivery_failures': {
                    'external_factors': {
                        'impact_percentages': {
                            'traffic_issues': 30.0,
                            'weather_issues': 20.0
                        }
                    }
                }
            },
            'city2_analysis': {
                'city': 'Delhi',
                'delivery_failures': {
                    'external_factors': {
                        'impact_percentages': {
                            'traffic_issues': 15.0,
                            'weather_issues': 10.0
                        }
                    }
                }
            }
        }
        
        insights = self.insights_generator.generate_comparison_insights(comparison)
        
        # Verify the result
        self.assertIsInstance(insights, dict)
        self.assertIn('insights', insights)
        self.assertIn('recommendations', insights)
        
        # Check that insights were generated
        self.assertGreater(len(insights['insights']), 0)
        
        # Check that specific insights were generated
        insight_texts = ' '.join(insights['insights'])
        self.assertIn('Mumbai', insight_texts)
        self.assertIn('Delhi', insight_texts)
        self.assertIn('5.00%', insight_texts)
        
        # Check that recommendations were generated
        self.assertGreater(len(insights['recommendations']), 0)
    
    def test_generate_risk_prediction_insights(self):
        """Test generating risk prediction insights."""
        # Create a sample prediction result
        prediction = {
            'current_metrics': {
                'order_count': 1000,
                'failure_rate': 5.0,
                'delay_rate': 10.0
            },
            'projected_metrics': {
                'order_count': 21000,
                'failure_count': 1050,
                'delay_count': 2100,
                'failure_rate': 5.0,
                'delay_rate': 10.0
            },
            'risk_factors': [
                {
                    'type': 'warehouse',
                    'warehouse_id': 1,
                    'warehouse_name': 'Warehouse A',
                    'failure_rate': 15.0,
                    'order_count': 500
                },
                {
                    'type': 'city',
                    'city': 'Mumbai',
                    'failure_rate': 10.0,
                    'delay_rate': 15.0,
                    'order_count': 300
                }
            ],
            'bottlenecks': [
                {
                    'type': 'warehouse',
                    'warehouse_id': 1,
                    'warehouse_name': 'Warehouse A',
                    'current_orders': 500,
                    'estimated_additional_orders': 10000,
                    'avg_processing_time_minutes': 150.0,
                    'risk_level': 'high'
                },
                {
                    'type': 'driver',
                    'driver_id': 1,
                    'driver_name': 'Driver X',
                    'current_orders': 100,
                    'estimated_additional_orders': 2000,
                    'risk_level': 'high'
                }
            ]
        }
        
        insights = self.insights_generator.generate_risk_prediction_insights(prediction)
        
        # Verify the result
        self.assertIsInstance(insights, dict)
        self.assertIn('insights', insights)
        self.assertIn('recommendations', insights)
        
        # Check that insights were generated
        self.assertGreater(len(insights['insights']), 0)
        
        # Check that specific insights were generated
        insight_texts = ' '.join(insights['insights'])
        self.assertIn('20000', insight_texts)
        self.assertIn('1050', insight_texts)
        self.assertIn('Warehouse A', insight_texts)
        
        # Check that recommendations were generated
        self.assertGreater(len(insights['recommendations']), 0)


if __name__ == '__main__':
    unittest.main()
