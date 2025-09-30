"""
Insights generation module for the logistics insights engine.
"""
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class InsightsGenerator:
    """
    Insights generator class for generating human-readable insights and recommendations.
    """
    
    def __init__(self):
        """Initialize the insights generator."""
        pass
    
    def generate_city_insights(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate insights for city performance analysis.
        
        Args:
            analysis: City performance analysis results
            
        Returns:
            Dict[str, Any]: Generated insights
        """
        if not analysis:
            return {'insights': [], 'recommendations': []}
        
        city = analysis.get('city', 'Unknown')
        order_count = analysis.get('order_count', 0)
        
        # Initialize variables with default values
        failure_rate = 0
        delay_rate = 0
        
        insights = []
        recommendations = []
        
        # Generate insights about delivery failures
        failure_analysis = analysis.get('delivery_failures', {})
        if failure_analysis:
            failure_count = failure_analysis.get('failure_count', 0)
            failure_rate = failure_analysis.get('failure_rate', 0)
            
            if failure_count > 0:
                insights.append(f"In {city}, {failure_count} orders failed out of {order_count} total orders, resulting in a failure rate of {failure_rate:.2f}%.")
                
                # Add insights about failure reasons
                failure_reasons = failure_analysis.get('failure_reasons', {})
                if failure_reasons:
                    categories = failure_reasons.get('categories', {})
                    if categories:
                        top_categories = sorted(categories.items(), key=lambda x: x[1], reverse=True)[:3]
                        
                        if top_categories:
                            category_text = ", ".join([f"{cat} ({count} orders)" for cat, count in top_categories])
                            insights.append(f"The main failure categories in {city} were: {category_text}.")
                    
                    counts = failure_reasons.get('counts', {})
                    if counts:
                        top_reasons = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:3]
                        
                        if top_reasons:
                            reason_text = ", ".join([f"\"{reason}\" ({count} orders)" for reason, count in top_reasons])
                            insights.append(f"The top specific failure reasons were: {reason_text}.")
        
        # Generate insights about delivery delays
        delay_analysis = analysis.get('delivery_delays', {})
        if delay_analysis:
            delay_count = delay_analysis.get('delay_count', 0)
            delay_rate = delay_analysis.get('delay_rate', 0)
            
            if delay_count > 0:
                insights.append(f"In {city}, {delay_count} orders were delayed out of {order_count} total orders, resulting in a delay rate of {delay_rate:.2f}%.")
                
                # Add insights about delay patterns
                delay_patterns = delay_analysis.get('delay_patterns', {})
                if delay_patterns:
                    statistics = delay_patterns.get('statistics', {})
                    if statistics:
                        mean_delay = statistics.get('mean_delay_hours', 0)
                        max_delay = statistics.get('max_delay_hours', 0)
                        
                        insights.append(f"The average delay in {city} was {mean_delay:.2f} hours, with the longest delay being {max_delay:.2f} hours.")
                    
                    by_day = delay_patterns.get('by_day_of_week', {})
                    if by_day:
                        worst_day = max(by_day.items(), key=lambda x: x[1])
                        insights.append(f"{worst_day[0]} had the highest average delay of {worst_day[1]:.2f} hours in {city}.")
        
        # Generate insights about external factors
        external_factors = failure_analysis.get('external_factors', {}) or delay_analysis.get('external_factors', {})
        if external_factors:
            impact_percentages = external_factors.get('impact_percentages', {})
            if impact_percentages:
                traffic_impact = impact_percentages.get('traffic_issues', 0)
                weather_impact = impact_percentages.get('weather_issues', 0)
                event_impact = impact_percentages.get('event_impacts', 0)
                
                if traffic_impact > 20 or weather_impact > 20 or event_impact > 20:
                    impact_text = []
                    if traffic_impact > 20:
                        impact_text.append(f"traffic issues ({traffic_impact:.2f}%)")
                    if weather_impact > 20:
                        impact_text.append(f"weather conditions ({weather_impact:.2f}%)")
                    if event_impact > 20:
                        impact_text.append(f"local events ({event_impact:.2f}%)")
                    
                    if impact_text:
                        insights.append(f"External factors significantly impacting deliveries in {city} include: {', '.join(impact_text)}.")
        
        # Generate recommendations based on insights
        if failure_rate > 10:
            recommendations.append(f"Conduct a detailed review of delivery operations in {city} to address the high failure rate of {failure_rate:.2f}%.")
        
        if delay_rate > 15:
            recommendations.append(f"Optimize delivery routes in {city} to reduce the delay rate of {delay_rate:.2f}%.")
        
        # Add recommendations based on external factors
        if external_factors:
            impact_percentages = external_factors.get('impact_percentages', {})
            if impact_percentages:
                traffic_impact = impact_percentages.get('traffic_issues', 0)
                weather_impact = impact_percentages.get('weather_issues', 0)
                
                if traffic_impact > 25:
                    recommendations.append(f"Adjust delivery schedules in {city} to avoid peak traffic times, which are affecting {traffic_impact:.2f}% of orders.")
                
                if weather_impact > 25:
                    recommendations.append(f"Develop contingency plans for adverse weather in {city}, which is affecting {weather_impact:.2f}% of orders.")
        
        return {
            'insights': insights,
            'recommendations': recommendations
        }
    
    def generate_client_insights(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate insights for client performance analysis.
        
        Args:
            analysis: Client performance analysis results
            
        Returns:
            Dict[str, Any]: Generated insights
        """
        if not analysis:
            return {'insights': [], 'recommendations': []}
        
        client_name = analysis.get('client_name', f"Client {analysis.get('client_id', 'Unknown')}")
        order_count = analysis.get('order_count', 0)
        
        # Initialize variables with default values
        failure_rate = 0
        delay_rate = 0
        average_rating = 0
        
        insights = []
        recommendations = []
        
        # Generate insights about delivery failures
        failure_analysis = analysis.get('delivery_failures', {})
        if failure_analysis:
            failure_count = failure_analysis.get('failure_count', 0)
            failure_rate = failure_analysis.get('failure_rate', 0)
            
            if failure_count > 0:
                insights.append(f"For {client_name}, {failure_count} orders failed out of {order_count} total orders, resulting in a failure rate of {failure_rate:.2f}%.")
                
                # Add insights about failure reasons
                failure_reasons = failure_analysis.get('failure_reasons', {})
                if failure_reasons:
                    categories = failure_reasons.get('categories', {})
                    if categories:
                        top_categories = sorted(categories.items(), key=lambda x: x[1], reverse=True)[:3]
                        
                        if top_categories:
                            category_text = ", ".join([f"{cat} ({count} orders)" for cat, count in top_categories])
                            insights.append(f"The main failure categories for {client_name} were: {category_text}.")
        
        # Generate insights about delivery delays
        delay_analysis = analysis.get('delivery_delays', {})
        if delay_analysis:
            delay_count = delay_analysis.get('delay_count', 0)
            delay_rate = delay_analysis.get('delay_rate', 0)
            
            if delay_count > 0:
                insights.append(f"For {client_name}, {delay_count} orders were delayed out of {order_count} total orders, resulting in a delay rate of {delay_rate:.2f}%.")
                
                # Add insights about delay patterns
                delay_patterns = delay_analysis.get('delay_patterns', {})
                if delay_patterns:
                    statistics = delay_patterns.get('statistics', {})
                    if statistics:
                        mean_delay = statistics.get('mean_delay_hours', 0)
                        
                        insights.append(f"The average delay for {client_name} was {mean_delay:.2f} hours.")
        
        # Generate insights about customer feedback
        feedback_analysis = failure_analysis.get('feedback_analysis', {}) or delay_analysis.get('feedback_analysis', {})
        if feedback_analysis:
            feedback_coverage = feedback_analysis.get('feedback_coverage', 0)
            average_rating = feedback_analysis.get('average_rating', 0)
            
            if feedback_coverage > 0:
                insights.append(f"{feedback_coverage:.2f}% of {client_name}'s orders received customer feedback, with an average rating of {average_rating:.1f}/5.")
                
                common_keywords = feedback_analysis.get('common_feedback_keywords', {})
                if common_keywords:
                    keyword_text = ", ".join(list(common_keywords.keys())[:5])
                    insights.append(f"Common themes in customer feedback for {client_name} include: {keyword_text}.")
        
        # Generate recommendations based on insights
        if failure_rate > 8:
            recommendations.append(f"Work with {client_name} to address the high failure rate of {failure_rate:.2f}%, focusing on order accuracy and packaging.")
        
        if delay_rate > 12:
            recommendations.append(f"Review delivery processes for {client_name} to reduce the delay rate of {delay_rate:.2f}%.")
        
        if feedback_analysis and average_rating < 3.5:
            recommendations.append(f"Implement a customer satisfaction improvement plan for {client_name} to address the low average rating of {average_rating:.1f}/5.")
        
        return {
            'insights': insights,
            'recommendations': recommendations
        }
    
    def generate_warehouse_insights(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate insights for warehouse performance analysis.
        
        Args:
            analysis: Warehouse performance analysis results
            
        Returns:
            Dict[str, Any]: Generated insights
        """
        if not analysis:
            return {'insights': [], 'recommendations': []}
        
        warehouse_name = analysis.get('warehouse_name', f"Warehouse {analysis.get('warehouse_id', 'Unknown')}")
        order_count = analysis.get('order_count', 0)
        
        # Initialize variables with default values
        failure_rate = 0
        
        insights = []
        recommendations = []
        
        # Generate insights about delivery failures
        failure_analysis = analysis.get('delivery_failures', {})
        if failure_analysis:
            failure_count = failure_analysis.get('failure_count', 0)
            failure_rate = failure_analysis.get('failure_rate', 0)
            
            if failure_count > 0:
                insights.append(f"For {warehouse_name}, {failure_count} orders failed out of {order_count} total orders, resulting in a failure rate of {failure_rate:.2f}%.")
        
        # Generate insights about warehouse processing
        warehouse_processing = analysis.get('warehouse_processing', {})
        if warehouse_processing:
            processing_times = warehouse_processing.get('processing_times', {})
            if processing_times:
                picking_time = processing_times.get('picking_time_minutes', {}).get('mean', 0)
                dispatch_time = processing_times.get('dispatch_prep_time_minutes', {}).get('mean', 0)
                total_time = processing_times.get('total_warehouse_time_minutes', {}).get('mean', 0)
                
                if total_time > 0:
                    insights.append(f"{warehouse_name} takes an average of {total_time:.2f} minutes to process an order, with {picking_time:.2f} minutes for picking and {dispatch_time:.2f} minutes for dispatch preparation.")
            
            # Check for common issues in notes
            common_note_keywords = warehouse_processing.get('common_note_keywords', {})
            if common_note_keywords:
                keyword_text = ", ".join(list(common_note_keywords.keys())[:5])
                insights.append(f"Common issues noted in {warehouse_name} include: {keyword_text}.")
        
        # Generate recommendations based on insights
        if failure_rate > 5:
            recommendations.append(f"Investigate the high failure rate of {failure_rate:.2f}% at {warehouse_name}, focusing on inventory management and order accuracy.")
        
        if warehouse_processing:
            processing_times = warehouse_processing.get('processing_times', {})
            if processing_times:
                total_time = processing_times.get('total_warehouse_time_minutes', {}).get('mean', 0)
                
                if total_time > 120:  # If processing takes more than 2 hours
                    recommendations.append(f"Optimize warehouse operations at {warehouse_name} to reduce the average processing time of {total_time:.2f} minutes.")
        
        return {
            'insights': insights,
            'recommendations': recommendations
        }
    
    def generate_comparison_insights(self, comparison: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate insights for city comparison analysis.
        
        Args:
            comparison: City comparison analysis results
            
        Returns:
            Dict[str, Any]: Generated insights
        """
        if not comparison:
            return {'insights': [], 'recommendations': []}
        
        cities = comparison.get('cities', ['Unknown', 'Unknown'])
        city1, city2 = cities
        
        order_counts = comparison.get('order_counts', [0, 0])
        failure_rates = comparison.get('failure_rates', [0, 0])
        failure_rate_diff = comparison.get('failure_rate_difference', 0)
        delay_rates = comparison.get('delay_rates', [0, 0])
        delay_rate_diff = comparison.get('delay_rate_difference', 0)
        
        insights = []
        recommendations = []
        
        # Generate insights about order volumes
        insights.append(f"Comparing {city1} ({order_counts[0]} orders) and {city2} ({order_counts[1]} orders):")
        
        # Generate insights about failure rates
        if abs(failure_rate_diff) > 1:  # Only mention if difference is significant
            better_city = city1 if failure_rate_diff < 0 else city2
            worse_city = city2 if failure_rate_diff < 0 else city1
            diff = abs(failure_rate_diff)
            
            insights.append(f"{better_city} has a {diff:.2f}% lower failure rate than {worse_city} ({min(failure_rates):.2f}% vs {max(failure_rates):.2f}%).")
        else:
            insights.append(f"Both cities have similar failure rates: {city1} at {failure_rates[0]:.2f}% and {city2} at {failure_rates[1]:.2f}%.")
        
        # Generate insights about delay rates
        if abs(delay_rate_diff) > 1:  # Only mention if difference is significant
            better_city = city1 if delay_rate_diff < 0 else city2
            worse_city = city2 if delay_rate_diff < 0 else city1
            diff = abs(delay_rate_diff)
            
            insights.append(f"{better_city} has a {diff:.2f}% lower delay rate than {worse_city} ({min(delay_rates):.2f}% vs {max(delay_rates):.2f}%).")
        else:
            insights.append(f"Both cities have similar delay rates: {city1} at {delay_rates[0]:.2f}% and {city2} at {delay_rates[1]:.2f}%.")
        
        # Analyze root causes for differences
        city1_analysis = comparison.get('city1_analysis', {})
        city2_analysis = comparison.get('city2_analysis', {})
        
        # Compare external factors
        city1_external = city1_analysis.get('delivery_failures', {}).get('external_factors', {})
        city2_external = city2_analysis.get('delivery_failures', {}).get('external_factors', {})
        
        if city1_external and city2_external:
            city1_traffic = city1_external.get('impact_percentages', {}).get('traffic_issues', 0)
            city2_traffic = city2_external.get('impact_percentages', {}).get('traffic_issues', 0)
            
            if abs(city1_traffic - city2_traffic) > 10:
                higher_city = city1 if city1_traffic > city2_traffic else city2
                insights.append(f"Traffic has a significantly higher impact on deliveries in {higher_city}.")
            
            city1_weather = city1_external.get('impact_percentages', {}).get('weather_issues', 0)
            city2_weather = city2_external.get('impact_percentages', {}).get('weather_issues', 0)
            
            if abs(city1_weather - city2_weather) > 10:
                higher_city = city1 if city1_weather > city2_weather else city2
                insights.append(f"Weather conditions have a significantly higher impact on deliveries in {higher_city}.")
        
        # Generate recommendations based on insights
        worse_failure_city = city1 if failure_rates[0] > failure_rates[1] else city2
        if max(failure_rates) > 8:
            recommendations.append(f"Prioritize improvements in {worse_failure_city} to address the high failure rate of {max(failure_rates):.2f}%.")
        
        worse_delay_city = city1 if delay_rates[0] > delay_rates[1] else city2
        if max(delay_rates) > 12:
            recommendations.append(f"Optimize delivery routes and timing in {worse_delay_city} to address the high delay rate of {max(delay_rates):.2f}%.")
        
        # If one city is performing significantly better, recommend learning from it
        if abs(failure_rate_diff) > 5 or abs(delay_rate_diff) > 5:
            better_city = city1 if (failure_rates[0] + delay_rates[0]) < (failure_rates[1] + delay_rates[1]) else city2
            worse_city = city2 if better_city == city1 else city1
            
            recommendations.append(f"Conduct a detailed study of operational differences between {better_city} and {worse_city} to identify best practices that can be transferred.")
        
        return {
            'insights': insights,
            'recommendations': recommendations
        }
    
    def generate_risk_prediction_insights(self, prediction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate insights for risk prediction analysis.
        
        Args:
            prediction: Risk prediction analysis results
            
        Returns:
            Dict[str, Any]: Generated insights
        """
        if not prediction:
            return {'insights': [], 'recommendations': []}
        
        current_metrics = prediction.get('current_metrics', {})
        projected_metrics = prediction.get('projected_metrics', {})
        risk_factors = prediction.get('risk_factors', [])
        bottlenecks = prediction.get('bottlenecks', [])
        
        insights = []
        recommendations = []
        
        # Generate insights about current vs projected metrics
        current_orders = current_metrics.get('order_count', 0)
        projected_orders = projected_metrics.get('order_count', 0)
        additional_orders = projected_orders - current_orders
        
        insights.append(f"Adding {additional_orders} orders would increase the total from {current_orders} to {projected_orders} orders.")
        
        current_failure_rate = current_metrics.get('failure_rate', 0)
        projected_failure_rate = projected_metrics.get('failure_rate', 0)
        projected_failures = projected_metrics.get('failure_count', 0)
        
        insights.append(f"Based on the current failure rate of {current_failure_rate:.2f}%, we project {projected_failures} failed orders with a rate of {projected_failure_rate:.2f}%.")
        
        current_delay_rate = current_metrics.get('delay_rate', 0)
        projected_delay_rate = projected_metrics.get('delay_rate', 0)
        projected_delays = projected_metrics.get('delay_count', 0)
        
        insights.append(f"Based on the current delay rate of {current_delay_rate:.2f}%, we project {projected_delays} delayed orders with a rate of {projected_delay_rate:.2f}%.")
        
        # Generate insights about risk factors
        if risk_factors:
            risk_types = {}
            for risk in risk_factors:
                risk_type = risk.get('type', 'unknown')
                if risk_type not in risk_types:
                    risk_types[risk_type] = []
                risk_types[risk_type].append(risk)
            
            for risk_type, risks in risk_types.items():
                if risk_type == 'warehouse':
                    warehouse_names = [r.get('warehouse_name', f"Warehouse {r.get('warehouse_id', 'Unknown')}") for r in risks[:3]]
                    if warehouse_names:
                        insights.append(f"High-risk warehouses include: {', '.join(warehouse_names)}.")
                
                elif risk_type == 'route':
                    route_codes = [r.get('route_code', 'Unknown') for r in risks[:3]]
                    if route_codes:
                        insights.append(f"High-risk delivery routes include: {', '.join(route_codes)}.")
                
                elif risk_type == 'city':
                    city_names = [r.get('city', 'Unknown') for r in risks[:3]]
                    if city_names:
                        insights.append(f"High-risk cities include: {', '.join(city_names)}.")
        
        # Generate insights about bottlenecks
        if bottlenecks:
            bottleneck_types = {}
            for bottleneck in bottlenecks:
                bottleneck_type = bottleneck.get('type', 'unknown')
                if bottleneck_type not in bottleneck_types:
                    bottleneck_types[bottleneck_type] = []
                bottleneck_types[bottleneck_type].append(bottleneck)
            
            high_risk_bottlenecks = [b for b in bottlenecks if b.get('risk_level') == 'high']
            if high_risk_bottlenecks:
                insights.append(f"Identified {len(high_risk_bottlenecks)} high-risk bottlenecks that could impact performance with increased order volume.")
            
            for bottleneck_type, bottlenecks_list in bottleneck_types.items():
                if bottleneck_type == 'warehouse':
                    warehouse_names = [b.get('warehouse_name', f"Warehouse {b.get('warehouse_id', 'Unknown')}") for b in bottlenecks_list[:3]]
                    if warehouse_names:
                        insights.append(f"Potential warehouse bottlenecks: {', '.join(warehouse_names)}.")
                
                elif bottleneck_type == 'driver':
                    driver_names = [b.get('driver_name', f"Driver {b.get('driver_id', 'Unknown')}") for b in bottlenecks_list[:3]]
                    if driver_names:
                        insights.append(f"Potential driver bottlenecks: {', '.join(driver_names)}.")
                
                elif bottleneck_type == 'city':
                    city_names = [b.get('city', 'Unknown') for b in bottlenecks_list[:3]]
                    if city_names:
                        insights.append(f"Potential city bottlenecks: {', '.join(city_names)}.")
        
        # Generate recommendations based on insights
        if bottlenecks:
            warehouse_bottlenecks = [b for b in bottlenecks if b.get('type') == 'warehouse' and b.get('risk_level') == 'high']
            if warehouse_bottlenecks:
                warehouse_names = [b.get('warehouse_name', f"Warehouse {b.get('warehouse_id', 'Unknown')}") for b in warehouse_bottlenecks[:3]]
                recommendations.append(f"Increase capacity or improve efficiency at high-risk warehouses: {', '.join(warehouse_names)}.")
            
            driver_bottlenecks = [b for b in bottlenecks if b.get('type') == 'driver' and b.get('risk_level') == 'high']
            if driver_bottlenecks:
                recommendations.append(f"Onboard additional drivers to handle the increased order volume, as {len(driver_bottlenecks)} drivers are at high risk of overload.")
            
            city_bottlenecks = [b for b in bottlenecks if b.get('type') == 'city' and b.get('risk_level') == 'high']
            if city_bottlenecks:
                city_names = [b.get('city', 'Unknown') for b in city_bottlenecks[:3]]
                recommendations.append(f"Strengthen delivery infrastructure in high-risk cities: {', '.join(city_names)}.")
        
        # General recommendations
        if additional_orders > current_orders * 0.2:  # If adding more than 20% more orders
            recommendations.append("Implement a phased approach to onboarding the additional order volume to allow for operational adjustments.")
        
        if projected_failure_rate > current_failure_rate:
            recommendations.append("Develop contingency plans for handling an increased number of failed deliveries.")
        
        return {
            'insights': insights,
            'recommendations': recommendations
        }
    
    def generate_general_insights(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate general insights from analysis results.
        
        Args:
            analysis_results: Analysis results
            
        Returns:
            Dict[str, Any]: Generated insights
        """
        # Initialize variables with default values
        failure_rate = 0
        delay_rate = 0
        
        insights = []
        recommendations = []
        
        # Generate insights about overall delivery performance
        delivery_failures = analysis_results.get('delivery_failures', {})
        if delivery_failures:
            failure_count = delivery_failures.get('failure_count', 0)
            failure_rate = delivery_failures.get('failure_rate', 0)
            
            insights.append(f"Overall, there were {failure_count} failed deliveries, representing a failure rate of {failure_rate:.2f}%.")
            
            # Add insights about top failure reasons
            failure_reasons = delivery_failures.get('failure_reasons', {})
            if failure_reasons:
                categories = failure_reasons.get('categories', {})
                if categories:
                    top_categories = sorted(categories.items(), key=lambda x: x[1], reverse=True)[:3]
                    
                    if top_categories:
                        category_text = ", ".join([f"{cat} ({count} orders)" for cat, count in top_categories])
                        insights.append(f"The main failure categories were: {category_text}.")
        
        # Generate insights about delivery delays
        delivery_delays = analysis_results.get('delivery_delays', {})
        if delivery_delays:
            delay_count = delivery_delays.get('delay_count', 0)
            delay_rate = delivery_delays.get('delay_rate', 0)
            
            insights.append(f"Overall, there were {delay_count} delayed deliveries, representing a delay rate of {delay_rate:.2f}%.")
            
            # Add insights about delay patterns
            delay_patterns = delivery_delays.get('delay_patterns', {})
            if delay_patterns:
                statistics = delay_patterns.get('statistics', {})
                if statistics:
                    mean_delay = statistics.get('mean_delay_hours', 0)
                    median_delay = statistics.get('median_delay_hours', 0)
                    
                    insights.append(f"The average delay was {mean_delay:.2f} hours, with a median of {median_delay:.2f} hours.")
        
        # Generate recommendations based on insights
        if failure_rate > 5:
            recommendations.append(f"Implement a comprehensive failure reduction program to address the {failure_rate:.2f}% failure rate.")
        
        if delay_rate > 10:
            recommendations.append(f"Develop a delivery optimization strategy to reduce the {delay_rate:.2f}% delay rate.")
        
        # Add recommendation about data collection if needed
        if not delivery_failures and not delivery_delays:
            recommendations.append("Improve data collection and tracking to enable more detailed analysis of delivery performance.")
        
        return {
            'insights': insights,
            'recommendations': recommendations
        }
