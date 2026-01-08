"""
What-If Scenario Analysis for testing different configurations and strategies.
"""

import copy
from typing import Dict, List, Optional, Any
from model.city_model import CityModel
from database.db_manager import DatabaseManager
import pandas as pd


class ScenarioAnalyzer:
    """
    Analyzes different scenarios to predict outcomes.
    """
    
    def __init__(self, base_config: Dict):
        """
        Initialize scenario analyzer.
        
        Args:
            base_config: Base simulation configuration
        """
        self.base_config = base_config
        self.scenario_results = []
    
    def run_scenario(self, scenario_name: str, config_overrides: Dict,
                    steps: int = 100) -> Dict:
        """
        Run a what-if scenario simulation.
        
        Args:
            scenario_name: Name of scenario
            config_overrides: Configuration changes from base
            steps: Number of simulation steps to run
        
        Returns:
            Scenario results dictionary
        """
        # Create modified config
        scenario_config = copy.deepcopy(self.base_config)
        scenario_config.update(config_overrides)
        
        # Create temporary model
        model = CityModel(**scenario_config)
        
        # Run simulation for specified steps
        for _ in range(steps):
            model.step()
            if not model.running:
                break
        
        # Collect results
        results = {
            'scenario_name': scenario_name,
            'config': scenario_config,
            'final_step': model.schedule.time,
            'metrics': {
                'total_passengers_served': sum(t.total_passengers for t in model.schedule.agents 
                                              if hasattr(t, 'total_passengers')),
                'avg_wait_time': model._calculate_avg_wait_time(),
                'taxi_utilization': model._calculate_taxi_utilization(),
                'total_revenue': sum(getattr(t, 'revenue', 0) for t in model.schedule.agents 
                                   if hasattr(t, 'revenue')),
                'waiting_passengers': len([a for a in model.schedule.agents 
                                         if hasattr(a, 'status') and a.status == "waiting"])
            }
        }
        
        self.scenario_results.append(results)
        
        return results
    
    def compare_scenarios(self, scenario_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Compare multiple scenarios.
        
        Args:
            scenario_names: List of scenario names to compare (None = all)
        
        Returns:
            DataFrame with comparison results
        """
        if scenario_names:
            scenarios = [s for s in self.scenario_results if s['scenario_name'] in scenario_names]
        else:
            scenarios = self.scenario_results
        
        if not scenarios:
            return pd.DataFrame()
        
        comparison_data = []
        for scenario in scenarios:
            row = {
                'Scenario': scenario['scenario_name'],
                'Steps': scenario['final_step'],
                'Passengers Served': scenario['metrics']['total_passengers_served'],
                'Avg Wait Time': scenario['metrics']['avg_wait_time'],
                'Utilization %': scenario['metrics']['taxi_utilization'],
                'Revenue': scenario['metrics']['total_revenue'],
                'Waiting': scenario['metrics']['waiting_passengers']
            }
            # Add config differences
            for key, value in scenario['config'].items():
                if key not in self.base_config or self.base_config[key] != value:
                    row[f'Config: {key}'] = value
            
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def analyze_fleet_size_impact(self, min_taxis: int = 5, max_taxis: int = 20,
                                  step_size: int = 5, simulation_steps: int = 100) -> pd.DataFrame:
        """
        Analyze impact of different fleet sizes.
        
        Args:
            min_taxis: Minimum number of taxis
            max_taxis: Maximum number of taxis
            step_size: Increment step
            simulation_steps: Steps per scenario
        
        Returns:
            DataFrame with fleet size analysis
        """
        results = []
        
        for num_taxis in range(min_taxis, max_taxis + 1, step_size):
            result = self.run_scenario(
                f"Fleet Size: {num_taxis}",
                {'num_taxis': num_taxis},
                steps=simulation_steps
            )
            results.append(result)
        
        return self.compare_scenarios([r['scenario_name'] for r in results])
    
    def analyze_rebalancing_impact(self, simulation_steps: int = 100) -> pd.DataFrame:
        """
        Compare scenarios with and without rebalancing.
        
        Args:
            simulation_steps: Steps per scenario
        
        Returns:
            DataFrame with comparison
        """
        scenarios = [
            ("No Rebalancing", {'enable_rebalancing': False}),
            ("With Rebalancing", {'enable_rebalancing': True})
        ]
        
        for name, config in scenarios:
            self.run_scenario(name, config, steps=simulation_steps)
        
        return self.compare_scenarios([name for name, _ in scenarios])
    
    def analyze_prediction_impact(self, simulation_steps: int = 100) -> pd.DataFrame:
        """
        Compare scenarios with and without demand prediction.
        
        Args:
            simulation_steps: Steps per scenario
        
        Returns:
            DataFrame with comparison
        """
        scenarios = [
            ("No Prediction", {'enable_prediction': False}),
            ("With Prediction", {'enable_prediction': True})
        ]
        
        for name, config in scenarios:
            self.run_scenario(name, config, steps=simulation_steps)
        
        return self.compare_scenarios([name for name, _ in scenarios])
    
    def get_recommendations(self) -> List[Dict]:
        """
        Generate recommendations based on scenario analysis.
        
        Returns:
            List of recommendation dictionaries
        """
        recommendations = []
        
        if not self.scenario_results:
            return recommendations
        
        # Compare scenarios
        comparison_df = self.compare_scenarios()
        
        if comparison_df.empty:
            return recommendations
        
        # Find best scenario for each metric
        best_wait = comparison_df.loc[comparison_df['Avg Wait Time'].idxmin()]
        best_util = comparison_df.loc[comparison_df['Utilization %'].idxmax()]
        best_revenue = comparison_df.loc[comparison_df['Revenue'].idxmax()]
        
        recommendations.append({
            'type': 'wait_time',
            'message': f"Best wait time: {best_wait['Scenario']} ({best_wait['Avg Wait Time']:.2f} steps)",
            'scenario': best_wait['Scenario']
        })
        
        recommendations.append({
            'type': 'utilization',
            'message': f"Best utilization: {best_util['Scenario']} ({best_util['Utilization %']:.2f}%)",
            'scenario': best_util['Scenario']
        })
        
        recommendations.append({
            'type': 'revenue',
            'message': f"Best revenue: {best_revenue['Scenario']} (${best_revenue['Revenue']:.2f})",
            'scenario': best_revenue['Scenario']
        })
        
        return recommendations

