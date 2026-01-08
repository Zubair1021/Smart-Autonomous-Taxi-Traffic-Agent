"""
Fleet rebalancing algorithm to optimize taxi distribution.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from agents.taxi import Taxi
from agents.passenger import Passenger


class FleetRebalancer:
    """
    Optimizes taxi distribution by moving idle taxis to high-demand areas.
    """
    
    def __init__(self, rebalance_threshold: float = 0.3, min_idle_time: int = 10):
        """
        Initialize fleet rebalancer.
        
        Args:
            rebalance_threshold: Minimum demand difference to trigger rebalancing
            min_idle_time: Minimum steps a taxi must be idle before rebalancing
        """
        self.rebalance_threshold = rebalance_threshold
        self.min_idle_time = min_idle_time
        self.taxi_idle_times = {}  # taxi_id -> idle_steps
        self.rebalancing_targets = {}  # taxi_id -> target_position
    
    def analyze_demand_distribution(self, model) -> Dict[Tuple[int, int], float]:
        """
        Analyze current demand distribution across the city.
        
        Args:
            model: CityModel instance
        
        Returns:
            Dictionary mapping (x, y) positions to demand scores
        """
        demand_map = {}
        
        # Count passengers waiting at each position
        for agent in model.schedule.agents:
            if isinstance(agent, Passenger) and agent.status == "waiting":
                pos = agent.position
                if pos not in demand_map:
                    demand_map[pos] = 0
                # Weight by priority
                priority_weight = {'emergency': 3.0, 'vip': 2.0, 'regular': 1.0}.get(
                    agent.priority, 1.0
                )
                demand_map[pos] += priority_weight
        
        return demand_map
    
    def analyze_taxi_distribution(self, model) -> Dict[Tuple[int, int], int]:
        """
        Analyze current taxi distribution.
        
        Args:
            model: CityModel instance
        
        Returns:
            Dictionary mapping (x, y) positions to taxi count
        """
        taxi_map = {}
        
        for agent in model.schedule.agents:
            if isinstance(agent, Taxi):
                pos = agent.position
                if pos not in taxi_map:
                    taxi_map[pos] = 0
                taxi_map[pos] += 1
        
        return taxi_map
    
    def find_rebalancing_targets(self, model, demand_predictions: Optional[Dict] = None) -> List[Dict]:
        """
        Find optimal positions for idle taxis.
        
        Args:
            model: CityModel instance
            demand_predictions: Optional dictionary of predicted demand by position
        
        Returns:
            List of rebalancing actions: [{'taxi_id': id, 'target': (x, y), 'reason': str}]
        """
        actions = []
        
        # Get current demand distribution
        demand_map = self.analyze_demand_distribution(model)
        
        # Use predictions if available
        if demand_predictions:
            for pos, predicted_demand in demand_predictions.items():
                if pos not in demand_map:
                    demand_map[pos] = 0
                demand_map[pos] += predicted_demand * 0.5  # Blend with current demand
        
        # Get taxi distribution
        taxi_map = self.analyze_taxi_distribution(model)
        
        # Find idle taxis
        idle_taxis = []
        for agent in model.schedule.agents:
            if isinstance(agent, Taxi) and agent.status == "idle":
                # Track idle time
                if agent.unique_id not in self.taxi_idle_times:
                    self.taxi_idle_times[agent.unique_id] = 0
                else:
                    self.taxi_idle_times[agent.unique_id] += 1
                
                if self.taxi_idle_times[agent.unique_id] >= self.min_idle_time:
                    idle_taxis.append(agent)
        
        if not idle_taxis or not demand_map:
            return actions
        
        # Find high-demand, low-taxi areas
        high_demand_areas = []
        for pos, demand in demand_map.items():
            taxi_count = taxi_map.get(pos, 0)
            # Calculate demand-to-taxi ratio
            if taxi_count == 0:
                ratio = demand * 10  # Very high if no taxis
            else:
                ratio = demand / taxi_count
            
            if ratio > self.rebalance_threshold:
                high_demand_areas.append({
                    'position': pos,
                    'demand': demand,
                    'taxi_count': taxi_count,
                    'ratio': ratio
                })
        
        # Sort by ratio (highest first)
        high_demand_areas.sort(key=lambda x: x['ratio'], reverse=True)
        
        # Assign idle taxis to high-demand areas
        for taxi in idle_taxis[:len(high_demand_areas)]:
            if high_demand_areas:
                target_area = high_demand_areas.pop(0)
                target_pos = target_area['position']
                
                # Check if taxi is already near this area
                current_pos = taxi.position
                distance = abs(current_pos[0] - target_pos[0]) + abs(current_pos[1] - target_pos[1])
                
                # Only rebalance if far enough away
                if distance > 5:
                    actions.append({
                        'taxi_id': taxi.unique_id,
                        'target': target_pos,
                        'reason': f"High demand area (demand: {target_area['demand']:.1f}, ratio: {target_area['ratio']:.2f})"
                    })
                    self.rebalancing_targets[taxi.unique_id] = target_pos
        
        return actions
    
    def execute_rebalancing(self, model, actions: List[Dict]):
        """
        Execute rebalancing actions by updating taxi paths.
        
        Args:
            model: CityModel instance
            actions: List of rebalancing actions
        """
        for action in actions:
            taxi_id = action['taxi_id']
            target_pos = action['target']
            
            # Find the taxi
            for agent in model.schedule.agents:
                if isinstance(agent, Taxi) and agent.unique_id == taxi_id:
                    if agent.status == "idle":
                        # Set rebalancing target
                        agent.rebalancing_target = target_pos
                        # Calculate path to target
                        if hasattr(agent, '_find_path'):
                            path = agent._find_path(agent.position, target_pos)
                            if path:
                                agent.path = path
                                agent.status = "rebalancing"
                    break
    
    def update_idle_times(self, model):
        """Update idle time tracking for all taxis."""
        for agent in model.schedule.agents:
            if isinstance(agent, Taxi):
                if agent.status == "idle":
                    if agent.unique_id not in self.taxi_idle_times:
                        self.taxi_idle_times[agent.unique_id] = 0
                    else:
                        self.taxi_idle_times[agent.unique_id] += 1
                else:
                    # Reset idle time when taxi becomes active
                    if agent.unique_id in self.taxi_idle_times:
                        self.taxi_idle_times[agent.unique_id] = 0
    
    def should_rebalance(self, model) -> bool:
        """
        Determine if rebalancing should occur.
        
        Args:
            model: CityModel instance
        
        Returns:
            True if rebalancing is recommended
        """
        # Check if there are idle taxis
        idle_taxis = [a for a in model.schedule.agents 
                     if isinstance(a, Taxi) and a.status == "idle"]
        
        if len(idle_taxis) == 0:
            return False
        
        # Check if there are waiting passengers
        waiting_passengers = [a for a in model.schedule.agents 
                            if isinstance(a, Passenger) and a.status == "waiting"]
        
        if len(waiting_passengers) == 0:
            return False
        
        # Check if any taxi has been idle long enough
        for taxi in idle_taxis:
            idle_time = self.taxi_idle_times.get(taxi.unique_id, 0)
            if idle_time >= self.min_idle_time:
                return True
        
        return False

