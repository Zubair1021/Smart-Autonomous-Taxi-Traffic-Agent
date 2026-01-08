"""
Multi-objective route optimization using Pareto-optimal solutions.
Balances time, distance, fuel cost, and passenger satisfaction.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from agents.taxi import Taxi
from agents.passenger import Passenger
import networkx as nx


class MultiObjectiveOptimizer:
    """
    Optimizes routes considering multiple objectives simultaneously.
    """
    
    def __init__(self, time_weight: float = 0.3, distance_weight: float = 0.25,
                 fuel_weight: float = 0.2, satisfaction_weight: float = 0.25):
        """
        Initialize multi-objective optimizer.
        
        Args:
            time_weight: Weight for time minimization
            distance_weight: Weight for distance minimization
            fuel_weight: Weight for fuel cost minimization
            satisfaction_weight: Weight for passenger satisfaction maximization
        """
        self.time_weight = time_weight
        self.distance_weight = distance_weight
        self.fuel_weight = fuel_weight
        self.satisfaction_weight = satisfaction_weight
        
        # Normalize weights
        total = time_weight + distance_weight + fuel_weight + satisfaction_weight
        self.time_weight /= total
        self.distance_weight /= total
        self.fuel_weight /= total
        self.satisfaction_weight /= total
    
    def calculate_objectives(self, route: List[Tuple[int, int]], 
                           taxi: Taxi, passengers: List[Passenger],
                           road_network: nx.Graph) -> Dict[str, float]:
        """
        Calculate all objective values for a route.
        
        Args:
            route: List of positions in route
            taxi: Taxi agent
            passengers: Passengers to serve
            road_network: Road network graph
        
        Returns:
            Dictionary with objective values
        """
        objectives = {}
        
        # 1. Time objective (estimated travel time)
        total_time = len(route)  # Steps to complete route
        objectives['time'] = total_time
        
        # 2. Distance objective (total distance traveled)
        total_distance = 0
        for i in range(len(route) - 1):
            total_distance += abs(route[i][0] - route[i+1][0]) + \
                            abs(route[i][1] - route[i+1][1])
        objectives['distance'] = total_distance
        
        # 3. Fuel cost objective (distance * fuel_rate)
        fuel_rate = 0.1  # Cost per unit distance
        objectives['fuel_cost'] = total_distance * fuel_rate
        
        # 4. Passenger satisfaction (based on wait time and priority)
        satisfaction = 0.0
        for passenger in passengers:
            priority_value = {'emergency': 3.0, 'vip': 2.0, 'regular': 1.0}.get(
                passenger.priority, 1.0
            )
            # Higher wait time reduces satisfaction
            wait_penalty = passenger.wait_time * 0.1
            satisfaction += priority_value * (100 - wait_penalty)
        objectives['satisfaction'] = satisfaction / len(passengers) if passengers else 0.0
        
        return objectives
    
    def calculate_composite_score(self, objectives: Dict[str, float], 
                                 normalize: bool = True) -> float:
        """
        Calculate composite score from multiple objectives.
        Lower score is better (for minimization).
        
        Args:
            objectives: Dictionary of objective values
            normalize: Whether to normalize objectives
        
        Returns:
            Composite score
        """
        # Normalize objectives (optional, for fair comparison)
        if normalize:
            # Simple normalization (can be improved with historical min/max)
            max_time = 100
            max_distance = 100
            max_fuel = 20
            max_satisfaction = 300
            
            normalized = {
                'time': objectives['time'] / max_time,
                'distance': objectives['distance'] / max_distance,
                'fuel_cost': objectives['fuel_cost'] / max_fuel,
                'satisfaction': 1.0 - (objectives['satisfaction'] / max_satisfaction)  # Invert for minimization
            }
        else:
            normalized = objectives.copy()
            normalized['satisfaction'] = -objectives['satisfaction']  # Invert for minimization
        
        # Weighted sum
        score = (
            normalized['time'] * self.time_weight +
            normalized['distance'] * self.distance_weight +
            normalized['fuel_cost'] * self.fuel_weight +
            normalized['satisfaction'] * self.satisfaction_weight
        )
        
        return score
    
    def optimize_route(self, taxi: Taxi, passengers: List[Passenger],
                      road_network: nx.Graph) -> List[Tuple[int, int]]:
        """
        Find optimal route considering multiple objectives.
        
        Args:
            taxi: Taxi agent
            passengers: Passengers to pick up
            road_network: Road network graph
        
        Returns:
            Optimized route (list of positions)
        """
        if not passengers:
            return []
        
        current_pos = taxi.position
        remaining_passengers = passengers.copy()
        best_route = []
        
        # Try different route orderings (permutations for small sets)
        if len(passengers) <= 3:
            # Try all permutations
            from itertools import permutations
            best_score = float('inf')
            
            for perm in permutations(remaining_passengers):
                route = self._build_route_for_order(current_pos, perm, road_network)
                objectives = self.calculate_objectives(route, taxi, passengers, road_network)
                score = self.calculate_composite_score(objectives)
                
                if score < best_score:
                    best_score = score
                    best_route = route
        else:
            # For larger sets, use greedy with multi-objective evaluation
            route_positions = [current_pos]
            temp_pos = current_pos
            
            while remaining_passengers:
                best_next = None
                best_score = float('inf')
                
                for passenger in remaining_passengers:
                    # Build partial route
                    partial_route = self._build_route_for_order(
                        temp_pos, [passenger], road_network
                    )
                    partial_objectives = self.calculate_objectives(
                        partial_route, taxi, [passenger], road_network
                    )
                    score = self.calculate_composite_score(partial_objectives)
                    
                    if score < best_score:
                        best_score = score
                        best_next = passenger
                
                if best_next:
                    next_route = self._build_route_for_order(
                        temp_pos, [best_next], road_network
                    )
                    route_positions.extend(next_route[1:])  # Skip first (current pos)
                    temp_pos = best_next.position
                    remaining_passengers.remove(best_next)
            
            best_route = route_positions
        
        return best_route[1:] if len(best_route) > 1 else []  # Remove starting position
    
    def _build_route_for_order(self, start_pos: Tuple[int, int], 
                               passenger_order: List[Passenger],
                               road_network: nx.Graph) -> List[Tuple[int, int]]:
        """
        Build route following passenger order.
        
        Args:
            start_pos: Starting position
            passenger_order: Ordered list of passengers
            road_network: Road network
        
        Returns:
            Route as list of positions
        """
        route = [start_pos]
        current_pos = start_pos
        
        for passenger in passenger_order:
            # Find path to passenger
            if current_pos in road_network and passenger.position in road_network:
                try:
                    path = nx.shortest_path(road_network, current_pos, passenger.position)
                    route.extend(path[1:])  # Skip first (current pos)
                    current_pos = passenger.position
                except nx.NetworkXNoPath:
                    # Fallback: direct movement
                    route.append(passenger.position)
                    current_pos = passenger.position
        
        return route
    
    def find_pareto_optimal_routes(self, taxi: Taxi, passengers: List[Passenger],
                                   road_network: nx.Graph, max_solutions: int = 10) -> List[Dict]:
        """
        Find Pareto-optimal solutions (non-dominated routes).
        
        Args:
            taxi: Taxi agent
            passengers: Passengers to serve
            road_network: Road network
        
        Returns:
            List of Pareto-optimal route solutions with objectives
        """
        if not passengers:
            return []
        
        # Generate candidate routes (different orderings)
        from itertools import permutations
        
        solutions = []
        candidate_orders = list(permutations(passengers))[:max_solutions]  # Limit permutations
        
        for order in candidate_orders:
            route = self._build_route_for_order(taxi.position, list(order), road_network)
            objectives = self.calculate_objectives(route, taxi, passengers, road_network)
            score = self.calculate_composite_score(objectives)
            
            solutions.append({
                'route': route,
                'objectives': objectives,
                'score': score,
                'order': [p.unique_id for p in order]
            })
        
        # Find Pareto-optimal (non-dominated) solutions
        pareto_optimal = []
        for i, sol1 in enumerate(solutions):
            dominated = False
            for j, sol2 in enumerate(solutions):
                if i == j:
                    continue
                # Check if sol1 is dominated by sol2
                if (sol2['objectives']['time'] <= sol1['objectives']['time'] and
                    sol2['objectives']['distance'] <= sol1['objectives']['distance'] and
                    sol2['objectives']['fuel_cost'] <= sol1['objectives']['fuel_cost'] and
                    sol2['objectives']['satisfaction'] >= sol1['objectives']['satisfaction']):
                    # Check if at least one is strictly better
                    if (sol2['objectives']['time'] < sol1['objectives']['time'] or
                        sol2['objectives']['distance'] < sol1['objectives']['distance'] or
                        sol2['objectives']['fuel_cost'] < sol1['objectives']['fuel_cost'] or
                        sol2['objectives']['satisfaction'] > sol1['objectives']['satisfaction']):
                        dominated = True
                        break
            
            if not dominated:
                pareto_optimal.append(sol1)
        
        # Sort by composite score
        pareto_optimal.sort(key=lambda x: x['score'])
        
        return pareto_optimal

