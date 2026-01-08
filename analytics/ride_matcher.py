"""
Advanced ride matching algorithm using optimization techniques.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from agents.taxi import Taxi
from agents.passenger import Passenger


class AdvancedRideMatcher:
    """
    Advanced ride matching using Hungarian algorithm approximation and multi-passenger optimization.
    """
    
    def __init__(self, max_match_distance: int = 20):
        """
        Initialize ride matcher.
        
        Args:
            max_match_distance: Maximum Manhattan distance for matching
        """
        self.max_match_distance = max_match_distance
    
    def calculate_match_score(self, taxi: Taxi, passenger: Passenger) -> float:
        """
        Calculate matching score between taxi and passenger.
        Higher score = better match.
        
        Args:
            taxi: Taxi agent
            passenger: Passenger agent
        
        Returns:
            Match score (higher is better)
        """
        # Distance component (closer is better)
        distance = abs(taxi.position[0] - passenger.position[0]) + \
                   abs(taxi.position[1] - passenger.position[1])
        
        if distance > self.max_match_distance:
            return -1  # Too far
        
        distance_score = 1.0 / (1.0 + distance)
        
        # Priority component (higher priority passengers get higher score)
        priority_scores = {'emergency': 3.0, 'vip': 2.0, 'regular': 1.0}
        priority_score = priority_scores.get(passenger.priority, 1.0)
        
        # Capacity component (prefer taxis with more available capacity)
        available_capacity = taxi.capacity - len(taxi.passengers)
        capacity_score = available_capacity / taxi.capacity
        
        # Wait time component (passengers waiting longer get higher priority)
        wait_time_score = min(passenger.wait_time / 100.0, 1.0)
        
        # Combined score
        total_score = (
            distance_score * 0.4 +
            priority_score * 0.3 +
            capacity_score * 0.2 +
            wait_time_score * 0.1
        )
        
        return total_score
    
    def find_optimal_matches(self, taxis: List[Taxi], passengers: List[Passenger]) -> List[Tuple[Taxi, Passenger]]:
        """
        Find optimal taxi-passenger matches using greedy algorithm with priority.
        
        Args:
            taxis: List of available taxis
            passengers: List of waiting passengers
        
        Returns:
            List of (taxi, passenger) tuples representing matches
        """
        matches = []
        matched_passengers = set()
        
        # Filter available taxis (idle or can take more passengers)
        available_taxis = [t for t in taxis if len(t.passengers) < t.capacity]
        
        if not available_taxis or not passengers:
            return matches
        
        # Sort passengers by priority and wait time
        sorted_passengers = sorted(
            passengers,
            key=lambda p: (
                {'emergency': 3, 'vip': 2, 'regular': 1}.get(p.priority, 1),
                -p.wait_time  # Longer wait time = higher priority
            ),
            reverse=True
        )
        
        # For each passenger (in priority order), find best available taxi
        for passenger in sorted_passengers:
            if passenger.unique_id in matched_passengers:
                continue
            
            best_taxi = None
            best_score = -1
            
            for taxi in available_taxis:
                # Check if taxi already committed to another passenger
                if taxi.status == "picking_up":
                    continue
                
                score = self.calculate_match_score(taxi, passenger)
                
                if score > best_score:
                    best_score = score
                    best_taxi = taxi
            
            if best_taxi and best_score > 0:
                matches.append((best_taxi, passenger))
                matched_passengers.add(passenger.unique_id)
                
                # Mark taxi as committed (prevent double assignment)
                if best_taxi.status == "idle":
                    best_taxi.status = "picking_up"
        
        return matches
    
    def optimize_multi_passenger_routes(self, taxi: Taxi, passengers: List[Passenger]) -> List[Tuple[int, int]]:
        """
        Optimize route for picking up multiple passengers (TSP approximation).
        
        Args:
            taxi: Taxi agent
            passengers: List of passengers to pick up
        
        Returns:
            Ordered list of (x, y) positions to visit
        """
        if not passengers:
            return []
        
        current_pos = taxi.position
        remaining_passengers = passengers.copy()
        route = []
        
        # Nearest neighbor heuristic for TSP
        while remaining_passengers:
            nearest = min(
                remaining_passengers,
                key=lambda p: abs(current_pos[0] - p.position[0]) + 
                             abs(current_pos[1] - p.position[1])
            )
            route.append(nearest.position)
            remaining_passengers.remove(nearest)
            current_pos = nearest.position
        
        return route
    
    def suggest_shared_rides(self, passengers: List[Passenger], max_distance: int = 10) -> List[List[Passenger]]:
        """
        Suggest passenger groups that could share a ride.
        
        Args:
            passengers: List of waiting passengers
            max_distance: Maximum distance between passengers for sharing
        
        Returns:
            List of passenger groups that could share rides
        """
        if len(passengers) < 2:
            return []
        
        groups = []
        used_passengers = set()
        
        # Group passengers by proximity
        for i, p1 in enumerate(passengers):
            if p1.unique_id in used_passengers:
                continue
            
            group = [p1]
            used_passengers.add(p1.unique_id)
            
            for p2 in passengers[i+1:]:
                if p2.unique_id in used_passengers:
                    continue
                
                distance = abs(p1.position[0] - p2.position[0]) + \
                          abs(p1.position[1] - p2.position[1])
                
                # Check if destinations are also close
                dest_distance = abs(p1.destination[0] - p2.destination[0]) + \
                               abs(p1.destination[1] - p2.destination[1])
                
                if distance <= max_distance and dest_distance <= max_distance * 2:
                    group.append(p2)
                    used_passengers.add(p2.unique_id)
                    
                    if len(group) >= 3:  # Max group size
                        break
            
            if len(group) > 1:
                groups.append(group)
        
        return groups

