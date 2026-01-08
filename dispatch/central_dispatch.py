"""
Central Dispatch System for optimal global taxi-passenger assignment.
Uses Hungarian algorithm approximation for optimal matching.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from agents.taxi import Taxi
from agents.passenger import Passenger


class CentralDispatchSystem:
    """
    Central dispatch system that optimally assigns taxis to passengers globally.
    Replaces decentralized taxi-selection with centralized optimization.
    """
    
    def __init__(self, assignment_method: str = "hungarian_approx"):
        """
        Initialize central dispatch system.
        
        Args:
            assignment_method: "hungarian_approx" or "greedy_priority"
        """
        self.assignment_method = assignment_method
        self.assignments = {}  # passenger_id -> taxi_id
        self.assignment_history = []  # Track all assignments
    
    def create_cost_matrix(self, taxis: List[Taxi], passengers: List[Passenger]) -> np.ndarray:
        """
        Create cost matrix for assignment problem.
        Lower cost = better match.
        
        Args:
            taxis: List of available taxis
            passengers: List of waiting passengers
        
        Returns:
            Cost matrix: rows = taxis, columns = passengers
        """
        # Initialize cost matrix
        cost_matrix = np.full((len(taxis), len(passengers)), np.inf)
        
        for i, taxi in enumerate(taxis):
            # Check if taxi has capacity
            available_capacity = taxi.capacity - len(taxi.passengers)
            if available_capacity <= 0:
                continue  # Taxi is full, skip
            
            for j, passenger in enumerate(passengers):
                # Skip if passenger already assigned
                if passenger.unique_id in self.assignments:
                    continue
                
                # Calculate distance (Manhattan)
                distance = abs(taxi.position[0] - passenger.position[0]) + \
                          abs(taxi.position[1] - passenger.position[1])
                
                # Cost = distance + priority penalty + wait time penalty
                # Lower is better
                priority_penalty = {'emergency': 0, 'vip': 5, 'regular': 10}.get(
                    passenger.priority, 10
                )
                wait_penalty = passenger.wait_time * 0.1
                
                cost = distance + priority_penalty + wait_penalty
                
                # Adjust for taxi capacity (prefer taxis with more space for shared rides)
                capacity_bonus = (taxi.capacity - available_capacity) * 2
                cost += capacity_bonus
                
                cost_matrix[i, j] = cost
        
        return cost_matrix
    
    def hungarian_approximation(self, cost_matrix: np.ndarray, 
                               taxis: List[Taxi], passengers: List[Passenger]) -> List[Tuple[int, int]]:
        """
        Approximate Hungarian algorithm for optimal assignment.
        Uses greedy matching with priority ordering.
        
        Args:
            cost_matrix: Cost matrix
            taxis: List of taxis
            passengers: List of passengers
        
        Returns:
            List of (taxi_index, passenger_index) assignments
        """
        assignments = []
        used_taxis = set()
        used_passengers = set()
        
        # Sort all possible assignments by cost
        possible_assignments = []
        for i in range(len(taxis)):
            for j in range(len(passengers)):
                if cost_matrix[i, j] < np.inf:
                    possible_assignments.append((cost_matrix[i, j], i, j))
        
        # Sort by cost (lowest first)
        possible_assignments.sort(key=lambda x: x[0])
        
        # Greedy assignment
        for cost, taxi_idx, passenger_idx in possible_assignments:
            if taxi_idx not in used_taxis and passenger_idx not in used_passengers:
                assignments.append((taxi_idx, passenger_idx))
                used_taxis.add(taxi_idx)
                used_passengers.add(passenger_idx)
        
        return assignments
    
    def optimal_assignments(self, taxis: List[Taxi], passengers: List[Passenger]) -> Dict[str, str]:
        """
        Find optimal global assignments.
        
        Args:
            taxis: List of available taxis
            passengers: List of waiting passengers
        
        Returns:
            Dictionary mapping passenger_id -> taxi_id
        """
        if not taxis or not passengers:
            return {}
        
        # Filter available taxis (have capacity)
        available_taxis = [t for t in taxis if len(t.passengers) < t.capacity]
        if not available_taxis:
            return {}
        
        # Filter unassigned passengers
        unassigned_passengers = [p for p in passengers if p.unique_id not in self.assignments]
        if not unassigned_passengers:
            return {}
        
        # Sort passengers by priority (process high priority first)
        unassigned_passengers.sort(
            key=lambda p: (
                {'emergency': 3, 'vip': 2, 'regular': 1}.get(p.priority, 1),
                -p.wait_time
            ),
            reverse=True
        )
        
        new_assignments = {}
        
        if self.assignment_method == "hungarian_approx":
            # Create cost matrix
            cost_matrix = self.create_cost_matrix(available_taxis, unassigned_passengers)
            
            # Find optimal assignments
            matched_pairs = self.hungarian_approximation(cost_matrix, available_taxis, unassigned_passengers)
            
            for taxi_idx, passenger_idx in matched_pairs:
                taxi = available_taxis[taxi_idx]
                passenger = unassigned_passengers[passenger_idx]
                new_assignments[passenger.unique_id] = taxi.unique_id
        else:
            # Greedy priority-based matching
            for passenger in unassigned_passengers:
                best_taxi = None
                best_cost = np.inf
                
                for taxi in available_taxis:
                    # Check if taxi already assigned in this round
                    if taxi.unique_id in new_assignments.values():
                        continue
                    
                    distance = abs(taxi.position[0] - passenger.position[0]) + \
                              abs(taxi.position[1] - passenger.position[1])
                    
                    priority_penalty = {'emergency': 0, 'vip': 5, 'regular': 10}.get(
                        passenger.priority, 10
                    )
                    wait_penalty = passenger.wait_time * 0.1
                    
                    cost = distance + priority_penalty + wait_penalty
                    
                    if cost < best_cost:
                        best_cost = cost
                        best_taxi = taxi
                
                if best_taxi:
                    new_assignments[passenger.unique_id] = best_taxi.unique_id
        
        # Update assignments
        self.assignments.update(new_assignments)
        
        # Track in history
        for passenger_id, taxi_id in new_assignments.items():
            self.assignment_history.append({
                'passenger_id': passenger_id,
                'taxi_id': taxi_id,
                'method': self.assignment_method
            })
        
        return new_assignments
    
    def execute_assignments(self, model, assignments: Dict[str, str]):
        """
        Execute assignments by updating taxi paths.
        
        Args:
            model: CityModel instance
            assignments: Dictionary of passenger_id -> taxi_id
        """
        for passenger_id, taxi_id in assignments.items():
            # Find passenger and taxi
            passenger = None
            taxi = None
            
            for agent in model.schedule.agents:
                if isinstance(agent, Passenger) and str(agent.unique_id) == str(passenger_id):
                    passenger = agent
                elif isinstance(agent, Taxi) and agent.unique_id == taxi_id:
                    taxi = agent
            
            if passenger and taxi and taxi.status == "idle":
                # Check capacity
                if len(taxi.passengers) < taxi.capacity:
                    # Set taxi to pick up passenger
                    taxi.status = "picking_up"
                    pickup_location = passenger.position
                    taxi.path = taxi._find_path(taxi.position, pickup_location)
    
    def clear_completed_assignments(self, model):
        """Clear assignments for completed trips."""
        completed = []
        for passenger_id in list(self.assignments.keys()):
            # Check if passenger still exists or is no longer waiting
            passenger = None
            for agent in model.schedule.agents:
                if isinstance(agent, Passenger) and str(agent.unique_id) == str(passenger_id):
                    passenger = agent
                    break
            
            if not passenger or passenger.status != "waiting":
                completed.append(passenger_id)
        
        for passenger_id in completed:
            del self.assignments[passenger_id]
    
    def get_assignment_statistics(self) -> Dict:
        """Get statistics about assignments."""
        return {
            'total_assignments': len(self.assignment_history),
            'active_assignments': len(self.assignments),
            'method': self.assignment_method
        }

