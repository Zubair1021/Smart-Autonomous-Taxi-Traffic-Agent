"""
Taxi agent with intelligent routing and congestion avoidance.
"""

import random
from mesa import Agent
import networkx as nx


class Taxi(Agent):
    """
    Intelligent taxi agent with ride sharing, multiple types, and enhanced routing.
    """
    
    def __init__(self, unique_id, model, position, taxi_type="economy"):
        super().__init__(unique_id, model)
        self.position = position
        self.taxi_type = taxi_type  # "economy", "premium", "luxury"
        self.passengers = []  # List of passengers (ride sharing support)
        self.destinations = []  # List of destinations to visit
        self.path = []  # Current path to follow
        self.status = "idle"  # idle, picking_up, transporting, rebalancing
        self.total_distance = 0
        self.total_passengers = 0
        self.revenue = 0.0
        self.capacity = self._get_capacity()
        self.speed = self._get_speed()
        self.rebalancing_target = None  # Target position for rebalancing
        
    def _get_capacity(self):
        """Get passenger capacity based on taxi type."""
        capacities = {
            "economy": 1,
            "premium": 2,
            "luxury": 3
        }
        return capacities.get(self.taxi_type, 1)
    
    def _get_speed(self):
        """Get movement speed based on taxi type."""
        speeds = {
            "economy": 1,  # Normal speed
            "premium": 1,  # Same speed but better service
            "luxury": 1    # Same speed
        }
        return speeds.get(self.taxi_type, 1)
        
    def step(self):
        """Execute one step of taxi behavior."""
        # Handle rebalancing
        if self.status == "rebalancing" and self.rebalancing_target:
            if self.position == self.rebalancing_target:
                # Reached rebalancing target, become idle
                self.status = "idle"
                self.rebalancing_target = None
                self.path = []
            elif self.path:
                # Continue moving to rebalancing target
                next_pos = self.path.pop(0)
                if self._can_move_to(next_pos):
                    self.model.grid.move_agent(self, next_pos)
                    self.position = next_pos
                    self.total_distance += 1
            else:
                # Recalculate path to rebalancing target
                self.path = self._find_path(self.position, self.rebalancing_target)
                if not self.path:
                    # Can't reach target, cancel rebalancing
                    self.status = "idle"
                    self.rebalancing_target = None
        elif len(self.passengers) == 0:
            # Look for nearby passengers
            self._find_passenger()
        else:
            # Transport passengers to destinations
            self._transport_passengers()
    
    def _find_passenger(self):
        """Find and pick up passengers (supports ride sharing and priority)."""
        from agents.passenger import Passenger
        
        # First, identify which passengers are already assigned to other taxis
        assigned_passenger_positions = set()
        for other_agent in self.model.schedule.agents:
            if isinstance(other_agent, Taxi) and other_agent != self:
                # Check if this taxi is heading to a passenger
                if other_agent.status == "picking_up" and hasattr(other_agent, 'path') and other_agent.path:
                    # Get the destination of the path (where passenger is)
                    pickup_dest = other_agent.path[-1] if other_agent.path else None
                    if pickup_dest:
                        assigned_passenger_positions.add(pickup_dest)
        
        # Find passengers, prioritizing by:
        # 1. Priority level (emergency > VIP > regular)
        # 2. Distance
        available_passengers = []
        
        for agent in self.model.schedule.agents:
            if isinstance(agent, Passenger) and agent.status == "waiting":
                # Check if this passenger is already assigned to another taxi
                if agent.position in assigned_passenger_positions:
                    continue  # Skip this passenger, already assigned
                
                if len(self.passengers) < self.capacity:  # Check capacity
                    distance = self._manhattan_distance(self.position, agent.position)
                    priority = getattr(agent, 'priority', 'regular')
                    priority_score = {'emergency': 3, 'vip': 2, 'regular': 1}.get(priority, 1)
                    available_passengers.append((agent, distance, priority_score))
        
        if available_passengers:
            # Sort by priority (descending) then distance (ascending)
            available_passengers.sort(key=lambda x: (-x[2], x[1]))
            nearest_passenger = available_passengers[0][0]
            
            # Move towards passenger
            self.status = "picking_up"
            pickup_location = nearest_passenger.position
            self.path = self._find_path(self.position, pickup_location)
            
            if self.path:
                next_pos = self.path.pop(0)
                if self._can_move_to(next_pos):
                    self.model.grid.move_agent(self, next_pos)
                    self.position = next_pos
                    self.total_distance += 1
                    
                    # Check if reached passenger
                    if self.position == pickup_location:
                        # Pick up passenger
                        self.passengers.append(nearest_passenger)
                        nearest_passenger.status = "in_taxi"
                        self.destinations.append(nearest_passenger.destination)
                        self.status = "transporting"
                        # Plan route to all destinations
                        self._plan_route()
        else:
            # No passengers, move randomly to explore
            self._random_move()
    
    def _plan_route(self):
        """Plan optimal route for multiple destinations (simple nearest-neighbor)."""
        if not self.destinations:
            return
        
        # Simple approach: go to nearest destination first
        current = self.position
        remaining = self.destinations.copy()
        route = []
        
        while remaining:
            nearest = min(remaining, key=lambda d: self._manhattan_distance(current, d))
            route.extend(self._find_path(current, nearest))
            remaining.remove(nearest)
            current = nearest
        
        self.path = route
    
    def _transport_passengers(self):
        """Transport passengers to their destinations (supports multiple passengers)."""
        if not self.path and self.destinations:
            self._plan_route()
        
        if self.path:
            next_pos = self.path.pop(0)
            if self._can_move_to(next_pos):
                self.model.grid.move_agent(self, next_pos)
                self.position = next_pos
                self.total_distance += 1
                
                # Move all passengers with taxi
                for passenger in self.passengers:
                    self.model.grid.move_agent(passenger, next_pos)
                    passenger.position = next_pos
                
                # Check if reached any destination
                passengers_to_drop = []
                for i, destination in enumerate(self.destinations):
                    if self.position == destination:
                        passenger = self.passengers[i]
                        passengers_to_drop.append((i, passenger))
                
                # Drop off passengers (in reverse order to maintain indices)
                for i, passenger in reversed(passengers_to_drop):
                    passenger.status = "arrived"
                    self.model.grid.remove_agent(passenger)
                    self.model.schedule.remove(passenger)
                    self.passengers.pop(i)
                    self.destinations.pop(i)
                    self.total_passengers += 1
                    # Calculate revenue
                    base_fare = 10.0
                    priority_multiplier = {'emergency': 2.0, 'vip': 1.5, 'regular': 1.0}.get(
                        getattr(passenger, 'priority', 'regular'), 1.0
                    )
                    self.revenue += base_fare * priority_multiplier
                
                # If all passengers dropped off, become idle
                if len(self.passengers) == 0:
                    self.destinations = []
                    self.status = "idle"
                    self.path = []
                elif self.path == [] and self.destinations:
                    # Need to replan route for remaining passengers
                    self._plan_route()
    
    def _find_path(self, start, end):
        """Find shortest path using NetworkX."""
        try:
            if start in self.model.road_network and end in self.model.road_network:
                path = nx.shortest_path(self.model.road_network, start, end)
                return path[1:] if len(path) > 1 else []  # Exclude current position
        except nx.NetworkXNoPath:
            pass
        return []
    
    def _can_move_to(self, pos):
        """Check if taxi can move to position (not blocked by traffic light)."""
        from agents.traffic_light import TrafficLight
        
        # Check traffic light at destination position
        cell_contents = self.model.grid.get_cell_list_contents([pos])
        for agent in cell_contents:
            if isinstance(agent, TrafficLight):
                # Determine movement direction
                dx = pos[0] - self.position[0]
                dy = pos[1] - self.position[1]
                
                # Check if movement is horizontal or vertical
                if abs(dx) > abs(dy):
                    # Horizontal movement
                    return agent.is_green_for_direction("horizontal")
                elif abs(dy) > abs(dx):
                    # Vertical movement
                    return agent.is_green_for_direction("vertical")
                else:
                    # Diagonal or same position - allow if green in any direction
                    return agent.is_green()
        return True
    
    def _manhattan_distance(self, pos1, pos2):
        """Calculate Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def _random_move(self):
        """Move randomly when idle."""
        neighbors = self.model.grid.get_neighborhood(
            self.position, moore=False, include_center=False
        )
        # Filter to only road positions
        road_neighbors = [n for n in neighbors if n in self.model.road_network]
        if road_neighbors:
            next_pos = random.choice(road_neighbors)
            if self._can_move_to(next_pos):
                self.model.grid.move_agent(self, next_pos)
                self.position = next_pos

