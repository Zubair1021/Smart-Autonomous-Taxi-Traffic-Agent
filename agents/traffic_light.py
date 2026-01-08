"""
Adaptive traffic light agent with intelligent timing control.
"""

from mesa import Agent


class TrafficLight(Agent):
    """
    Adaptive traffic light that adjusts timing based on traffic density.
    """
    
    def __init__(self, unique_id, model, position, direction="horizontal"):
        super().__init__(unique_id, model)
        self.position = position
        self.direction = direction  # "horizontal" or "vertical"
        self.state = "green"  # "green" or "red"
        self.time_in_state = 0
        self.min_green_time = 5
        self.max_green_time = 15
        self.min_red_time = 3
        
    def step(self):
        """Execute one step of traffic light behavior."""
        self.time_in_state += 1
        
        # Count vehicles waiting in each direction
        traffic_density = self._count_traffic()
        
        # Adaptive logic: switch if conditions met
        if self.state == "green":
            if self.time_in_state >= self.min_green_time:
                # Switch if low traffic or max time reached
                if traffic_density[self.direction] == 0 or self.time_in_state >= self.max_green_time:
                    if traffic_density[self._opposite_direction()] > 0:
                        self._switch_to_red()
        else:  # red
            if self.time_in_state >= self.min_red_time:
                # Switch if high traffic in this direction or other direction is clear
                other_density = traffic_density[self._opposite_direction()]
                if traffic_density[self.direction] > other_density * 1.5 or other_density == 0:
                    self._switch_to_green()
    
    def _count_traffic(self):
        """Count vehicles in each direction near the intersection."""
        from agents.taxi import Taxi
        
        horizontal_count = 0
        vertical_count = 0
        
        # Check neighboring cells
        neighbors = self.model.grid.get_neighborhood(
            self.position, moore=True, include_center=True, radius=1
        )
        
        for pos in neighbors:
            cell_contents = self.model.grid.get_cell_list_contents([pos])
            for agent in cell_contents:
                if isinstance(agent, Taxi):
                    # Determine direction based on position relative to intersection
                    dx = pos[0] - self.position[0]
                    dy = pos[1] - self.position[1]
                    
                    if abs(dx) > abs(dy):
                        horizontal_count += 1
                    elif abs(dy) > abs(dx):
                        vertical_count += 1
        
        return {
            "horizontal": horizontal_count,
            "vertical": vertical_count
        }
    
    def _opposite_direction(self):
        """Get opposite direction."""
        return "vertical" if self.direction == "horizontal" else "horizontal"
    
    def _switch_to_green(self):
        """Switch traffic light to green."""
        self.state = "green"
        self.time_in_state = 0
    
    def _switch_to_red(self):
        """Switch traffic light to red."""
        self.state = "red"
        self.time_in_state = 0
    
    def is_green(self):
        """Check if traffic light is green (general check)."""
        return self.state == "green"
    
    def is_green_for_direction(self, direction):
        """Check if traffic light is green for a specific direction."""
        if self.direction == direction:
            return self.state == "green"
        else:
            # If controlling opposite direction, check if red (allowing this direction)
            return self.state == "red"
    
    def get_state(self):
        """Get current state of traffic light."""
        return self.state

