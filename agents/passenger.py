"""
Passenger agent that spawns and waits for taxis.
"""

import random
from mesa import Agent


class Passenger(Agent):
    """
    Passenger agent with priority levels and enhanced features.
    """
    
    def __init__(self, unique_id, model, position, destination, priority="regular"):
        super().__init__(unique_id, model)
        self.position = position
        self.destination = destination
        self.priority = priority  # "regular", "vip", "emergency"
        self.status = "waiting"  # waiting, in_taxi, arrived
        self.wait_time = 0
        self.spawn_time = model.schedule.time
        self.max_wait_time = self._get_max_wait_time()
        
    def _get_max_wait_time(self):
        """Get maximum wait time based on priority."""
        max_waits = {
            "regular": 500,
            "vip": 300,
            "emergency": 100
        }
        return max_waits.get(self.priority, 500)
        
    def step(self):
        """Execute one step of passenger behavior."""
        if self.status == "waiting":
            self.wait_time += 1
            # Remove if waited too long (based on priority)
            if self.wait_time > self.max_wait_time:
                self.model.grid.remove_agent(self)
                self.model.schedule.remove(self)

