"""
Data collection and analytics for simulation metrics.
"""

from collections import defaultdict


class SimulationDataCollector:
    """
    Collects and stores simulation metrics for analysis.
    """
    
    def __init__(self):
        self.wait_times = []
        self.travel_times = []
        self.traffic_density = []
        self.taxi_utilization = []
        self.passenger_data = []
        self.taxi_data = []
        
    def collect_step_data(self, model):
        """Collect data for current simulation step."""
        from agents.taxi import Taxi
        from agents.passenger import Passenger
        
        # Collect passenger wait times
        waiting_passengers = [a for a in model.schedule.agents if isinstance(a, Passenger) and a.status == "waiting"]
        if waiting_passengers:
            avg_wait = sum(p.wait_time for p in waiting_passengers) / len(waiting_passengers)
            self.wait_times.append(avg_wait)
        
        # Collect taxi utilization
        taxis = [a for a in model.schedule.agents if isinstance(a, Taxi)]
        if taxis:
            active_taxis = sum(1 for t in taxis if t.status != "idle")
            utilization = active_taxis / len(taxis) if taxis else 0
            self.taxi_utilization.append(utilization)
        
        # Collect traffic density (vehicles per cell)
        occupied_cells = 0
        for x in range(model.grid.width):
            for y in range(model.grid.height):
                cell_contents = model.grid.get_cell_list_contents([(x, y)])
                if any(isinstance(a, Taxi) for a in cell_contents):
                    occupied_cells += 1
        total_cells = model.grid.width * model.grid.height
        density = occupied_cells / total_cells if total_cells > 0 else 0
        self.traffic_density.append(density)
        
        # Store passenger data
        for passenger in waiting_passengers:
            self.passenger_data.append({
                'step': model.schedule.time,
                'wait_time': passenger.wait_time,
                'status': passenger.status
            })
        
        # Store taxi data
        for taxi in taxis:
            self.taxi_data.append({
                'step': model.schedule.time,
                'status': taxi.status,
                'total_passengers': taxi.total_passengers,
                'total_distance': taxi.total_distance
            })
    
    def get_summary_stats(self):
        """Get summary statistics of collected data."""
        stats = {
            'avg_wait_time': sum(self.wait_times) / len(self.wait_times) if self.wait_times else 0,
            'max_wait_time': max(self.wait_times) if self.wait_times else 0,
            'avg_taxi_utilization': sum(self.taxi_utilization) / len(self.taxi_utilization) if self.taxi_utilization else 0,
            'avg_traffic_density': sum(self.traffic_density) / len(self.traffic_density) if self.traffic_density else 0,
            'total_passengers_served': len([d for d in self.passenger_data if d['status'] == 'arrived']),
        }
        return stats

