"""
Main city simulation model with agents and road network.
"""

import random
from mesa import Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
import networkx as nx

from agents.taxi import Taxi
from agents.passenger import Passenger
from agents.traffic_light import TrafficLight
from utils.pathfinding import create_road_network, get_random_road_position
from analytics.data_collector import SimulationDataCollector
from analytics.demand_predictor import DemandPredictor
from analytics.fleet_rebalancer import FleetRebalancer
from analytics.ride_matcher import AdvancedRideMatcher
from rl.rl_router import RLRouter, RLTaxiAgent
from dispatch.central_dispatch import CentralDispatchSystem
from optimization.multi_objective_optimizer import MultiObjectiveOptimizer
from database.db_manager import DatabaseManager


class CityModel(Model):
    """
    Main model for the Smart Autonomous Taxi & Traffic System.
    """
    
    def __init__(self, width=30, height=30, num_taxis=10, num_traffic_lights=8, 
                 passenger_spawn_rate=0.1, enable_rush_hour=True, enable_weather=False,
                 enable_database=True, enable_prediction=True, enable_rebalancing=True,
                 enable_rl_routing=False, enable_central_dispatch=False, 
                 enable_multi_objective=False):
        super().__init__()
        
        # Model parameters
        self.width = width
        self.height = height
        self.num_taxis = num_taxis
        self.num_traffic_lights = num_traffic_lights
        self.passenger_spawn_rate = passenger_spawn_rate
        self.enable_rush_hour = enable_rush_hour
        self.enable_weather = enable_weather
        self.enable_database = enable_database
        self.enable_prediction = enable_prediction
        self.enable_rebalancing = enable_rebalancing
        self.enable_rl_routing = enable_rl_routing
        self.enable_central_dispatch = enable_central_dispatch
        self.enable_multi_objective = enable_multi_objective
        
        # Rush hour settings
        self.rush_hour_active = False
        self.rush_hour_multiplier = 2.0  # 2x spawn rate during rush hour
        
        # Weather settings
        self.weather = "clear"  # "clear", "rain", "snow"
        self.weather_speed_modifier = 1.0
        
        # Create grid and road network
        self.grid = MultiGrid(width, height, True)
        self.road_network = create_road_network(width, height)
        
        # Schedule for agent activation
        self.schedule = RandomActivation(self)
        
        # Data collector for custom analytics
        self.custom_datacollector = SimulationDataCollector()
        
        # Initialize advanced features
        if self.enable_database:
            self.db_manager = DatabaseManager()
            self.run_id = self.db_manager.start_simulation_run(
                width=width, height=height, num_taxis=num_taxis,
                num_traffic_lights=num_traffic_lights,
                passenger_spawn_rate=passenger_spawn_rate
            )
        else:
            self.db_manager = None
            self.run_id = None
        
        if self.enable_prediction:
            self.demand_predictor = DemandPredictor()
            # Load historical data if available
            if self.db_manager:
                try:
                    historical_data = self.db_manager.get_historical_demand()
                    if not historical_data.empty:
                        self.demand_predictor.update_with_historical_data(historical_data)
                except Exception as e:
                    print(f"Warning: Could not load historical data: {e}")
        else:
            self.demand_predictor = None
        
        if self.enable_rebalancing:
            self.fleet_rebalancer = FleetRebalancer()
        else:
            self.fleet_rebalancer = None
        
        # Advanced ride matcher
        self.ride_matcher = AdvancedRideMatcher()
        
        # Phase 2: RL Routing
        if self.enable_rl_routing:
            self.rl_router = RLRouter()
        else:
            self.rl_router = None
        
        # Phase 2: Central Dispatch System
        if self.enable_central_dispatch:
            self.central_dispatch = CentralDispatchSystem()
        else:
            self.central_dispatch = None
        
        # Phase 2: Multi-Objective Optimizer
        if self.enable_multi_objective:
            self.route_optimizer = MultiObjectiveOptimizer()
        else:
            self.route_optimizer = None
        
        # Rebalancing frequency (every N steps)
        self.rebalance_frequency = 5
        
        # Mesa DataCollector for charts (must be named 'datacollector' for Mesa charts)
        self.datacollector = DataCollector(
            model_reporters={
                "Waiting_Passengers": lambda m: len([a for a in m.schedule.agents 
                                                     if isinstance(a, Passenger) and a.status == "waiting"]),
                "Active_Taxis": lambda m: len([a for a in m.schedule.agents 
                                               if isinstance(a, Taxi) and a.status != "idle"]),
                "Avg_Wait_Time": lambda m: self._calculate_avg_wait_time(),
                "Taxi_Utilization": lambda m: self._calculate_taxi_utilization(),
                "Traffic_Density": lambda m: self._calculate_traffic_density(),
            },
            agent_reporters={
                "Status": lambda a: a.status if hasattr(a, 'status') else None,
            }
        )
        
        # Initialize agents
        self._create_taxis()
        self._create_traffic_lights()
        
        # Track intersections for traffic lights
        self.intersections = self._identify_intersections()
        
        # Running statistics
        self.running = True
        
    def _create_taxis(self):
        """Create taxi agents at random road positions with different types."""
        taxi_types = ["economy", "premium", "luxury"]
        for i in range(self.num_taxis):
            pos = get_random_road_position(self.road_network)
            if pos:
                # Distribute taxi types: 60% economy, 30% premium, 10% luxury
                if i < int(self.num_taxis * 0.6):
                    taxi_type = "economy"
                elif i < int(self.num_taxis * 0.9):
                    taxi_type = "premium"
                else:
                    taxi_type = "luxury"
                
                taxi = Taxi(i, self, pos, taxi_type=taxi_type)
                self.schedule.add(taxi)
                self.grid.place_agent(taxi, pos)
    
    def _create_traffic_lights(self):
        """Create traffic light agents at intersections."""
        intersections = self._identify_intersections()
        selected_intersections = random.sample(
            intersections, min(self.num_traffic_lights, len(intersections))
        )
        
        for i, intersection in enumerate(selected_intersections):
            # Alternate between horizontal and vertical control
            direction = "horizontal" if i % 2 == 0 else "vertical"
            traffic_light = TrafficLight(f"tl_{i}", self, intersection, direction)
            self.schedule.add(traffic_light)
            self.grid.place_agent(traffic_light, intersection)
    
    def _identify_intersections(self):
        """Identify intersection points in the road network."""
        intersections = []
        for node in self.road_network.nodes():
            # Count neighbors (degree) - intersections have degree > 2
            degree = self.road_network.degree(node)
            if degree >= 3:  # Intersection point
                intersections.append(node)
        return intersections if intersections else list(self.road_network.nodes())[:self.num_traffic_lights]
    
    def step(self):
        """Execute one step of the simulation."""
        # Update rush hour status
        if self.enable_rush_hour:
            self._update_rush_hour()
        
        # Update weather (random changes)
        if self.enable_weather:
            self._update_weather()
        
        # Calculate effective spawn rate (rush hour multiplier)
        effective_spawn_rate = self.passenger_spawn_rate
        if self.rush_hour_active:
            effective_spawn_rate *= self.rush_hour_multiplier
        
        # Spawn new passengers
        if random.random() < effective_spawn_rate:
            self._spawn_passenger()
        
        # Phase 2: Central Dispatch System (global optimization)
        if self.enable_central_dispatch and self.central_dispatch:
            waiting_passengers = [a for a in self.schedule.agents 
                                if isinstance(a, Passenger) and a.status == "waiting"]
            available_taxis = [a for a in self.schedule.agents 
                             if isinstance(a, Taxi) and len(a.passengers) < a.capacity]
            
            if waiting_passengers and available_taxis:
                # Find optimal global assignments
                assignments = self.central_dispatch.optimal_assignments(available_taxis, waiting_passengers)
                # Execute assignments
                if assignments:
                    self.central_dispatch.execute_assignments(self, assignments)
            
            # Clear completed assignments
            self.central_dispatch.clear_completed_assignments(self)
        
        # Fleet rebalancing (every N steps)
        if self.enable_rebalancing and self.fleet_rebalancer:
            if self.schedule.time % self.rebalance_frequency == 0:
                self.fleet_rebalancer.update_idle_times(self)
                if self.fleet_rebalancer.should_rebalance(self):
                    # Get demand predictions if available
                    demand_predictions = None
                    if self.enable_prediction and self.demand_predictor:
                        grid_predictions = self.demand_predictor.predict_demand(
                            self.schedule.time, self.width, self.height
                        )
                        # Convert grid predictions to position-based predictions
                        demand_predictions = {}
                        for (grid_x, grid_y), demand in grid_predictions.items():
                            center_x = grid_x * 5 + 2
                            center_y = grid_y * 5 + 2
                            if 0 <= center_x < self.width and 0 <= center_y < self.height:
                                demand_predictions[(center_x, center_y)] = demand
                    
                    actions = self.fleet_rebalancer.find_rebalancing_targets(self, demand_predictions)
                    if actions:
                        # Save rebalancing actions to database before executing
                        if self.enable_database and self.db_manager and self.run_id:
                            for action in actions:
                                # Get taxi's current position
                                for agent in self.schedule.agents:
                                    if isinstance(agent, Taxi) and agent.unique_id == action['taxi_id']:
                                        action_data = {
                                            'taxi_id': action['taxi_id'],
                                            'from_x': agent.position[0],
                                            'from_y': agent.position[1],
                                            'to_x': action['target'][0],
                                            'to_y': action['target'][1],
                                            'reason': action.get('reason', 'High demand area')
                                        }
                                        self.db_manager.save_rebalancing_action(
                                            self.run_id, 
                                            self.schedule.time, 
                                            action_data
                                        )
                                        break
                        
                        # Execute rebalancing
                        self.fleet_rebalancer.execute_rebalancing(self, actions)
        
        # Advance all agents
        self.schedule.step()
        
        # Update demand predictor with real-time data
        if self.enable_prediction and self.demand_predictor:
            waiting_passengers = [a for a in self.schedule.agents 
                                if isinstance(a, Passenger) and a.status == "waiting"]
            passenger_data = [{'x': p.position[0], 'y': p.position[1], 'priority': p.priority} 
                            for p in waiting_passengers]
            self.demand_predictor.update_realtime(passenger_data)
        
        # Collect data
        self.custom_datacollector.collect_step_data(self)
        self.datacollector.collect(self)
        
        # Save to database
        if self.enable_database and self.db_manager and self.run_id:
            self._save_step_to_database()
        
        # Stop after reasonable number of steps (optional)
        if self.schedule.time >= 1000:
            self.running = False
            if self.enable_database and self.db_manager and self.run_id:
                self.db_manager.end_simulation_run(self.run_id, self.schedule.time)
    
    def _save_step_to_database(self):
        """Save current step data to database."""
        if not self.db_manager or not self.run_id:
            return
        
        step = self.schedule.time
        
        # Collect metrics
        metrics = {
            'waiting_passengers': len([a for a in self.schedule.agents 
                                      if isinstance(a, Passenger) and a.status == "waiting"]),
            'active_taxis': len([a for a in self.schedule.agents 
                               if isinstance(a, Taxi) and a.status != "idle"]),
            'avg_wait_time': self._calculate_avg_wait_time(),
            'taxi_utilization': self._calculate_taxi_utilization(),
            'traffic_density': self._calculate_traffic_density(),
            'passengers_served': sum(t.total_passengers for t in self.schedule.agents 
                                   if isinstance(t, Taxi)),
            'total_revenue': sum(t.revenue for t in self.schedule.agents 
                               if isinstance(t, Taxi))
        }
        
        self.db_manager.save_step_metrics(self.run_id, step, metrics)
        
        # Save passenger demand data
        waiting_passengers = [a for a in self.schedule.agents 
                            if isinstance(a, Passenger) and a.status == "waiting"]
        passenger_data = [{
            'x': p.position[0],
            'y': p.position[1],
            'priority': p.priority,
            'wait_time': p.wait_time
        } for p in waiting_passengers]
        if passenger_data:
            self.db_manager.save_passenger_demand(self.run_id, step, passenger_data)
        
        # Save taxi positions
        taxis = [a for a in self.schedule.agents if isinstance(a, Taxi)]
        taxi_data = [{
            'id': t.unique_id,
            'x': t.position[0],
            'y': t.position[1],
            'status': t.status,
            'type': t.taxi_type,
            'passengers_count': len(t.passengers)
        } for t in taxis]
        if taxi_data:
            self.db_manager.save_taxi_positions(self.run_id, step, taxi_data)
        
        # Save demand predictions if available
        if self.enable_prediction and self.demand_predictor:
            grid_predictions = self.demand_predictor.predict_demand(
                step, self.width, self.height
            )
            prediction_data = []
            for (grid_x, grid_y), demand in grid_predictions.items():
                center_x = grid_x * 5 + 2
                center_y = grid_y * 5 + 2
                if 0 <= center_x < self.width and 0 <= center_y < self.height:
                    prediction_data.append({
                        'x': center_x,
                        'y': center_y,
                        'demand': demand,
                        'confidence': min(demand / 5.0, 1.0)  # Simple confidence metric
                    })
            if prediction_data:
                self.db_manager.save_demand_prediction(self.run_id, step, prediction_data)
    
    def _update_rush_hour(self):
        """Update rush hour status based on simulation time."""
        step = self.schedule.time
        # Rush hours: steps 50-150 (morning) and 300-400 (evening)
        if (50 <= step <= 150) or (300 <= step <= 400):
            self.rush_hour_active = True
        else:
            self.rush_hour_active = False
    
    def _update_weather(self):
        """Randomly change weather conditions."""
        if random.random() < 0.01:  # 1% chance per step
            weathers = ["clear", "rain", "snow"]
            self.weather = random.choice(weathers)
            # Weather affects movement speed
            if self.weather == "rain":
                self.weather_speed_modifier = 0.8  # 20% slower
            elif self.weather == "snow":
                self.weather_speed_modifier = 0.6  # 40% slower
            else:
                self.weather_speed_modifier = 1.0
    
    def _calculate_avg_wait_time(self):
        """Calculate average wait time for waiting passengers."""
        waiting_passengers = [a for a in self.schedule.agents 
                             if isinstance(a, Passenger) and a.status == "waiting"]
        if waiting_passengers:
            return sum(p.wait_time for p in waiting_passengers) / len(waiting_passengers)
        return 0
    
    def _calculate_taxi_utilization(self):
        """Calculate taxi utilization percentage."""
        taxis = [a for a in self.schedule.agents if isinstance(a, Taxi)]
        if taxis:
            active = sum(1 for t in taxis if t.status != "idle")
            return (active / len(taxis)) * 100
        return 0
    
    def _calculate_traffic_density(self):
        """Calculate traffic density (vehicles per cell)."""
        from agents.taxi import Taxi
        occupied_cells = 0
        for x in range(self.grid.width):
            for y in range(self.grid.height):
                cell_contents = self.grid.get_cell_list_contents([(x, y)])
                if any(isinstance(a, Taxi) for a in cell_contents):
                    occupied_cells += 1
        total_cells = self.width * self.height
        return (occupied_cells / total_cells) * 100 if total_cells > 0 else 0
    
    def _spawn_passenger(self):
        """Spawn a new passenger with random priority."""
        start_pos = get_random_road_position(self.road_network)
        end_pos = get_random_road_position(self.road_network)
        
        # Ensure start and end are different
        while end_pos == start_pos:
            end_pos = get_random_road_position(self.road_network)
        
        if start_pos and end_pos:
            # Assign priority: 5% emergency, 15% VIP, 80% regular
            rand = random.random()
            if rand < 0.05:
                priority = "emergency"
            elif rand < 0.20:
                priority = "vip"
            else:
                priority = "regular"
            
            passenger_id = f"passenger_{self.schedule.time}_{random.randint(1000, 9999)}"
            passenger = Passenger(passenger_id, self, start_pos, end_pos, priority=priority)
            self.schedule.add(passenger)
            self.grid.place_agent(passenger, start_pos)

