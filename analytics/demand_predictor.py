"""
Demand prediction system using time-series forecasting and spatial analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from datetime import datetime


class DemandPredictor:
    """
    Predicts passenger demand at different locations using historical data.
    Uses simple moving average and spatial clustering for prediction.
    """
    
    def __init__(self, window_size: int = 50, spatial_grid_size: int = 5):
        """
        Initialize demand predictor.
        
        Args:
            window_size: Number of historical steps to consider
            spatial_grid_size: Size of spatial grid cells for clustering
        """
        self.window_size = window_size
        self.spatial_grid_size = spatial_grid_size
        self.demand_history = defaultdict(list)  # (grid_x, grid_y) -> [demand values]
        self.prediction_cache = {}
    
    def update_with_historical_data(self, historical_data: pd.DataFrame):
        """
        Update predictor with historical demand data from database.
        
        Args:
            historical_data: DataFrame with columns: step, x_position, y_position, priority
        """
        if historical_data.empty:
            return
        
        # Group by spatial grid
        historical_data['grid_x'] = (historical_data['x_position'] / self.spatial_grid_size).astype(int)
        historical_data['grid_y'] = (historical_data['y_position'] / self.spatial_grid_size).astype(int)
        
        # Aggregate demand by grid cell and step
        grouped = historical_data.groupby(['step', 'grid_x', 'grid_y']).size().reset_index(name='demand')
        
        for _, row in grouped.iterrows():
            grid_key = (row['grid_x'], row['grid_y'])
            self.demand_history[grid_key].append({
                'step': row['step'],
                'demand': row['demand']
            })
        
        # Keep only recent history
        for grid_key in self.demand_history:
            self.demand_history[grid_key] = self.demand_history[grid_key][-self.window_size:]
    
    def predict_demand(self, current_step: int, city_width: int, city_height: int) -> Dict[Tuple[int, int], float]:
        """
        Predict demand for all grid cells.
        
        Args:
            current_step: Current simulation step
            city_width: Width of city grid
            city_height: Height of city grid
        
        Returns:
            Dictionary mapping (grid_x, grid_y) to predicted demand value
        """
        predictions = {}
        
        # Calculate grid dimensions
        grid_width = (city_width + self.spatial_grid_size - 1) // self.spatial_grid_size
        grid_height = (city_height + self.spatial_grid_size - 1) // self.spatial_grid_size
        
        for grid_x in range(grid_width):
            for grid_y in range(grid_height):
                grid_key = (grid_x, grid_y)
                
                if grid_key in self.demand_history and len(self.demand_history[grid_key]) > 0:
                    # Use moving average for prediction
                    recent_demands = [d['demand'] for d in self.demand_history[grid_key][-10:]]
                    predicted_demand = np.mean(recent_demands) if recent_demands else 0.0
                    
                    # Add trend component (simple linear trend)
                    if len(recent_demands) >= 3:
                        trend = (recent_demands[-1] - recent_demands[0]) / len(recent_demands)
                        predicted_demand += trend * 2  # Project forward
                    
                    predictions[grid_key] = max(0.0, predicted_demand)
                else:
                    # No history, use default low prediction
                    predictions[grid_key] = 0.1
        
        return predictions
    
    def predict_for_position(self, x: int, y: int, current_step: int, city_width: int, city_height: int) -> float:
        """
        Predict demand for a specific position.
        
        Args:
            x: X coordinate
            y: Y coordinate
            current_step: Current simulation step
            city_width: Width of city grid
            city_height: Height of city grid
        
        Returns:
            Predicted demand value
        """
        grid_x = int(x / self.spatial_grid_size)
        grid_y = int(y / self.spatial_grid_size)
        
        predictions = self.predict_demand(current_step, city_width, city_height)
        return predictions.get((grid_x, grid_y), 0.1)
    
    def update_realtime(self, passengers: List[Dict]):
        """
        Update predictor with real-time passenger data.
        
        Args:
            passengers: List of passenger dictionaries with 'x', 'y', 'priority'
        """
        # Group passengers by spatial grid
        grid_demand = defaultdict(int)
        
        for passenger in passengers:
            grid_x = int(passenger.get('x', 0) / self.spatial_grid_size)
            grid_y = int(passenger.get('y', 0) / self.spatial_grid_size)
            grid_demand[(grid_x, grid_y)] += 1
        
        # Update history
        current_time = datetime.now().timestamp()
        for grid_key, demand in grid_demand.items():
            self.demand_history[grid_key].append({
                'step': current_time,
                'demand': demand
            })
            # Keep only recent history
            if len(self.demand_history[grid_key]) > self.window_size:
                self.demand_history[grid_key].pop(0)
    
    def get_demand_heatmap_data(self, current_step: int, city_width: int, city_height: int) -> Tuple[List[int], List[int], List[float]]:
        """
        Get data for heatmap visualization.
        
        Returns:
            Tuple of (x_positions, y_positions, demand_values)
        """
        predictions = self.predict_demand(current_step, city_width, city_height)
        
        x_positions = []
        y_positions = []
        demand_values = []
        
        for (grid_x, grid_y), demand in predictions.items():
            # Convert grid coordinates back to approximate city coordinates
            center_x = grid_x * self.spatial_grid_size + self.spatial_grid_size // 2
            center_y = grid_y * self.spatial_grid_size + self.spatial_grid_size // 2
            
            x_positions.append(center_x)
            y_positions.append(center_y)
            demand_values.append(demand)
        
        return x_positions, y_positions, demand_values

