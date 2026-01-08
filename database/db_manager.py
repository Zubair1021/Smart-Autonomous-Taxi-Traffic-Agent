"""
Database manager for storing and retrieving historical simulation data.
Uses SQLite for simplicity, but can be easily switched to PostgreSQL.
"""

import sqlite3
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
import pandas as pd


class DatabaseManager:
    """
    Manages database operations for storing historical simulation data.
    """
    
    def __init__(self, db_path: str = "simulation_data.db"):
        """
        Initialize database manager.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.conn = None
        self._initialize_database()
    
    def _initialize_database(self):
        """Create database tables if they don't exist."""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row  # Enable column access by name
        
        cursor = self.conn.cursor()
        
        # Simulation runs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS simulation_runs (
                run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                end_time TIMESTAMP,
                width INTEGER,
                height INTEGER,
                num_taxis INTEGER,
                num_traffic_lights INTEGER,
                passenger_spawn_rate REAL,
                total_steps INTEGER,
                notes TEXT
            )
        """)
        
        # Step-level metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS step_metrics (
                metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER,
                step INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                waiting_passengers INTEGER,
                active_taxis INTEGER,
                avg_wait_time REAL,
                taxi_utilization REAL,
                traffic_density REAL,
                passengers_served INTEGER,
                total_revenue REAL,
                FOREIGN KEY (run_id) REFERENCES simulation_runs(run_id)
            )
        """)
        
        # Passenger demand data (for prediction)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS passenger_demand (
                demand_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER,
                step INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                x_position INTEGER,
                y_position INTEGER,
                priority TEXT,
                wait_time INTEGER,
                FOREIGN KEY (run_id) REFERENCES simulation_runs(run_id)
            )
        """)
        
        # Taxi positions and status (for fleet rebalancing)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS taxi_positions (
                position_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER,
                step INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                taxi_id INTEGER,
                x_position INTEGER,
                y_position INTEGER,
                status TEXT,
                taxi_type TEXT,
                passengers_count INTEGER,
                FOREIGN KEY (run_id) REFERENCES simulation_runs(run_id)
            )
        """)
        
        # Demand predictions
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS demand_predictions (
                prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER,
                step INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                x_position INTEGER,
                y_position INTEGER,
                predicted_demand REAL,
                confidence REAL,
                FOREIGN KEY (run_id) REFERENCES simulation_runs(run_id)
            )
        """)
        
        # Fleet rebalancing actions
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS rebalancing_actions (
                action_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER,
                step INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                taxi_id INTEGER,
                from_x INTEGER,
                from_y INTEGER,
                to_x INTEGER,
                to_y INTEGER,
                reason TEXT,
                FOREIGN KEY (run_id) REFERENCES simulation_runs(run_id)
            )
        """)
        
        self.conn.commit()
    
    def start_simulation_run(self, **params) -> int:
        """
        Start a new simulation run and return run_id.
        
        Args:
            **params: Simulation parameters (width, height, num_taxis, etc.)
        
        Returns:
            run_id: Unique identifier for this simulation run
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO simulation_runs 
            (width, height, num_taxis, num_traffic_lights, passenger_spawn_rate, notes)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            params.get('width'),
            params.get('height'),
            params.get('num_taxis'),
            params.get('num_traffic_lights'),
            params.get('passenger_spawn_rate'),
            params.get('notes', '')
        ))
        self.conn.commit()
        return cursor.lastrowid
    
    def end_simulation_run(self, run_id: int, total_steps: int):
        """Mark simulation run as ended."""
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE simulation_runs 
            SET end_time = CURRENT_TIMESTAMP, total_steps = ?
            WHERE run_id = ?
        """, (total_steps, run_id))
        self.conn.commit()
    
    def save_step_metrics(self, run_id: int, step: int, metrics: Dict[str, Any]):
        """
        Save step-level metrics.
        
        Args:
            run_id: Simulation run ID
            step: Current simulation step
            metrics: Dictionary with metric values
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO step_metrics 
            (run_id, step, waiting_passengers, active_taxis, avg_wait_time, 
             taxi_utilization, traffic_density, passengers_served, total_revenue)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run_id,
            step,
            metrics.get('waiting_passengers', 0),
            metrics.get('active_taxis', 0),
            metrics.get('avg_wait_time', 0.0),
            metrics.get('taxi_utilization', 0.0),
            metrics.get('traffic_density', 0.0),
            metrics.get('passengers_served', 0),
            metrics.get('total_revenue', 0.0)
        ))
        self.conn.commit()
    
    def save_passenger_demand(self, run_id: int, step: int, passengers: List[Dict]):
        """
        Save passenger demand data for prediction.
        
        Args:
            run_id: Simulation run ID
            step: Current simulation step
            passengers: List of passenger dictionaries with position, priority, wait_time
        """
        cursor = self.conn.cursor()
        for passenger in passengers:
            cursor.execute("""
                INSERT INTO passenger_demand 
                (run_id, step, x_position, y_position, priority, wait_time)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                run_id,
                step,
                passenger.get('x', 0),
                passenger.get('y', 0),
                passenger.get('priority', 'regular'),
                passenger.get('wait_time', 0)
            ))
        self.conn.commit()
    
    def save_taxi_positions(self, run_id: int, step: int, taxis: List[Dict]):
        """
        Save taxi positions for fleet rebalancing.
        
        Args:
            run_id: Simulation run ID
            step: Current simulation step
            taxis: List of taxi dictionaries with position, status, type
        """
        cursor = self.conn.cursor()
        for taxi in taxis:
            cursor.execute("""
                INSERT INTO taxi_positions 
                (run_id, step, taxi_id, x_position, y_position, status, taxi_type, passengers_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run_id,
                step,
                taxi.get('id', 0),
                taxi.get('x', 0),
                taxi.get('y', 0),
                taxi.get('status', 'idle'),
                taxi.get('type', 'economy'),
                taxi.get('passengers_count', 0)
            ))
        self.conn.commit()
    
    def save_demand_prediction(self, run_id: int, step: int, predictions: List[Dict]):
        """
        Save demand predictions.
        
        Args:
            run_id: Simulation run ID
            step: Current simulation step
            predictions: List of prediction dictionaries
        """
        cursor = self.conn.cursor()
        for pred in predictions:
            cursor.execute("""
                INSERT INTO demand_predictions 
                (run_id, step, x_position, y_position, predicted_demand, confidence)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                run_id,
                step,
                pred.get('x', 0),
                pred.get('y', 0),
                pred.get('demand', 0.0),
                pred.get('confidence', 0.0)
            ))
        self.conn.commit()
    
    def save_rebalancing_action(self, run_id: int, step: int, action: Dict):
        """
        Save fleet rebalancing action.
        
        Args:
            run_id: Simulation run ID
            step: Current simulation step
            action: Action dictionary with taxi_id, positions, reason
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO rebalancing_actions 
            (run_id, step, taxi_id, from_x, from_y, to_x, to_y, reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run_id,
            step,
            action.get('taxi_id'),
            action.get('from_x'),
            action.get('from_y'),
            action.get('to_x'),
            action.get('to_y'),
            action.get('reason', '')
        ))
        self.conn.commit()
    
    def get_historical_demand(self, window_size: int = 100) -> pd.DataFrame:
        """
        Get historical demand data for prediction.
        
        Args:
            window_size: Number of recent steps to retrieve
        
        Returns:
            DataFrame with historical demand data
        """
        query = """
            SELECT step, x_position, y_position, priority, wait_time, timestamp
            FROM passenger_demand
            ORDER BY timestamp DESC
            LIMIT ?
        """
        return pd.read_sql_query(query, self.conn, params=(window_size,))
    
    def get_historical_metrics(self, run_id: Optional[int] = None, limit: int = 1000) -> pd.DataFrame:
        """
        Get historical metrics.
        
        Args:
            run_id: Optional run ID to filter by
            limit: Maximum number of records to retrieve
        
        Returns:
            DataFrame with historical metrics
        """
        if run_id:
            query = """
                SELECT * FROM step_metrics
                WHERE run_id = ?
                ORDER BY step DESC
                LIMIT ?
            """
            return pd.read_sql_query(query, self.conn, params=(run_id, limit))
        else:
            query = """
                SELECT * FROM step_metrics
                ORDER BY timestamp DESC
                LIMIT ?
            """
            return pd.read_sql_query(query, self.conn, params=(limit,))
    
    def get_latest_run_id(self) -> Optional[int]:
        """Get the most recent simulation run ID."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT MAX(run_id) FROM simulation_runs")
        result = cursor.fetchone()
        return result[0] if result and result[0] else None
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

