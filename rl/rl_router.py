"""
Reinforcement Learning Router using Q-Learning for optimal routing decisions.
Simplified implementation suitable for real-time simulation.
"""

import numpy as np
import random
from typing import Dict, Tuple, Optional, List
from collections import defaultdict
import json
import os


class RLRouter:
    """
    Q-Learning based router that learns optimal routing decisions.
    Uses state-action-reward framework to learn from experience.
    """
    
    def __init__(self, learning_rate: float = 0.1, discount_factor: float = 0.95, 
                 epsilon: float = 0.2, q_table_path: str = "rl/q_table.json"):
        """
        Initialize RL Router.
        
        Args:
            learning_rate: How fast the agent learns (0-1)
            discount_factor: Importance of future rewards (0-1)
            epsilon: Exploration rate (0-1, higher = more exploration)
            q_table_path: Path to save/load Q-table
        """
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_table_path = q_table_path
        
        # Q-table: state -> {action: Q-value}
        # State: (taxi_pos_x, taxi_pos_y, passenger_pos_x, passenger_pos_y, passenger_priority)
        # Action: direction to move (0=up, 1=right, 2=down, 3=left, 4=stay)
        self.q_table = defaultdict(lambda: defaultdict(float))
        
        # Load existing Q-table if available
        self._load_q_table()
        
        # Track recent experiences for learning
        self.last_state = None
        self.last_action = None
        self.total_rewards = 0.0
        self.step_count = 0
    
    def _load_q_table(self):
        """Load Q-table from file if it exists."""
        if os.path.exists(self.q_table_path):
            try:
                with open(self.q_table_path, 'r') as f:
                    data = json.load(f)
                    # Convert string keys back to tuples
                    self.q_table = defaultdict(lambda: defaultdict(float))
                    for state_str, actions in data.items():
                        state = eval(state_str)  # Convert string to tuple
                        self.q_table[state] = defaultdict(float, actions)
                print(f"✅ Loaded Q-table with {len(self.q_table)} states")
            except Exception as e:
                print(f"⚠️ Could not load Q-table: {e}")
    
    def _save_q_table(self):
        """Save Q-table to file."""
        try:
            os.makedirs(os.path.dirname(self.q_table_path), exist_ok=True)
            # Convert tuples to strings for JSON
            data = {str(k): dict(v) for k, v in self.q_table.items()}
            with open(self.q_table_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"⚠️ Could not save Q-table: {e}")
    
    def get_state_key(self, taxi_pos: Tuple[int, int], passenger_pos: Tuple[int, int], 
                     passenger_priority: str) -> Tuple:
        """
        Convert current situation to state key.
        
        Args:
            taxi_pos: (x, y) position of taxi
            passenger_pos: (x, y) position of passenger
            passenger_priority: Priority level
        
        Returns:
            State tuple for Q-table
        """
        # Simplify state space by using relative position and grid discretization
        dx = passenger_pos[0] - taxi_pos[0]
        dy = passenger_pos[1] - taxi_pos[1]
        
        # Discretize to reduce state space (bin positions)
        dx_bin = dx // 5  # Group positions in 5-unit bins
        dy_bin = dy // 5
        
        priority_num = {'emergency': 3, 'vip': 2, 'regular': 1}.get(passenger_priority, 1)
        
        return (dx_bin, dy_bin, priority_num)
    
    def choose_action(self, state: Tuple) -> int:
        """
        Choose action using epsilon-greedy policy.
        
        Args:
            state: Current state tuple
        
        Returns:
            Action: 0=up, 1=right, 2=down, 3=left, 4=stay
        """
        # Exploration: random action
        if random.random() < self.epsilon:
            return random.randint(0, 4)
        
        # Exploitation: best known action
        actions = self.q_table[state]
        if not actions:
            # No experience, choose random
            return random.randint(0, 4)
        
        # Get action with highest Q-value
        best_action = max(actions.items(), key=lambda x: x[1])[0]
        return best_action
    
    def calculate_reward(self, taxi_pos: Tuple[int, int], passenger_pos: Tuple[int, int],
                        passenger_priority: str, distance_reduced: float, 
                        reached_passenger: bool) -> float:
        """
        Calculate reward for an action.
        
        Args:
            taxi_pos: Current taxi position
            passenger_pos: Target passenger position
            passenger_priority: Passenger priority
            distance_reduced: How much closer we got (can be negative)
            reached_passenger: Whether we reached the passenger
        
        Returns:
            Reward value
        """
        reward = 0.0
        
        # Reward for getting closer
        if distance_reduced > 0:
            reward += 10.0 * distance_reduced
        
        # Penalty for getting farther
        if distance_reduced < 0:
            reward += 5.0 * distance_reduced  # Negative reward
        
        # Large reward for reaching passenger
        if reached_passenger:
            priority_multiplier = {'emergency': 3.0, 'vip': 2.0, 'regular': 1.0}.get(
                passenger_priority, 1.0
            )
            reward += 100.0 * priority_multiplier
        
        # Small penalty for staying still (encourages movement)
        if distance_reduced == 0:
            reward -= 1.0
        
        return reward
    
    def update_q_value(self, state: Tuple, action: int, reward: float, next_state: Tuple):
        """
        Update Q-value using Q-Learning algorithm.
        
        Q(s,a) = Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state after action
        """
        current_q = self.q_table[state][action]
        
        # Get max Q-value for next state
        next_state_actions = self.q_table[next_state]
        max_next_q = max(next_state_actions.values()) if next_state_actions else 0.0
        
        # Q-Learning update
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        
        self.q_table[state][action] = new_q
        self.total_rewards += reward
        self.step_count += 1
    
    def get_best_direction(self, taxi_pos: Tuple[int, int], passenger_pos: Tuple[int, int],
                          passenger_priority: str) -> Tuple[int, int]:
        """
        Get best direction to move towards passenger using RL.
        
        Args:
            taxi_pos: Current taxi position
            passenger_pos: Target passenger position
            passenger_priority: Passenger priority
        
        Returns:
            Direction vector (dx, dy) to move
        """
        state = self.get_state_key(taxi_pos, passenger_pos, passenger_priority)
        action = self.choose_action(state)
        
        # Convert action to direction
        directions = {
            0: (0, -1),   # Up
            1: (1, 0),    # Right
            2: (0, 1),    # Down
            3: (-1, 0),   # Left
            4: (0, 0)     # Stay
        }
        
        return directions.get(action, (0, 0))
    
    def learn_from_experience(self, old_pos: Tuple[int, int], new_pos: Tuple[int, int],
                            passenger_pos: Tuple[int, int], passenger_priority: str,
                            reached_passenger: bool):
        """
        Learn from an experience and update Q-table.
        
        Args:
            old_pos: Taxi position before action
            new_pos: Taxi position after action
            passenger_pos: Target passenger position
            passenger_priority: Passenger priority
            reached_passenger: Whether passenger was reached
        """
        if self.last_state is None:
            return
        
        # Calculate distance change
        old_dist = abs(old_pos[0] - passenger_pos[0]) + abs(old_pos[1] - passenger_pos[1])
        new_dist = abs(new_pos[0] - passenger_pos[0]) + abs(new_pos[1] - passenger_pos[1])
        distance_reduced = old_dist - new_dist
        
        # Calculate reward
        reward = self.calculate_reward(old_pos, passenger_pos, passenger_priority,
                                      distance_reduced, reached_passenger)
        
        # Get next state
        next_state = self.get_state_key(new_pos, passenger_pos, passenger_priority)
        
        # Update Q-value
        self.update_q_value(self.last_state, self.last_action, reward, next_state)
        
        # Save Q-table periodically (every 100 steps)
        if self.step_count % 100 == 0:
            self._save_q_table()
        
        # Update for next iteration
        self.last_state = next_state
        self.last_action = None  # Will be set by next action choice
    
    def start_episode(self, taxi_pos: Tuple[int, int], passenger_pos: Tuple[int, int],
                     passenger_priority: str):
        """
        Start a new episode (route to passenger).
        
        Args:
            taxi_pos: Starting taxi position
            passenger_pos: Target passenger position
            passenger_priority: Passenger priority
        """
        self.last_state = self.get_state_key(taxi_pos, passenger_pos, passenger_priority)
        state = self.last_state
        self.last_action = self.choose_action(state)
    
    def save_model(self):
        """Save the Q-table model."""
        self._save_q_table()
        print(f"💾 Saved RL model: {len(self.q_table)} states, avg reward: {self.total_rewards/max(1, self.step_count):.2f}")


class RLTaxiAgent:
    """
    Wrapper that adds RL routing capability to taxis.
    """
    
    def __init__(self, rl_router: RLRouter):
        """
        Initialize RL taxi agent.
        
        Args:
            rl_router: RL Router instance
        """
        self.rl_router = rl_router
    
    def get_rl_route_suggestion(self, taxi_pos: Tuple[int, int], passenger_pos: Tuple[int, int],
                               passenger_priority: str) -> Optional[Tuple[int, int]]:
        """
        Get route suggestion from RL agent.
        
        Args:
            taxi_pos: Current taxi position
            passenger_pos: Target passenger position
            passenger_priority: Passenger priority
        
        Returns:
            Suggested next position (dx, dy) relative to current, or None for fallback
        """
        # Start episode if not already started
        if self.rl_router.last_state is None:
            self.rl_router.start_episode(taxi_pos, passenger_pos, passenger_priority)
        
        # Get best direction from RL
        direction = self.rl_router.get_best_direction(taxi_pos, passenger_pos, passenger_priority)
        
        # Calculate next position
        next_pos = (taxi_pos[0] + direction[0], taxi_pos[1] + direction[1])
        
        return next_pos

