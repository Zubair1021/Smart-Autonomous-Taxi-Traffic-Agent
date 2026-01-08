"""
Pathfinding and road network utilities.
"""

import random
import networkx as nx
from networkx.generators.lattice import grid_2d_graph


def create_road_network(width, height, grid_spacing=1):
    """
    Create a grid-based road network using NetworkX.
    
    Args:
        width: Width of the city grid
        height: Height of the city grid
        grid_spacing: Spacing between grid points
    
    Returns:
        NetworkX graph representing the road network
    """
    G = grid_2d_graph(width, height)
    return G


def get_random_road_position(road_network):
    """
    Get a random position on the road network.
    
    Args:
        road_network: NetworkX graph of roads
    
    Returns:
        Random (x, y) position from the network
    """
    if len(road_network.nodes) == 0:
        return None
    return random.choice(list(road_network.nodes))

