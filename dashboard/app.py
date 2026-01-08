"""
Professional Dashboard using Plotly Dash for Smart Autonomous Taxi & Traffic System
"""

import dash
from dash import dcc, html, Input, Output, State, callback, callback_context, no_update
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import plotly.express as px
from datetime import datetime
import threading
import time
import numpy as np
import pandas as pd
import json

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.city_model import CityModel

# Initialize Dash app with Bootstrap theme
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP],
    title="Smart Autonomous Taxi & Traffic System",
    suppress_callback_exceptions=True
)

# Expose server for deployment platforms (Render, Heroku, etc.)
server = app.server

# Model parameters with new features
model_params = {
    "width": 30,
    "height": 30,
    "num_taxis": 10,
    "num_traffic_lights": 8,
    "passenger_spawn_rate": 0.1,
    "enable_rush_hour": True,
    "enable_weather": False
}

# Global model instance
model = None
model_thread = None
running = False
simulation_speed = 5  # Default speed: matches slider default (5 = ~1.5s per step for better analysis)
data_history = {
    'steps': [],
    'wait_times': [],
    'utilization': [],
    'traffic_density': [],
    'active_taxis': [],
    'waiting_passengers': [],
    'passengers_served': []
}

def run_simulation():
    """Run simulation in background thread."""
    global model, running, simulation_speed
    running = True
    while running and model.running:
        model.step()
        # Speed control: slider value 0-20 maps to 2.0s to 0.1s delay
        # Lower slider = slower (more delay), Higher slider = faster (less delay)
        delay = 2.0 - (simulation_speed * 0.095)  # Maps 0->2.0s, 20->0.1s
        delay = max(0.1, min(2.0, delay))  # Clamp between 0.1s and 2.0s
        time.sleep(delay)

# Initialize model
model = CityModel(**model_params)

# Define the layout
app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H1([
                    html.I(className="bi bi-taxi-front-fill me-2"),
                    "Smart Autonomous Taxi & Traffic System"
                ], className="text-white mb-2"),
                html.P("AI-Powered Urban Mobility Simulation", className="text-white-50 mb-0")
            ], className="header-content")
        ], width=12)
    ], className="header-section mb-4"),
    
    # Control Panel with Export
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Button([
                                html.I(className="bi bi-play-fill me-2"),
                                "Start"
                            ], id="btn-start", color="success", className="me-2")
                        ], width="auto"),
                        dbc.Col([
                            dbc.Button([
                                html.I(className="bi bi-pause-fill me-2"),
                                "Pause"
                            ], id="btn-pause", color="warning", className="me-2")
                        ], width="auto"),
                        dbc.Col([
                            dbc.Button([
                                html.I(className="bi bi-skip-forward me-2"),
                                "Step"
                            ], id="btn-step", color="info", className="me-2")
                        ], width="auto"),
                        dbc.Col([
                            dbc.Button([
                                html.I(className="bi bi-arrow-clockwise me-2"),
                                "Reset"
                            ], id="btn-reset", color="danger", className="me-2")
                        ], width="auto"),
                        dbc.Col([
                            html.Label("Speed:", className="me-2"),
                            dcc.Slider(
                                id="speed-slider",
                                min=0,
                                max=20,
                                step=1,
                                value=5,  # Default to slower speed (5 = ~1.5s per step)
                                marks={0: "Slow", 5: "Medium", 10: "Fast", 15: "Very Fast", 20: "Max"},
                                tooltip={"placement": "bottom", "always_visible": True}
                            ),
                            html.Small(id="speed-display", className="text-muted ms-2")
                        ], width=True),
                        dbc.Col([
                            dbc.ButtonGroup([
                                dbc.Button([
                                    html.I(className="bi bi-download me-2"),
                                    "Export CSV"
                                ], id="btn-export-csv", color="secondary", size="sm", outline=True),
                                dbc.Button([
                                    html.I(className="bi bi-filetype-json me-2"),
                                    "Export JSON"
                                ], id="btn-export-json", color="secondary", size="sm", outline=True)
                            ])
                        ], width="auto")
                    ], align="center")
                ])
            ], className="shadow-sm")
        ], width=12)
    ], className="mb-4"),
    
    # Performance Score & Alerts Row
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.I(className="bi bi-speedometer2 me-2"),
                    "System Performance Score"
                ]),
                dbc.CardBody([
                    html.Div([
                        html.H1(id="performance-score", className="text-center mb-2", style={"fontSize": "4rem", "fontWeight": "bold"}),
                        html.P(id="performance-status", className="text-center mb-0", style={"fontSize": "1.2rem"}),
                        html.Div(id="performance-details", className="mt-3")
                    ])
                ])
            ], className="shadow-sm", id="performance-card")
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.I(className="bi bi-bell-fill me-2"),
                    "Real-time Alerts",
                    html.Span(id="alert-count", className="badge bg-danger ms-2", children="0")
                ]),
                dbc.CardBody([
                    html.Div(id="alerts-panel", children=[])
                ], style={"maxHeight": "200px", "overflowY": "auto"})
            ], className="shadow-sm")
        ], width=9)
    ], className="mb-4"),
    
    # Statistics Summary Row
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.I(className="bi bi-graph-up me-2"),
                    "Statistics Summary"
                ]),
                dbc.CardBody([
                    html.Div(id="statistics-summary", children=[])
                ])
            ], className="shadow-sm")
        ], width=12)
    ], className="mb-4"),
    
    # Key Metrics Cards
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.I(className="bi bi-taxi-front metric-icon text-warning"),
                        html.Div([
                            html.H3(id="metric-total-taxis", className="mb-0"),
                            html.Small("Total Fleet", className="text-muted")
                        ])
                    ], className="d-flex align-items-center")
                ])
            ], className="metric-card shadow-sm border-0")
        ], width=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.I(className="bi bi-check-circle-fill metric-icon text-success"),
                        html.Div([
                            html.H3(id="metric-active-taxis", className="mb-0"),
                            html.Small("Active Service", className="text-muted")
                        ])
                    ], className="d-flex align-items-center")
                ])
            ], className="metric-card shadow-sm border-0")
        ], width=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.I(className="bi bi-hourglass-split metric-icon text-warning"),
                        html.Div([
                            html.H3(id="metric-waiting", className="mb-0"),
                            html.Small("Waiting Passengers", className="text-muted")
                        ])
                    ], className="d-flex align-items-center")
                ])
            ], className="metric-card shadow-sm border-0")
        ], width=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.I(className="bi bi-people-fill metric-icon text-info"),
                        html.Div([
                            html.H3(id="metric-served", className="mb-0"),
                            html.Small("Passengers Served", className="text-muted")
                        ])
                    ], className="d-flex align-items-center")
                ])
            ], className="metric-card shadow-sm border-0")
        ], width=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.I(className="bi bi-clock-history metric-icon text-danger"),
                        html.Div([
                            html.H3(id="metric-wait-time", className="mb-0"),
                            html.Small("Avg Wait Time", className="text-muted")
                        ])
                    ], className="d-flex align-items-center")
                ])
            ], className="metric-card shadow-sm border-0")
        ], width=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.I(className="bi bi-graph-up-arrow metric-icon text-success"),
                        html.Div([
                            html.H3(id="metric-utilization", className="mb-0"),
                            html.Small("Utilization", className="text-muted")
                        ])
                    ], className="d-flex align-items-center")
                ])
            ], className="metric-card shadow-sm border-0")
        ], width=2)
    ], className="mb-4"),
    
    # Charts Row
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Wait Time Analysis"),
                dbc.CardBody([
                    dcc.Graph(id="chart-wait-time", config={'displayModeBar': False})
                ])
            ], className="shadow-sm")
        ], width=4),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Taxi Utilization"),
                dbc.CardBody([
                    dcc.Graph(id="chart-utilization", config={'displayModeBar': False})
                ])
            ], className="shadow-sm")
        ], width=4),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Traffic Density"),
                dbc.CardBody([
                    dcc.Graph(id="chart-density", config={'displayModeBar': False})
                ])
            ], className="shadow-sm")
        ], width=4)
    ], className="mb-4"),
    
    # Advanced Analytics & Cost Analysis Row
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.I(className="bi bi-lightbulb-fill me-2"),
                    "Advanced Analytics & Insights"
                ]),
                dbc.CardBody([
                    html.Div(id="analytics-insights", children=[])
                ])
            ], className="shadow-sm")
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.I(className="bi bi-cash-stack me-2"),
                    "Cost Analysis & Revenue"
                ]),
                dbc.CardBody([
                    html.Div(id="cost-analysis", children=[])
                ])
            ], className="shadow-sm")
        ], width=6)
    ], className="mb-4"),
    
    # Simulation Grid Visualization
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.I(className="bi bi-grid-3x3-gap me-2"),
                    "City Simulation Grid",
                    html.Small(" (Click any agent to see details)", className="text-muted ms-2")
                ]),
                dbc.CardBody([
                    dcc.Graph(id="simulation-grid", config={'displayModeBar': False}),
                    html.Div(id="step-counter", className="text-center mt-3")
                ])
            ], className="shadow-sm")
        ], width=12)
    ], className="mb-4"),
    
    # Real-Time Heatmap Visualization
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.I(className="bi bi-map me-2"),
                    "Real-Time Demand Heatmap",
                    html.Small(" (Current passenger demand density)", className="text-muted ms-2")
                ]),
                dbc.CardBody([
                    dcc.Graph(id="heatmap-visualization", config={'displayModeBar': False})
                ])
            ], className="shadow-sm")
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.I(className="bi bi-graph-up-arrow me-2"),
                    "Demand Prediction Heatmap",
                    html.Small(" (AI-predicted future demand)", className="text-muted ms-2")
                ]),
                dbc.CardBody([
                    dcc.Graph(id="prediction-heatmap", config={'displayModeBar': False})
                ])
            ], className="shadow-sm")
        ], width=6)
    ], className="mb-4"),
    
    # Agent Details Modal
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle(id="modal-title")),
        dbc.ModalBody(id="modal-body"),
        dbc.ModalFooter([
            dbc.Button("Close", id="modal-close", className="ms-auto", n_clicks=0)
        ])
    ], id="agent-modal", is_open=False, size="lg"),
    
    # Hidden div to store clicked agent data
    dcc.Store(id="clicked-agent-store", data=None),
    
    # Auto-refresh interval
    dcc.Interval(
        id='interval-component',
        interval=500,  # Update every 500ms
        n_intervals=0,
        disabled=True  # Will be enabled when simulation starts
    ),
    
    # Store for running state
    dcc.Store(id='running-state-store', data=False),
    
    # Dummy outputs for button callbacks
    html.Div(id='dummy-output-step', style={'display': 'none'}),
    html.Div(id='dummy-output-reset', style={'display': 'none'}),
    html.Div(id='dummy-output-start', style={'display': 'none'}),
    html.Div(id='dummy-output-pause', style={'display': 'none'}),
    
    # Hidden download triggers
    dcc.Download(id="download-csv"),
    dcc.Download(id="download-json")
], fluid=True, className="dashboard-container")

# Custom CSS
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            }
            .dashboard-container {
                background: #f8f9fa;
                min-height: 100vh;
                padding: 20px;
            }
            .header-section {
                background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
                padding: 30px;
                border-radius: 15px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                margin-bottom: 30px !important;
            }
            .header-content h1 {
                font-size: 2.5rem;
                font-weight: 700;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            }
            .metric-card {
                transition: transform 0.2s, box-shadow 0.2s;
                border-left: 4px solid;
            }
            .metric-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 8px 16px rgba(0,0,0,0.15) !important;
            }
            .metric-icon {
                font-size: 2.5rem;
                margin-right: 15px;
            }
            .card {
                border-radius: 12px;
                border: none;
            }
            .card-header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                font-weight: 600;
                border-radius: 12px 12px 0 0 !important;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Callbacks - MUST be defined after layout
@app.callback(
    Output('metric-total-taxis', 'children'),
    Output('metric-active-taxis', 'children'),
    Output('metric-waiting', 'children'),
    Output('metric-served', 'children'),
    Output('metric-wait-time', 'children'),
    Output('metric-utilization', 'children'),
    Output('step-counter', 'children'),
    Output('chart-wait-time', 'figure'),
    Output('chart-utilization', 'figure'),
    Output('chart-density', 'figure'),
    Output('simulation-grid', 'figure'),
    Input('interval-component', 'n_intervals'),
    Input('btn-step', 'n_clicks'),
    prevent_initial_call=False
)
def update_dashboard(n_intervals, step_clicks):
    """Update dashboard with current model state."""
    global model, data_history, running
    
    # Check if this is triggered by interval or step button
    ctx = dash.callback_context
    if not ctx.triggered:
        trigger_id = 'interval-component'
    else:
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    try:
        # Always ensure n_intervals is valid
        if n_intervals is None:
            n_intervals = 0
        
        if model is None:
            empty_fig = go.Figure()
            empty_fig.update_layout(template='plotly_white', height=250)
            return (
                "0", "0", "0", "0", "0.0s", "0.0%", 
                html.H5("Simulation Step: 0", className="text-primary"),
                empty_fig, empty_fig, empty_fig, empty_fig
            )
        
        # Only update data history if simulation is running or step was clicked
        should_update = running or trigger_id == 'btn-step'
        
        from agents.taxi import Taxi
        from agents.passenger import Passenger
        
        taxis = [a for a in model.schedule.agents if isinstance(a, Taxi)]
        passengers = [a for a in model.schedule.agents if isinstance(a, Passenger)]
        
        active_taxis = len([t for t in taxis if t.status != "idle"])
        waiting = len([p for p in passengers if p.status == "waiting"])
        total_served = sum(t.total_passengers for t in taxis)
        avg_wait = model._calculate_avg_wait_time()
        utilization = model._calculate_taxi_utilization()
        traffic_density = model._calculate_traffic_density()
        
        # Update history only if simulation is running or step was clicked
        step = model.schedule.time
        if should_update:
            data_history['steps'].append(step)
            data_history['wait_times'].append(avg_wait)
            data_history['utilization'].append(utilization)
            data_history['traffic_density'].append(traffic_density)
            data_history['active_taxis'].append(active_taxis)
            data_history['waiting_passengers'].append(waiting)
            data_history['passengers_served'].append(total_served)
        
            # Keep only last 100 data points
            for key in data_history:
                if len(data_history[key]) > 100:
                    data_history[key] = data_history[key][-100:]
        
        # Create charts
        wait_time_fig = go.Figure()
        if data_history['steps']:
            wait_time_fig.add_trace(go.Scatter(
                x=data_history['steps'],
                y=data_history['wait_times'],
                mode='lines',
                name='Avg Wait Time',
                line=dict(color='#ef4444', width=2),
                fill='tozeroy',
                fillcolor='rgba(239,68,68,0.1)'
            ))
        wait_time_fig.update_layout(
            template='plotly_white',
            height=250,
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis_title="Step",
            yaxis_title="Wait Time (steps)",
            showlegend=False
        )
    
        utilization_fig = go.Figure()
        if data_history['steps']:
            utilization_fig.add_trace(go.Scatter(
                x=data_history['steps'],
                y=data_history['utilization'],
                mode='lines',
                name='Utilization',
                line=dict(color='#10b981', width=2),
                fill='tozeroy',
                fillcolor='rgba(16,185,129,0.1)'
            ))
        utilization_fig.update_layout(
            template='plotly_white',
            height=250,
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis_title="Step",
            yaxis_title="Utilization (%)",
            showlegend=False
        )
    
        density_fig = go.Figure()
        if data_history['steps']:
            density_fig.add_trace(go.Scatter(
                x=data_history['steps'],
                y=data_history['traffic_density'],
                mode='lines',
                name='Traffic Density',
                line=dict(color='#f59e0b', width=2),
                fill='tozeroy',
                fillcolor='rgba(245,158,11,0.1)'
            ))
        density_fig.update_layout(
            template='plotly_white',
            height=250,
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis_title="Step",
            yaxis_title="Density",
            showlegend=False
        )
    
        # Create professional grid visualization
        from agents.traffic_light import TrafficLight
    
        grid_fig = go.Figure()
    
        # Clean road network background
        road_network = model.road_network
        road_x, road_y = [], []
        for edge in road_network.edges():
            x0, y0 = edge[0]
            x1, y1 = edge[1]
            road_x.extend([x0, x1, None])
            road_y.extend([y0, y1, None])
        
        if road_x:
            grid_fig.add_trace(go.Scatter(
                x=road_x,
                y=road_y,
                mode='lines',
                name='Road Network',
                line=dict(color='#e5e7eb', width=0.8),
                hoverinfo='skip',
                showlegend=False
            ))
    
        # Subtle path visualization (only for active taxis)
        for taxi in taxis:
            if taxi.status != "idle" and hasattr(taxi, 'path') and taxi.path and len(taxi.path) > 0:
                path_x = [taxi.position[0]]
                path_y = [taxi.position[1]]
                for pos in taxi.path[:8]:
                    path_x.append(pos[0])
                    path_y.append(pos[1])
                
                if len(path_x) > 1:
                    status_color = {
                        "picking_up": "rgba(251,146,60,0.25)",
                        "transporting": "rgba(74,222,128,0.25)"
                    }.get(taxi.status, "rgba(156,163,175,0.2)")
                    
                    grid_fig.add_trace(go.Scatter(
                        x=path_x,
                        y=path_y,
                        mode='lines',
                        name=f'Route',
                        line=dict(color=status_color, width=1.5, dash='dot'),
                        hoverinfo='skip',
                        showlegend=False
                    ))
    
        # Enhanced taxi visualization with icons and no overlapping
        taxi_x, taxi_y, taxi_colors, taxi_text, taxi_ids, taxi_icons = [], [], [], [], [], []
    
        # Group taxis by position to prevent overlapping
        position_groups = {}
        for taxi in taxis:
            pos = taxi.position
            key = f"{pos[0]}_{pos[1]}"
            if key not in position_groups:
                position_groups[key] = []
            position_groups[key].append(taxi)
        
        for key, taxi_group in position_groups.items():
            for idx, taxi in enumerate(taxi_group):
                base_x, base_y = taxi.position
                # Offset multiple taxis at same position
                offset_x = (idx % 3 - 1) * 0.3
                offset_y = (idx // 3) * 0.3
                
                taxi_x.append(base_x + offset_x)
                taxi_y.append(base_y + offset_y)
                taxi_ids.append(f"taxi_{taxi.unique_id}")
                
                taxi_type = getattr(taxi, 'taxi_type', 'economy')
                passenger_count = len(taxi.passengers) if hasattr(taxi, 'passengers') else 0
                
                if taxi.status == "idle":
                    color_map = {"economy": "#fbbf24", "premium": "#f59e0b", "luxury": "#d97706"}
                elif taxi.status == "picking_up":
                    color_map = {"economy": "#fb923c", "premium": "#f97316", "luxury": "#ea580c"}
                else:
                    color_map = {"economy": "#10b981", "premium": "#059669", "luxury": "#047857"}
                
                # Set icon based on taxi type
                if taxi_type == "economy":
                    icon = "🚕"
                elif taxi_type == "premium":
                    icon = "🚗"
                else:
                    icon = "🏎️"
                
                taxi_icons.append(icon)
                taxi_colors.append(color_map.get(taxi_type, "#fbbf24"))
                
                revenue = getattr(taxi, 'revenue', 0.0)
                distance = getattr(taxi, 'total_distance', 0)
                served = getattr(taxi, 'total_passengers', 0)
                
                # Enhanced tooltip with passenger info
                passenger_info = ""
                if hasattr(taxi, 'passengers') and taxi.passengers:
                    passenger_info = f"<br>Onboard: {', '.join([f'P#{p.unique_id}' for p in taxi.passengers])}"
                
                taxi_text.append(
                    f"<b>{icon} {taxi_type.upper()} TAXI #{taxi.unique_id}</b><br>"
                    f"Status: <b>{taxi.status.replace('_', ' ').title()}</b><br>"
                    f"Passengers: {passenger_count}/{getattr(taxi, 'capacity', 1)}{passenger_info}<br>"
                    f"Served: {served} | Distance: {distance}<br>"
                    f"Revenue: ${revenue:.2f}<br>"
                    f"<i>Click for details</i>"
                )
        
        if taxi_x:
            # Calculate marker size based on passenger count
            taxi_sizes = []
            for taxi in taxis:
                passenger_count = len(taxi.passengers) if hasattr(taxi, 'passengers') else 0
                base_size = 35
                size = base_size + (passenger_count * 8)  # Larger when carrying passengers
                taxi_sizes.append(size)
            
            grid_fig.add_trace(go.Scatter(
                x=taxi_x,
                y=taxi_y,
                mode='markers+text',
                name='Taxis',
                marker=dict(
                    size=taxi_sizes,
                    color=taxi_colors,
                    symbol='circle',
                    line=dict(width=3, color='#ffffff'),
                    opacity=0.85
                ),
                text=taxi_icons,
                textposition="middle center",
                textfont=dict(size=18),
                hovertemplate='%{text}<extra></extra>',
                customdata=taxi_ids,
                ids=taxi_ids
            ))
            
            # Add passenger count labels on taxis
            taxi_labels_x, taxi_labels_y, taxi_labels_text = [], [], []
            for idx, taxi in enumerate(taxis):
                if hasattr(taxi, 'passengers') and taxi.passengers:
                    passenger_count = len(taxi.passengers)
                    if passenger_count > 0:
                        # Position label slightly above taxi
                        base_x, base_y = taxi.position
                        offset_x = (idx % 3 - 1) * 0.3
                        offset_y = (idx // 3) * 0.3
                        taxi_labels_x.append(base_x + offset_x)
                        taxi_labels_y.append(base_y + offset_y + 0.8)
                        taxi_labels_text.append(f"{passenger_count}")
            
            if taxi_labels_x:
                grid_fig.add_trace(go.Scatter(
                    x=taxi_labels_x,
                    y=taxi_labels_y,
                    mode='text',
                    name='Passenger Count',
                    text=taxi_labels_text,
                    textfont=dict(size=12, color='white', family='Arial Black'),
                    textposition="middle center",
                    hoverinfo='skip',
                    showlegend=False
                ))
    
        # Enhanced passenger visualization with icons and no overlapping
        passenger_x, passenger_y, passenger_colors, passenger_text, passenger_ids, passenger_icons = [], [], [], [], [], []
    
        # Group passengers by position to prevent overlapping
        passenger_position_groups = {}
        for passenger in passengers:
            if passenger.status == "waiting":
                pos = passenger.position
                key = f"{pos[0]}_{pos[1]}"
                if key not in passenger_position_groups:
                    passenger_position_groups[key] = []
                passenger_position_groups[key].append(passenger)
        
        for key, passenger_group in passenger_position_groups.items():
            for idx, passenger in enumerate(passenger_group):
                base_x, base_y = passenger.position
                # Offset multiple passengers at same position
                offset_x = (idx % 3 - 1) * 0.25
                offset_y = (idx // 3) * 0.25
                
                passenger_x.append(base_x + offset_x)
                passenger_y.append(base_y + offset_y)
                passenger_ids.append(f"passenger_{passenger.unique_id}")
                
                priority = getattr(passenger, 'priority', 'regular')
                
                # Check if a taxi is coming to pick up this passenger
                assigned_taxi = None
                for taxi in taxis:
                    if taxi.status == "picking_up" and hasattr(taxi, 'path') and taxi.path:
                        taxi_dest = taxi.path[-1] if taxi.path else None
                        if taxi_dest == passenger.position:
                            assigned_taxi = taxi
                            break
                
                if priority == "emergency":
                    passenger_colors.append("#ef4444")
                    icon = "🚨"
                    taxi_info = f"<br>🚕 <b>Taxi #{assigned_taxi.unique_id} coming!</b>" if assigned_taxi else ""
                    passenger_text.append(
                        f"<b>{icon} EMERGENCY PASSENGER</b><br>"
                        f"Wait: <b>{passenger.wait_time} steps</b><br>"
                        f"Priority: HIGHEST{taxi_info}<br>"
                        f"<i>Click for details</i>"
                    )
                elif priority == "vip":
                    passenger_colors.append("#f59e0b")
                    icon = "⭐"
                    taxi_info = f"<br>🚕 <b>Taxi #{assigned_taxi.unique_id} coming!</b>" if assigned_taxi else ""
                    passenger_text.append(
                        f"<b>{icon} VIP PASSENGER</b><br>"
                        f"Wait: {passenger.wait_time} steps<br>"
                        f"Priority: HIGH{taxi_info}<br>"
                        f"<i>Click for details</i>"
                    )
                else:
                    passenger_colors.append("#3b82f6")
                    icon = "👤"
                    taxi_info = f"<br>🚕 <b>Taxi #{assigned_taxi.unique_id} coming!</b>" if assigned_taxi else ""
                    passenger_text.append(
                        f"<b>{icon} REGULAR PASSENGER</b><br>"
                        f"Wait: {passenger.wait_time} steps<br>"
                        f"Priority: Standard{taxi_info}<br>"
                        f"<i>Click for details</i>"
                    )
                
                passenger_icons.append(icon)
        
        if passenger_x:
            # Calculate passenger sizes based on priority
            passenger_sizes = []
            for passenger in passengers:
                if passenger.status == "waiting":
                    priority = getattr(passenger, 'priority', 'regular')
                    if priority == "emergency":
                        size = 28
                    elif priority == "vip":
                        size = 24
                    else:
                        size = 20
                    passenger_sizes.append(size)
            
            grid_fig.add_trace(go.Scatter(
                x=passenger_x,
                y=passenger_y,
                mode='markers+text',
                name='Waiting Passengers',
                marker=dict(
                    size=passenger_sizes,
                    color=passenger_colors,
                    symbol='circle',
                    line=dict(width=2.5, color='#ffffff'),
                    opacity=0.85
                ),
                text=passenger_icons,
                textposition="middle center",
                textfont=dict(size=16),
                hovertemplate='%{text}<extra></extra>',
                customdata=passenger_ids,
                ids=passenger_ids
            ))
        
        # Show connection lines from passenger perspective - which taxi is coming
        pickup_lines_x, pickup_lines_y, pickup_line_colors = [], [], []
        passenger_to_taxi = {}  # Track which taxi is picking up which passenger
        
        # For each waiting passenger, find the taxi that's coming to pick them up
        for passenger in passengers:
            if passenger.status == "waiting":
                # Find taxi that's heading to this passenger's position
                for taxi in taxis:
                    if taxi.status == "picking_up" and hasattr(taxi, 'path') and taxi.path:
                        # Check if taxi's path destination matches passenger position
                        taxi_dest = taxi.path[-1] if taxi.path else None
                        if taxi_dest == passenger.position:
                            # Draw line from passenger to taxi
                            pickup_lines_x.extend([passenger.position[0], taxi.position[0], None])
                            pickup_lines_y.extend([passenger.position[1], taxi.position[1], None])
                            
                            # Color based on passenger priority
                            priority = getattr(passenger, 'priority', 'regular')
                            if priority == "emergency":
                                pickup_line_colors.append("#ef4444")  # Red for emergency
                            elif priority == "vip":
                                pickup_line_colors.append("#f59e0b")  # Orange for VIP
                            else:
                                pickup_line_colors.append("#3b82f6")  # Blue for regular
                            
                            passenger_to_taxi[passenger.unique_id] = taxi.unique_id
                            break
        
        # Also show lines from taxi perspective (for taxis picking up)
        for taxi in taxis:
            if taxi.status == "picking_up" and hasattr(taxi, 'path') and taxi.path:
                taxi_dest = taxi.path[-1] if taxi.path else None
                # Only add if not already added from passenger perspective
                already_added = False
                for passenger in passengers:
                    if passenger.status == "waiting" and passenger.position == taxi_dest:
                        if passenger.unique_id in passenger_to_taxi:
                            already_added = True
                            break
                
                if not already_added:
                    # Find passenger at destination
                    for passenger in passengers:
                        if passenger.status == "waiting" and passenger.position == taxi_dest:
                            pickup_lines_x.extend([taxi.position[0], passenger.position[0], None])
                            pickup_lines_y.extend([taxi.position[1], passenger.position[1], None])
                            
                            priority = getattr(passenger, 'priority', 'regular')
                            if priority == "emergency":
                                pickup_line_colors.append("#ef4444")
                            elif priority == "vip":
                                pickup_line_colors.append("#f59e0b")
                            else:
                                pickup_line_colors.append("#3b82f6")
                            break
        
        if pickup_lines_x:
            # Create multiple traces for different colored lines
            # Group lines by color
            color_groups = {}
            for i in range(0, len(pickup_lines_x), 3):
                if pickup_lines_x[i] is not None:
                    color = pickup_line_colors[i // 3] if i // 3 < len(pickup_line_colors) else "#f59e0b"
                    if color not in color_groups:
                        color_groups[color] = {'x': [], 'y': []}
                    color_groups[color]['x'].extend([pickup_lines_x[i], pickup_lines_x[i+1], None])
                    color_groups[color]['y'].extend([pickup_lines_y[i], pickup_lines_y[i+1], None])
            
            # Add trace for each color
            for color, coords in color_groups.items():
                line_style = "dash" if color == "#3b82f6" else "solid"
                line_width = 4 if color == "#ef4444" else 3
                grid_fig.add_trace(go.Scatter(
                    x=coords['x'],
                    y=coords['y'],
                    mode='lines',
                    name='Taxi Coming',
                    line=dict(color=color, width=line_width, dash=line_style),
                    hoverinfo='skip',
                    showlegend=False
                ))
    
        # Enhanced traffic light visualization with icons
        light_x, light_y, light_colors, light_text, light_ids, light_icons = [], [], [], [], [], []
        for agent in model.schedule.agents:
            if isinstance(agent, TrafficLight):
                light_x.append(agent.position[0])
                light_y.append(agent.position[1])
                light_ids.append(f"light_{agent.unique_id}")
                
                state = agent.get_state()
                direction = getattr(agent, 'direction', 'N/A')
                
                if state == "green":
                    light_colors.append("#10b981")
                    icon = "🟢"
                    light_text.append(f"<b>{icon} GREEN LIGHT</b><br>Direction: {direction.title()}<br>Status: GO<br><i>Click for details</i>")
                else:
                    light_colors.append("#ef4444")
                    icon = "🔴"
                    light_text.append(f"<b>{icon} RED LIGHT</b><br>Direction: {direction.title()}<br>Status: STOP<br><i>Click for details</i>")
                
                light_icons.append(icon)
        
        if light_x:
            grid_fig.add_trace(go.Scatter(
                x=light_x,
                y=light_y,
                mode='markers+text',
                name='Traffic Lights',
                marker=dict(
                    size=30,
                    color=light_colors,
                    symbol='square',
                    line=dict(width=3, color='#ffffff'),
                    opacity=0.9
                ),
                text=light_icons,
                textposition="middle center",
                textfont=dict(size=18),
                hovertemplate='%{text}<extra></extra>',
                customdata=light_ids,
                ids=light_ids
            ))
    
        # Professional clean layout
        grid_fig.update_layout(
        template='plotly_white',
        xaxis=dict(
            range=[-1, model.width], 
            showgrid=True, 
            gridcolor='#e5e7eb', 
            gridwidth=1,
            title=dict(text='X Position', font=dict(size=12, color='#6b7280')),
            showline=True,
            linecolor='#d1d5db',
            linewidth=1.5,
            zeroline=False
        ),
        yaxis=dict(
            range=[-1, model.height], 
            showgrid=True, 
            gridcolor='#e5e7eb', 
            gridwidth=1,
            title=dict(text='Y Position', font=dict(size=12, color='#6b7280')),
            scaleanchor="x", 
            scaleratio=1,
            showline=True,
            linecolor='#d1d5db',
            linewidth=1.5,
            zeroline=False
        ),
        height=900,
        margin=dict(l=70, r=40, t=60, b=70),
        showlegend=True,
        legend=dict(
            orientation="h", 
            yanchor="bottom", 
            y=1.02, 
            xanchor="center", 
            x=0.5,
            bgcolor='rgba(255,255,255,0.98)',
            bordercolor='#d1d5db',
            borderwidth=2,
            font=dict(size=11, color='#374151'),
            itemclick="toggleothers",
            itemdoubleclick="toggle"
        ),
        plot_bgcolor='#fafafa',
        paper_bgcolor='#ffffff',
        # title=dict(
        #     text="<b>City Simulation Grid</b>",
        #     x=0.5,
        #     xanchor='center',
        #     font=dict(size=16, color='#111827', family='Arial'),
        #     pad=dict(t=15, b=10)
        # ),
        hovermode='closest',
        modebar_remove=['lasso2d', 'select2d', 'autoScale2d', 'resetScale2d']
        )
    
        return (
            str(len(taxis)),
            str(active_taxis),
            str(waiting),
            str(total_served),
            f"{avg_wait:.1f}s",
            f"{utilization:.1f}%",
            html.H5(f"Simulation Step: {step}", className="text-primary"),
            wait_time_fig,
            utilization_fig,
            density_fig,
            grid_fig
        )
    except Exception as e:
        # Return safe defaults on any error
        empty_fig = go.Figure()
        empty_fig.update_layout(template='plotly_white', height=250)
        return (
            "0", "0", "0", "0", "0.0s", "0.0%", 
            html.H5("Simulation Step: 0", className="text-primary"),
            empty_fig, empty_fig, empty_fig, empty_fig
        )

# Interval control is handled by start/pause callbacks - no separate callback needed

# Callback to handle grid clicks and open modal
@app.callback(
    Output('agent-modal', 'is_open'),
    Output('modal-title', 'children'),
    Output('modal-body', 'children'),
    Output('clicked-agent-store', 'data'),
    Input('simulation-grid', 'clickData'),
    Input('modal-close', 'n_clicks'),
    State('agent-modal', 'is_open'),
    State('clicked-agent-store', 'data')
)
def toggle_modal(clickData, close_clicks, is_open, stored_data):
    """Handle grid clicks to show agent details in modal."""
    global model
    
    ctx = dash.callback_context
    if not ctx.triggered:
        return False, "", "", None
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Close modal if close button clicked
    if trigger_id == 'modal-close':
        return False, "", "", None
    
    # Open modal if agent clicked
    if trigger_id == 'simulation-grid' and clickData:
        points = clickData.get('points', [])
        if points:
            point = points[0]
            # Try to get customdata or pointNumber to identify agent
            customdata = point.get('customdata')
            point_number = point.get('pointNumber', -1)
            curve_number = point.get('curveNumber', -1)
            
            if customdata and model:
                try:
                    agent_id = customdata
                    agent_type, agent_num = agent_id.split('_', 1)
                    
                    # Find the agent
                    agent = None
                    for a in model.schedule.agents:
                        if str(a.unique_id) == agent_num:
                            agent = a
                            break
                    
                    if agent:
                        # Create modal content based on agent type
                        if agent_type == 'taxi':
                            return create_taxi_modal(agent, True)
                        elif agent_type == 'passenger':
                            return create_passenger_modal(agent, True)
                        elif agent_type == 'light':
                            return create_traffic_light_modal(agent, True)
                except (ValueError, AttributeError):
                    pass
            
            # Fallback: try to find agent by position
            if point_number >= 0 and model:
                x = point.get('x')
                y = point.get('y')
                
                # Find agent at this position
                from agents.taxi import Taxi
                from agents.passenger import Passenger
                from agents.traffic_light import TrafficLight
                
                cell_contents = model.grid.get_cell_list_contents([(int(x), int(y))])
                for agent in cell_contents:
                    if isinstance(agent, Taxi):
                        return create_taxi_modal(agent, True)
                    elif isinstance(agent, Passenger):
                        return create_passenger_modal(agent, True)
                    elif isinstance(agent, TrafficLight):
                        return create_traffic_light_modal(agent, True)
    
    return False, "", "", None

def create_taxi_modal(taxi, is_open):
    """Create modal content for taxi agent."""
    global model
    passenger_count = len(taxi.passengers) if hasattr(taxi, 'passengers') else 0
    revenue = getattr(taxi, 'revenue', 0.0)
    distance = getattr(taxi, 'total_distance', 0)
    served = getattr(taxi, 'total_passengers', 0)
    capacity = getattr(taxi, 'capacity', 1)
    taxi_type = getattr(taxi, 'taxi_type', 'economy')
    
    # Get passenger details
    passenger_list = []
    if hasattr(taxi, 'passengers') and taxi.passengers:
        for passenger in taxi.passengers:
            priority = getattr(passenger, 'priority', 'regular')
            priority_icon = "🚨" if priority == "emergency" else "⭐" if priority == "vip" else "👤"
            passenger_list.append({
                'id': passenger.unique_id,
                'priority': priority,
                'icon': priority_icon,
                'destination': getattr(passenger, 'destination', 'N/A')
            })
    
    # Check if taxi is picking up a passenger
    picking_up_passenger = None
    if taxi.status == "picking_up" and hasattr(taxi, 'path') and taxi.path and model:
        taxi_dest = taxi.path[-1] if taxi.path else None
        from agents.passenger import Passenger
        for agent in model.schedule.agents:
            if isinstance(agent, Passenger) and agent.status == "waiting" and agent.position == taxi_dest:
                picking_up_passenger = agent
                break
    
    title = f"🚕 {taxi_type.upper()} Taxi #{taxi.unique_id}"
    
    body = dbc.Container([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Basic Information"),
                    dbc.CardBody([
                        html.P([
                            html.Strong("Type: "), taxi_type.title()
                        ]),
                        html.P([
                            html.Strong("Status: "), 
                            html.Span(
                                taxi.status.replace('_', ' ').title(),
                                className=f"badge bg-{'warning' if taxi.status == 'idle' else 'info' if taxi.status == 'picking_up' else 'success'} ms-2"
                            )
                        ]),
                        html.P([
                            html.Strong("Position: "), f"({taxi.position[0]}, {taxi.position[1]})"
                        ]),
                        html.P([
                            html.Strong("Capacity: "), f"{capacity} passenger(s)"
                        ])
                    ])
                ], className="mb-3")
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Performance Metrics"),
                    dbc.CardBody([
                        html.P([
                            html.Strong("Current Passengers: "), 
                            html.Span(f"{passenger_count}/{capacity}", className="badge bg-primary ms-2")
                        ]),
                        html.P([
                            html.Strong("Total Served: "), 
                            html.Span(str(served), className="badge bg-success ms-2")
                        ]),
                        html.P([
                            html.Strong("Distance Traveled: "), f"{distance} units"
                        ]),
                        html.P([
                            html.Strong("Revenue Earned: "), 
                            html.Span(f"${revenue:.2f}", className="text-success fw-bold")
                        ])
                    ])
                ], className="mb-3")
            ], width=6)
        ]),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className="bi bi-people-fill me-2"),
                        f"Passengers Onboard ({passenger_count})"
                    ]),
                    dbc.CardBody([
                        html.Div([
                            dbc.ListGroup([
                                dbc.ListGroupItem([
                                    html.Div([
                                        html.Span(f"{p['icon']} Passenger #{p['id']}", className="fw-bold me-2"),
                                        html.Span(
                                            p['priority'].upper(),
                                            className=f"badge bg-{'danger' if p['priority'] == 'emergency' else 'warning' if p['priority'] == 'vip' else 'secondary'} ms-2"
                                        )
                                    ], className="d-flex justify-content-between align-items-center"),
                                    html.Small(f"Destination: {p['destination']}", className="text-muted d-block mt-1")
                                ]) for p in passenger_list
                            ]) if passenger_list else html.P("No passengers currently onboard", className="text-muted mb-0")
                        ])
                    ])
                ], className="mb-3")
            ], width=12)
        ]) if passenger_list else html.Div(),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className="bi bi-person-walking me-2"),
                        "Picking Up"
                    ]),
                    dbc.CardBody([
                        html.Div([
                            html.P([
                                html.Span(f"🚨 Passenger #{picking_up_passenger.unique_id}", className="fw-bold me-2"),
                                html.Span(
                                    getattr(picking_up_passenger, 'priority', 'regular').upper(),
                                    className=f"badge bg-{'danger' if getattr(picking_up_passenger, 'priority', 'regular') == 'emergency' else 'warning' if getattr(picking_up_passenger, 'priority', 'regular') == 'vip' else 'secondary'} ms-2"
                                )
                            ]),
                            html.Small(f"Position: {picking_up_passenger.position}", className="text-muted")
                        ]) if picking_up_passenger else html.P("Not currently picking up any passenger", className="text-muted mb-0")
                    ])
                ], className="mb-3")
            ], width=12)
        ]) if taxi.status == "picking_up" else html.Div(),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Route Information"),
                    dbc.CardBody([
                        html.P([
                            html.Strong("Path Length: "), 
                            len(taxi.path) if hasattr(taxi, 'path') and taxi.path else 0
                        ]),
                        html.P([
                            html.Strong("Destinations: "), 
                            len(taxi.destinations) if hasattr(taxi, 'destinations') else 0
                        ]),
                        html.P([
                            html.Strong("Next Destination: "), 
                            str(taxi.destinations[0]) if hasattr(taxi, 'destinations') and taxi.destinations else "None"
                        ])
                    ])
                ])
            ], width=12)
        ])
    ], fluid=True)
    
    return True, title, body, f"taxi_{taxi.unique_id}"

def create_passenger_modal(passenger, is_open):
    """Create modal content for passenger agent."""
    global model
    priority = getattr(passenger, 'priority', 'regular')
    destination = getattr(passenger, 'destination', None)
    max_wait = getattr(passenger, 'max_wait_time', 500)
    spawn_time = getattr(passenger, 'spawn_time', 0)
    
    priority_badge_color = {
        'emergency': 'danger',
        'vip': 'warning',
        'regular': 'secondary'
    }.get(priority, 'secondary')
    
    # Find which taxi is picking up this passenger
    assigned_taxi = None
    if passenger.status == "waiting" and model:
        from agents.taxi import Taxi
        for agent in model.schedule.agents:
            if isinstance(agent, Taxi) and agent.status == "picking_up" and hasattr(agent, 'path') and agent.path:
                taxi_dest = agent.path[-1] if agent.path else None
                if taxi_dest == passenger.position:
                    assigned_taxi = agent
                    break
    
    # Check if passenger is in a taxi
    in_taxi = None
    if passenger.status == "in_taxi" and model:
        from agents.taxi import Taxi
        for agent in model.schedule.agents:
            if isinstance(agent, Taxi) and hasattr(agent, 'passengers') and passenger in agent.passengers:
                in_taxi = agent
                break
    
    priority_icon = "🚨" if priority == "emergency" else "⭐" if priority == "vip" else "👤"
    title = f"{priority_icon} Passenger #{passenger.unique_id}"
    
    body = dbc.Container([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Basic Information"),
                    dbc.CardBody([
                        html.P([
                            html.Strong("Priority: "), 
                            html.Span(
                                priority.upper(),
                                className=f"badge bg-{priority_badge_color} ms-2"
                            )
                        ]),
                        html.P([
                            html.Strong("Status: "), 
                            html.Span(
                                passenger.status.replace('_', ' ').title(),
                                className="badge bg-info ms-2"
                            )
                        ]),
                        html.P([
                            html.Strong("Current Position: "), f"({passenger.position[0]}, {passenger.position[1]})"
                        ]),
                        html.P([
                            html.Strong("Destination: "), 
                            str(destination) if destination else "N/A"
                        ])
                    ])
                ], className="mb-3")
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Wait Time Information"),
                    dbc.CardBody([
                        html.P([
                            html.Strong("Wait Time: "), 
                            html.Span(
                                f"{passenger.wait_time} steps",
                                className="badge bg-warning ms-2"
                            )
                        ]),
                        html.P([
                            html.Strong("Max Wait Time: "), f"{max_wait} steps"
                        ]),
                        html.P([
                            html.Strong("Time Remaining: "), 
                            html.Span(
                                f"{max_wait - passenger.wait_time} steps",
                                className="text-success" if (max_wait - passenger.wait_time) > 50 else "text-danger"
                            )
                        ]),
                        html.P([
                            html.Strong("Spawned At: "), f"Step {spawn_time}"
                        ])
                    ])
                ], className="mb-3")
            ], width=6)
        ]),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className="bi bi-taxi-front-fill me-2"),
                        "Taxi Assignment"
                    ]),
                    dbc.CardBody([
                        html.Div([
                            html.P([
                                html.Span("🚕 ", className="me-2"),
                                html.Strong(f"Taxi #{assigned_taxi.unique_id}"),
                                html.Span(
                                    f" ({getattr(assigned_taxi, 'taxi_type', 'economy').title()})",
                                    className="text-muted ms-2"
                                )
                            ], className="mb-2"),
                            html.P([
                                html.Small("Status: ", className="text-muted"),
                                html.Span(
                                    assigned_taxi.status.replace('_', ' ').title(),
                                    className="badge bg-info ms-2"
                                )
                            ], className="mb-2"),
                            html.P([
                                html.Small("Taxi Position: ", className="text-muted"),
                                f"({assigned_taxi.position[0]}, {assigned_taxi.position[1]})"
                            ], className="mb-0"),
                            html.P([
                                html.Small("Distance: ", className="text-muted"),
                                f"{abs(assigned_taxi.position[0] - passenger.position[0]) + abs(assigned_taxi.position[1] - passenger.position[1])} units away"
                            ], className="mb-0 text-success")
                        ]) if assigned_taxi else (
                            html.Div([
                                html.P([
                                    html.Span("🚕 ", className="me-2"),
                                    html.Strong(f"Taxi #{in_taxi.unique_id}"),
                                    html.Span(
                                        f" ({getattr(in_taxi, 'taxi_type', 'economy').title()})",
                                        className="text-muted ms-2"
                                    )
                                ], className="mb-2"),
                                html.P([
                                    html.Small("Status: ", className="text-muted"),
                                    html.Span(
                                        "In Vehicle",
                                        className="badge bg-success ms-2"
                                    )
                                ], className="mb-0")
                            ]) if in_taxi else html.P("No taxi assigned yet. Waiting for pickup...", className="text-muted mb-0")
                        )
                    ])
                ], className="mb-3")
            ], width=12)
        ])
    ], fluid=True)
    
    return True, title, body, f"passenger_{passenger.unique_id}"

def create_traffic_light_modal(light, is_open):
    """Create modal content for traffic light agent."""
    state = light.get_state()
    direction = getattr(light, 'direction', 'N/A')
    time_in_state = getattr(light, 'time_in_state', 0)
    
    title = f"🚦 Traffic Light #{light.unique_id}"
    
    body = dbc.Container([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Status Information"),
                    dbc.CardBody([
                        html.P([
                            html.Strong("Current State: "), 
                            html.Span(
                                state.upper(),
                                className=f"badge bg-{'success' if state == 'green' else 'danger'} ms-2"
                            )
                        ]),
                        html.P([
                            html.Strong("Direction: "), 
                            html.Span(
                                direction.title(),
                                className="badge bg-info ms-2"
                            )
                        ]),
                        html.P([
                            html.Strong("Position: "), f"({light.position[0]}, {light.position[1]})"
                        ]),
                        html.P([
                            html.Strong("Time in State: "), f"{time_in_state} steps"
                        ])
                    ])
                ], className="mb-3")
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Timing Configuration"),
                    dbc.CardBody([
                        html.P([
                            html.Strong("Min Green Time: "), 
                            getattr(light, 'min_green_time', 5)
                        ]),
                        html.P([
                            html.Strong("Max Green Time: "), 
                            getattr(light, 'max_green_time', 15)
                        ]),
                        html.P([
                            html.Strong("Min Red Time: "), 
                            getattr(light, 'min_red_time', 3)
                        ]),
                        html.P([
                            html.Strong("Status: "), 
                            html.Span(
                                "GO" if state == "green" else "STOP",
                                className=f"badge bg-{'success' if state == 'green' else 'danger'} ms-2"
                            )
                        ])
                    ])
                ], className="mb-3")
            ], width=6)
        ])
    ], fluid=True)
    
    return True, title, body, f"light_{light.unique_id}"

# Combined callback for all simulation control buttons
@app.callback(
    Output('dummy-output-start', 'children'),
    Output('dummy-output-pause', 'children'),
    Output('dummy-output-step', 'children'),
    Output('dummy-output-reset', 'children'),
    Output('running-state-store', 'data'),
    Input('btn-start', 'n_clicks'),
    Input('btn-pause', 'n_clicks'),
    Input('btn-step', 'n_clicks'),
    Input('btn-reset', 'n_clicks'),
    prevent_initial_call=True
)
def control_simulation(start_clicks, pause_clicks, step_clicks, reset_clicks):
    """Handle all simulation control buttons in a single callback."""
    global model, running, model_thread, data_history
    
    ctx = dash.callback_context
    if not ctx.triggered:
        return "", "", "", "", no_update
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    try:
        if trigger_id == 'btn-start':
            if start_clicks and not running:
                running = True
                model_thread = threading.Thread(target=run_simulation, daemon=True)
                model_thread.start()
                return "", "", "", "", True  # Enable simulation
            return "", "", "", "", no_update
            
        elif trigger_id == 'btn-pause':
            if pause_clicks:
                running = False
                return "", "", "", "", False  # Disable simulation
            return "", "", "", "", no_update
            
        elif trigger_id == 'btn-step':
            if step_clicks and model:
                model.step()
                return "", "", "", "", no_update
            return "", "", "", "", no_update
            
        elif trigger_id == 'btn-reset':
            if reset_clicks:
                running = False
                model = CityModel(**model_params)
                data_history = {key: [] for key in data_history}
                return "", "", "", "", False  # Disable simulation and reset
            return "", "", "", "", no_update
    except Exception as e:
        print(f"Error in control_simulation: {e}")
        pass
    
    return "", "", "", "", no_update

# Single callback to control interval based on running state
@app.callback(
    Output('interval-component', 'disabled'),
    Input('running-state-store', 'data'),
    prevent_initial_call=True
)
def control_interval(is_running):
    """Control interval component based on running state."""
    if is_running is None or is_running is False:
        return True  # Disabled when not running
    return False  # Enabled when running

# Callback to control simulation speed
@app.callback(
    Output('speed-display', 'children'),
    Input('speed-slider', 'value'),
    prevent_initial_call=False
)
def update_speed(speed_value):
    """Update simulation speed based on slider value."""
    global simulation_speed
    if speed_value is None:
        speed_value = 5
    
    simulation_speed = speed_value
    
    # Calculate delay for display
    delay = 2.0 - (speed_value * 0.095)
    delay = max(0.1, min(2.0, delay))
    
    # Display speed info
    if speed_value <= 3:
        speed_text = "Very Slow"
    elif speed_value <= 7:
        speed_text = "Slow"
    elif speed_value <= 12:
        speed_text = "Medium"
    elif speed_value <= 17:
        speed_text = "Fast"
    else:
        speed_text = "Very Fast"
    
    return f"({speed_text} - {delay:.2f}s/step)"

# Real-Time Heatmap Visualization Callback
@app.callback(
    Output('heatmap-visualization', 'figure'),
    Output('prediction-heatmap', 'figure'),
    Input('interval-component', 'n_intervals'),
    Input('btn-step', 'n_clicks'),
    prevent_initial_call=False
)
def update_heatmaps(n_intervals, step_clicks):
    """Generate real-time heatmap showing passenger demand density."""
    global model
    
    # Create empty figure as default
    empty_fig = go.Figure()
    empty_fig.update_layout(
        template='plotly_white',
        height=400,
        xaxis_title="X Position",
        yaxis_title="Y Position",
        title="No data available"
    )
    
    if model is None:
        return empty_fig, empty_fig
    
    try:
        from agents.passenger import Passenger
        from agents.taxi import Taxi
        
        # Get model dimensions
        width = model.width
        height = model.height
        
        # Initialize density grid
        demand_grid = np.zeros((height, width))
        taxi_grid = np.zeros((height, width))
        
        # Collect passenger positions (waiting passengers = demand)
        passengers = [a for a in model.schedule.agents if isinstance(a, Passenger)]
        taxis = [a for a in model.schedule.agents if isinstance(a, Taxi)]
        
        # Count passengers at each grid cell (create demand heatmap)
        for passenger in passengers:
            if passenger.status == "waiting":
                x, y = int(passenger.position[0]), int(passenger.position[1])
                if 0 <= x < width and 0 <= y < height:
                    # Weight by priority: Emergency=3, VIP=2, Regular=1
                    priority_weight = {'emergency': 3, 'vip': 2, 'regular': 1}.get(
                        getattr(passenger, 'priority', 'regular'), 1
                    )
                    demand_grid[y, x] += priority_weight
        
        # Count taxis at each grid cell (for reference)
        for taxi in taxis:
            x, y = int(taxi.position[0]), int(taxi.position[1])
            if 0 <= x < width and 0 <= y < height:
                taxi_grid[y, x] += 1
        
        # Create heatmap using Plotly
        heatmap_fig = go.Figure()
        
        # Add demand heatmap (passenger waiting locations)
        heatmap_fig.add_trace(go.Heatmap(
            z=demand_grid,
            x=list(range(width)),
            y=list(range(height)),
            colorscale='YlOrRd',  # Yellow-Orange-Red (hotter = more demand)
            name='Passenger Demand',
            colorbar=dict(
                title="Demand Intensity"
            ),
            hovertemplate='<b>Location:</b> (%{x}, %{y})<br>' +
                         '<b>Demand Intensity:</b> %{z}<br>' +
                         '<extra></extra>',
            showscale=True
        ))
        
        # Add taxi overlay as scatter points (optional - shows taxi locations)
        taxi_x = [t.position[0] for t in taxis]
        taxi_y = [t.position[1] for t in taxis]
        
        if taxi_x:
            heatmap_fig.add_trace(go.Scatter(
                x=taxi_x,
                y=taxi_y,
                mode='markers',
                marker=dict(
                    symbol='square',
                    size=8,
                    color='cyan',
                    line=dict(width=1, color='darkblue')
                ),
                name='Taxis',
                hovertemplate='<b>Taxi Location:</b> (%{x}, %{y})<extra></extra>',
                showlegend=True
            ))
        
        # Update layout for professional appearance
        heatmap_fig.update_layout(
            template='plotly_white',
            height=500,
            xaxis=dict(
                title="X Position",
                range=[-0.5, width - 0.5],
                scaleanchor="y",
                scaleratio=1,
                showgrid=True,
                gridcolor='lightgray'
            ),
            yaxis=dict(
                title="Y Position",
                range=[-0.5, height - 0.5],
                showgrid=True,
                gridcolor='lightgray'
            ),
            margin=dict(l=60, r=20, t=40, b=50),
            hovermode='closest',
            showlegend=False
        )
        
        # Create prediction heatmap if demand predictor is available
        # Create a new empty figure instead of copying (Plotly figures don't support copy())
        prediction_fig = go.Figure()
        prediction_fig.update_layout(
            template='plotly_white',
            height=500,
            xaxis_title="X Position",
            yaxis_title="Y Position",
            title="No prediction data available"
        )
        
        if hasattr(model, 'demand_predictor') and model.demand_predictor:
            try:
                # Get predictions
                x_positions, y_positions, demand_values = model.demand_predictor.get_demand_heatmap_data(
                    model.schedule.time, model.width, model.height
                )
                
                if x_positions and y_positions:
                    # Create prediction grid
                    pred_grid = np.zeros((model.height, model.width))
                    for x, y, demand in zip(x_positions, y_positions, demand_values):
                        if 0 <= x < model.width and 0 <= y < model.height:
                            # Spread prediction to nearby cells
                            for dx in range(-2, 3):
                                for dy in range(-2, 3):
                                    nx, ny = x + dx, y + dy
                                    if 0 <= nx < model.width and 0 <= ny < model.height:
                                        distance = np.sqrt(dx*dx + dy*dy)
                                        weight = np.exp(-distance) if distance > 0 else 1.0
                                        pred_grid[ny, nx] += demand * weight
                    
                    prediction_fig = go.Figure()
                    prediction_fig.add_trace(go.Heatmap(
                        z=pred_grid,
                        x=list(range(model.width)),
                        y=list(range(model.height)),
                        colorscale='Viridis',  # Different color scale for predictions
                        name='Predicted Demand',
                        colorbar=dict(
                            title="Predicted Intensity"
                        ),
                        hovertemplate='<b>Location:</b> (%{x}, %{y})<br>' +
                                     '<b>Predicted Demand:</b> %{z:.2f}<br>' +
                                     '<extra></extra>',
                        showscale=True
                    ))
                    
                    prediction_fig.update_layout(
                        template='plotly_white',
                        height=500,
                        xaxis=dict(
                            title="X Position",
                            range=[-0.5, model.width - 0.5],
                            scaleanchor="y",
                            scaleratio=1,
                            showgrid=True,
                            gridcolor='lightgray'
                        ),
                        yaxis=dict(
                            title="Y Position",
                            range=[-0.5, model.height - 0.5],
                            showgrid=True,
                            gridcolor='lightgray'
                        ),
                        margin=dict(l=60, r=20, t=40, b=50),
                        hovermode='closest',
                        showlegend=False
                    )
            except Exception as e:
                prediction_fig.update_layout(title=f"Prediction Error: {str(e)}")
        
        return heatmap_fig, prediction_fig
        
    except Exception as e:
        # Return empty figure on error
        empty_fig.update_layout(title=f"Error: {str(e)}")
        return empty_fig, empty_fig

# Advanced Features Callbacks

@app.callback(
    Output('performance-score', 'children'),
    Output('performance-status', 'children'),
    Output('performance-details', 'children'),
    Output('performance-card', 'className'),
    Input('interval-component', 'n_intervals')
)
def update_performance_score(n_intervals):
    """Calculate and display system performance score."""
    global model, data_history
    if model is None or not data_history['steps']:
        return "0", "No Data", "", "card shadow-sm"
    
    try:
        from agents.taxi import Taxi
        from agents.passenger import Passenger
        
        taxis = [a for a in model.schedule.agents if isinstance(a, Taxi)]
        passengers = [a for a in model.schedule.agents if isinstance(a, Passenger)]
        
        # Calculate metrics
        avg_wait = model._calculate_avg_wait_time() if hasattr(model, '_calculate_avg_wait_time') else 0
        utilization = model._calculate_taxi_utilization() if hasattr(model, '_calculate_taxi_utilization') else 0
        waiting = len([p for p in passengers if p.status == "waiting"])
        total_taxis = len(taxis)
        active_taxis = len([t for t in taxis if t.status != "idle"])
        
        # Performance scoring (0-100)
        # Wait time score (lower is better, max 100)
        wait_score = max(0, 100 - (avg_wait * 2)) if avg_wait < 50 else max(0, 100 - avg_wait)
        
        # Utilization score (50-80% is optimal)
        if 50 <= utilization <= 80:
            util_score = 100
        elif utilization < 50:
            util_score = utilization * 2
        else:
            util_score = max(0, 100 - ((utilization - 80) * 2))
        
        # Coverage score (active taxis / total taxis)
        coverage_score = (active_taxis / total_taxis * 100) if total_taxis > 0 else 0
        
        # Waiting passengers penalty
        wait_penalty = min(30, waiting * 2)
        
        # Overall score
        score = (wait_score * 0.4 + util_score * 0.3 + coverage_score * 0.2 - wait_penalty * 0.1)
        score = max(0, min(100, int(score)))
        
        # Status and color
        if score >= 75:
            status = "Excellent"
            color = "success"
            card_class = "card shadow-sm border-success border-3"
        elif score >= 50:
            status = "Good"
            color = "warning"
            card_class = "card shadow-sm border-warning border-3"
        else:
            status = "Needs Improvement"
            color = "danger"
            card_class = "card shadow-sm border-danger border-3"
        
        # Details
        details = dbc.Row([
            dbc.Col([
                html.Small(f"Wait Time: {wait_score:.0f}/100", className="d-block"),
                html.Small(f"Utilization: {util_score:.0f}/100", className="d-block"),
                html.Small(f"Coverage: {coverage_score:.0f}/100", className="d-block")
            ], width=12)
        ])
        
        return str(score), status, details, card_class
    except:
        return "0", "Error", "", "card shadow-sm"

@app.callback(
    Output('alerts-panel', 'children'),
    Output('alert-count', 'children'),
    Input('interval-component', 'n_intervals')
)
def update_alerts(n_intervals):
    """Generate real-time alerts based on system state."""
    global model
    if model is None:
        return [], "0"
    
    try:
        from agents.taxi import Taxi
        from agents.passenger import Passenger
        
        taxis = [a for a in model.schedule.agents if isinstance(a, Taxi)]
        passengers = [a for a in model.schedule.agents if isinstance(a, Passenger)]
        
        alerts = []
        
        # Check for high wait times
        waiting_passengers = [p for p in passengers if p.status == "waiting"]
        for p in waiting_passengers:
            if p.wait_time > 100:
                alerts.append(dbc.Alert([
                    html.I(className="bi bi-exclamation-triangle-fill me-2"),
                    f"Passenger #{p.unique_id} waiting {p.wait_time} steps (High Priority!)"
                ], color="danger", className="mb-2"))
            elif p.wait_time > 50:
                alerts.append(dbc.Alert([
                    html.I(className="bi bi-hourglass-split me-2"),
                    f"Passenger #{p.unique_id} waiting {p.wait_time} steps"
                ], color="warning", className="mb-2"))
        
        # Check for emergency passengers
        emergency = [p for p in waiting_passengers if getattr(p, 'priority', 'regular') == 'emergency']
        if emergency:
            alerts.append(dbc.Alert([
                html.I(className="bi bi-ambulance me-2"),
                f"{len(emergency)} Emergency passenger(s) waiting!"
            ], color="danger", className="mb-2"))
        
        # Check for low utilization
        utilization = model._calculate_taxi_utilization() if hasattr(model, '_calculate_taxi_utilization') else 0
        if utilization < 30:
            alerts.append(dbc.Alert([
                html.I(className="bi bi-graph-down me-2"),
                f"Low taxi utilization: {utilization:.1f}%"
            ], color="warning", className="mb-2"))
        
        # Check for high traffic density
        density = model._calculate_traffic_density() if hasattr(model, '_calculate_traffic_density') else 0
        if density > 0.7:
            alerts.append(dbc.Alert([
                html.I(className="bi bi-traffic-light me-2"),
                f"High traffic density: {density:.2f}"
            ], color="warning", className="mb-2"))
        
        # Check for idle taxis when passengers waiting
        idle_taxis = len([t for t in taxis if t.status == "idle"])
        if idle_taxis > len(taxis) * 0.5 and len(waiting_passengers) > 0:
            alerts.append(dbc.Alert([
                html.I(className="bi bi-info-circle me-2"),
                f"{idle_taxis} taxis idle while {len(waiting_passengers)} passengers waiting"
            ], color="info", className="mb-2"))
        
        if not alerts:
            alerts = [html.P("No alerts - System running smoothly!", className="text-muted text-center mb-0")]
        
        return alerts, str(len([a for a in alerts if isinstance(a, dbc.Alert)]))
    except:
        return [html.P("Error loading alerts", className="text-muted")], "0"

@app.callback(
    Output('statistics-summary', 'children'),
    Input('interval-component', 'n_intervals')
)
def update_statistics_summary(n_intervals):
    """Display comprehensive statistics summary."""
    global model, data_history
    if model is None or not data_history['steps']:
        return html.P("No data available yet", className="text-muted")
    
    try:
        import numpy as np
        from agents.taxi import Taxi
        
        taxis = [a for a in model.schedule.agents if isinstance(a, Taxi)]
        
        # Calculate statistics
        wait_times = data_history['wait_times'] if data_history['wait_times'] else [0]
        utilizations = data_history['utilization'] if data_history['utilization'] else [0]
        densities = data_history['traffic_density'] if data_history['traffic_density'] else [0]
        
        wait_stats = {
            'min': f"{min(wait_times):.1f}s",
            'max': f"{max(wait_times):.1f}s",
            'avg': f"{np.mean(wait_times):.1f}s",
            'std': f"{np.std(wait_times):.1f}s"
        }
        
        util_stats = {
            'min': f"{min(utilizations):.1f}%",
            'max': f"{max(utilizations):.1f}%",
            'avg': f"{np.mean(utilizations):.1f}%",
            'std': f"{np.std(utilizations):.1f}%"
        }
        
        # Revenue stats
        revenues = [getattr(t, 'revenue', 0) for t in taxis]
        total_revenue = sum(revenues)
        avg_revenue = np.mean(revenues) if revenues else 0
        
        return dbc.Row([
            dbc.Col([
                html.H6("Wait Time Statistics", className="text-primary"),
                html.Table([
                    html.Tr([html.Td("Minimum:"), html.Td(wait_stats['min'], className="text-end")]),
                    html.Tr([html.Td("Maximum:"), html.Td(wait_stats['max'], className="text-end")]),
                    html.Tr([html.Td("Average:"), html.Td(wait_stats['avg'], className="text-end")]),
                    html.Tr([html.Td("Std Dev:"), html.Td(wait_stats['std'], className="text-end")])
                ], className="table table-sm")
            ], width=3),
            dbc.Col([
                html.H6("Utilization Statistics", className="text-success"),
                html.Table([
                    html.Tr([html.Td("Minimum:"), html.Td(util_stats['min'], className="text-end")]),
                    html.Tr([html.Td("Maximum:"), html.Td(util_stats['max'], className="text-end")]),
                    html.Tr([html.Td("Average:"), html.Td(util_stats['avg'], className="text-end")]),
                    html.Tr([html.Td("Std Dev:"), html.Td(util_stats['std'], className="text-end")])
                ], className="table table-sm")
            ], width=3),
            dbc.Col([
                html.H6("Revenue Statistics", className="text-info"),
                html.Table([
                    html.Tr([html.Td("Total Revenue:"), html.Td(f"${total_revenue:.2f}", className="text-end")]),
                    html.Tr([html.Td("Avg per Taxi:"), html.Td(f"${avg_revenue:.2f}", className="text-end")]),
                    html.Tr([html.Td("Total Trips:"), html.Td(str(sum(getattr(t, 'total_passengers', 0) for t in taxis)), className="text-end")]),
                    html.Tr([html.Td("Steps Run:"), html.Td(str(len(data_history['steps'])), className="text-end")])
                ], className="table table-sm")
            ], width=3),
            dbc.Col([
                html.H6("System Overview", className="text-warning"),
                html.Table([
                    html.Tr([html.Td("Total Taxis:"), html.Td(str(len(taxis)), className="text-end")]),
                    html.Tr([html.Td("Active Taxis:"), html.Td(str(len([t for t in taxis if t.status != "idle"])), className="text-end")]),
                    html.Tr([html.Td("Avg Density:"), html.Td(f"{np.mean(densities):.2f}", className="text-end")]),
                    html.Tr([html.Td("Data Points:"), html.Td(str(len(data_history['steps'])), className="text-end")])
                ], className="table table-sm")
            ], width=3)
        ])
    except Exception as e:
        return html.P(f"Error: {str(e)}", className="text-danger")

@app.callback(
    Output('analytics-insights', 'children'),
    Input('interval-component', 'n_intervals')
)
def update_analytics_insights(n_intervals):
    """Generate advanced analytics insights."""
    global model, data_history
    if model is None or len(data_history['steps']) < 5:
        return html.P("Collecting data... Need at least 5 data points for insights.", className="text-muted")
    
    try:
        import numpy as np
        from agents.taxi import Taxi
        from agents.passenger import Passenger
        
        taxis = [a for a in model.schedule.agents if isinstance(a, Taxi)]
        passengers = [a for a in model.schedule.agents if isinstance(a, Passenger)]
        
        insights = []
        
        # Trend analysis
        if len(data_history['wait_times']) >= 5:
            recent_wait = np.mean(data_history['wait_times'][-5:])
            earlier_wait = np.mean(data_history['wait_times'][:5])
            if recent_wait < earlier_wait * 0.9:
                insights.append(dbc.Alert([
                    html.I(className="bi bi-arrow-down-circle-fill me-2"),
                    html.Strong("Improving Trend: "),
                    "Wait times decreased by {:.1f}%".format((1 - recent_wait/earlier_wait) * 100)
                ], color="success", className="mb-2"))
            elif recent_wait > earlier_wait * 1.1:
                insights.append(dbc.Alert([
                    html.I(className="bi bi-arrow-up-circle-fill me-2"),
                    html.Strong("Degrading Trend: "),
                    "Wait times increased by {:.1f}%".format((recent_wait/earlier_wait - 1) * 100)
                ], color="warning", className="mb-2"))
        
        # Efficiency insights
        utilization = model._calculate_taxi_utilization() if hasattr(model, '_calculate_taxi_utilization') else 0
        if 50 <= utilization <= 80:
            insights.append(dbc.Alert([
                html.I(className="bi bi-check-circle-fill me-2"),
                html.Strong("Optimal Utilization: "),
                f"Taxi utilization at {utilization:.1f}% is in the optimal range (50-80%)"
            ], color="success", className="mb-2"))
        elif utilization < 30:
            insights.append(dbc.Alert([
                html.I(className="bi bi-exclamation-triangle-fill me-2"),
                html.Strong("Low Efficiency: "),
                f"Only {utilization:.1f}% of taxis are active. Consider reducing fleet size."
            ], color="warning", className="mb-2"))
        
        # Revenue insights
        revenues = [getattr(t, 'revenue', 0) for t in taxis]
        total_revenue = sum(revenues)
        if total_revenue > 0:
            top_taxi = max(taxis, key=lambda t: getattr(t, 'revenue', 0))
            insights.append(dbc.Alert([
                html.I(className="bi bi-trophy-fill me-2"),
                html.Strong("Top Performer: "),
                f"Taxi #{top_taxi.unique_id} generated ${getattr(top_taxi, 'revenue', 0):.2f} revenue"
            ], color="info", className="mb-2"))
        
        # Passenger priority insights
        emergency_waiting = len([p for p in passengers if p.status == "waiting" and getattr(p, 'priority', 'regular') == 'emergency'])
        if emergency_waiting > 0:
            insights.append(dbc.Alert([
                html.I(className="bi bi-ambulance me-2"),
                html.Strong("Priority Alert: "),
                f"{emergency_waiting} emergency passenger(s) currently waiting"
            ], color="danger", className="mb-2"))
        
        if not insights:
            insights = [html.P("System operating normally. No significant insights at this time.", className="text-muted")]
        
        return html.Div(insights)
    except Exception as e:
        return html.P(f"Error generating insights: {str(e)}", className="text-danger")

@app.callback(
    Output('cost-analysis', 'children'),
    Input('interval-component', 'n_intervals')
)
def update_cost_analysis(n_intervals):
    """Display cost analysis and revenue breakdown."""
    global model
    if model is None:
        return html.P("No data available", className="text-muted")
    
    try:
        from agents.taxi import Taxi
        
        taxis = [a for a in model.schedule.agents if isinstance(a, Taxi)]
        
        # Calculate costs and revenue
        total_revenue = sum(getattr(t, 'revenue', 0) for t in taxis)
        total_distance = sum(getattr(t, 'total_distance', 0) for t in taxis)
        total_trips = sum(getattr(t, 'total_passengers', 0) for t in taxis)
        
        # Estimated costs (fuel, maintenance, etc.)
        fuel_cost_per_unit = 0.1  # $0.10 per distance unit
        maintenance_cost_per_trip = 0.5  # $0.50 per trip
        base_cost_per_taxi = 2.0  # $2.00 base cost per taxi
        
        total_fuel_cost = total_distance * fuel_cost_per_unit
        total_maintenance = total_trips * maintenance_cost_per_trip
        total_base_cost = len(taxis) * base_cost_per_taxi
        total_costs = total_fuel_cost + total_maintenance + total_base_cost
        
        profit = total_revenue - total_costs
        profit_margin = (profit / total_revenue * 100) if total_revenue > 0 else 0
        
        # Revenue by taxi type
        revenue_by_type = {}
        for taxi in taxis:
            taxi_type = getattr(taxi, 'taxi_type', 'economy')
            if taxi_type not in revenue_by_type:
                revenue_by_type[taxi_type] = 0
            revenue_by_type[taxi_type] += getattr(taxi, 'revenue', 0)
        
        return dbc.Row([
            dbc.Col([
                html.H6("Financial Summary", className="text-primary mb-3"),
                html.Div([
                    html.Div([
                        html.Small("Total Revenue", className="text-muted d-block"),
                        html.H4(f"${total_revenue:.2f}", className="text-success mb-0")
                    ], className="mb-3"),
                    html.Div([
                        html.Small("Total Costs", className="text-muted d-block"),
                        html.H4(f"${total_costs:.2f}", className="text-danger mb-0")
                    ], className="mb-3"),
                    html.Div([
                        html.Small("Net Profit", className="text-muted d-block"),
                        html.H4(f"${profit:.2f}", className="text-info mb-0"),
                        html.Small(f"Margin: {profit_margin:.1f}%", className="text-muted")
                    ])
                ])
            ], width=6),
            dbc.Col([
                html.H6("Cost Breakdown", className="text-primary mb-3"),
                html.Table([
                    html.Tr([html.Td("Fuel Cost:"), html.Td(f"${total_fuel_cost:.2f}", className="text-end")]),
                    html.Tr([html.Td("Maintenance:"), html.Td(f"${total_maintenance:.2f}", className="text-end")]),
                    html.Tr([html.Td("Base Costs:"), html.Td(f"${total_base_cost:.2f}", className="text-end")]),
                    html.Tr([html.Td(html.Strong("Total:"), className="border-top"), html.Td(html.Strong(f"${total_costs:.2f}"), className="text-end border-top")])
                ], className="table table-sm"),
                html.H6("Revenue by Type", className="text-primary mt-3 mb-2"),
                html.Div([
                    html.Div([
                        html.Small(f"{ttype.title()}: ${rev:.2f}", className="d-block")
                    ]) for ttype, rev in revenue_by_type.items()
                ])
            ], width=6)
        ])
    except Exception as e:
        return html.P(f"Error: {str(e)}", className="text-danger")

@app.callback(
    Output('download-csv', 'data'),
    Input('btn-export-csv', 'n_clicks'),
    prevent_initial_call=True
)
def export_csv(n_clicks):
    """Export simulation data to CSV."""
    global model, data_history
    if n_clicks is None or model is None:
        return None
    
    try:
        # Create DataFrame
        df = pd.DataFrame({
            'Step': data_history['steps'],
            'Wait_Time': data_history['wait_times'],
            'Utilization': data_history['utilization'],
            'Traffic_Density': data_history['traffic_density'],
            'Active_Taxis': data_history['active_taxis'],
            'Waiting_Passengers': data_history['waiting_passengers'],
            'Passengers_Served': data_history['passengers_served']
        })
        
        csv_string = df.to_csv(index=False)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return dict(content=csv_string, filename=f"simulation_data_{timestamp}.csv")
    except:
        return None

@app.callback(
    Output('download-json', 'data'),
    Input('btn-export-json', 'n_clicks'),
    prevent_initial_call=True
)
def export_json(n_clicks):
    """Export simulation data to JSON."""
    global model, data_history
    if n_clicks is None or model is None:
        return None
    
    try:
        from agents.taxi import Taxi
        from agents.passenger import Passenger
        
        taxis = [a for a in model.schedule.agents if isinstance(a, Taxi)]
        passengers = [a for a in model.schedule.agents if isinstance(a, Passenger)]
        
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'simulation_parameters': model_params,
            'current_step': model.schedule.time,
            'metrics_history': data_history,
            'current_state': {
                'total_taxis': len(taxis),
                'active_taxis': len([t for t in taxis if t.status != "idle"]),
                'waiting_passengers': len([p for p in passengers if p.status == "waiting"]),
                'total_served': sum(t.total_passengers for t in taxis),
                'total_revenue': sum(getattr(t, 'revenue', 0) for t in taxis)
            },
            'taxi_details': [
                {
                    'id': t.unique_id,
                    'type': getattr(t, 'taxi_type', 'economy'),
                    'status': t.status,
                    'revenue': getattr(t, 'revenue', 0),
                    'passengers_served': getattr(t, 'total_passengers', 0),
                    'distance': getattr(t, 'total_distance', 0)
                } for t in taxis
            ]
        }
        
        json_string = json.dumps(export_data, indent=2)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return dict(content=json_string, filename=f"simulation_data_{timestamp}.json")
    except:
        return None

# Note: This file is imported by dashboard_main.py
# Do not run app directly - use dashboard_main.py instead

