"""
Main entry point for Professional Dashboard using Plotly Dash
Deployment-ready version with environment variable support
"""

import os
from dashboard.app import app

# Get port from environment variable (for cloud deployment) or use default
PORT = int(os.environ.get('PORT', 8050))
HOST = os.environ.get('HOST', '0.0.0.0')

# Expose server for deployment platforms that need it
server = app.server

if __name__ == '__main__':
    print("=" * 60)
    print("=" * 60)
    print(f"Starting dashboard server on {HOST}:{PORT}...")
    
    print("=" * 60)
    print("Press Ctrl+C to stop the server")
    print("=" * 60)
    
    # Run the app
    app.run(debug=False, port=PORT, host=HOST)

