"""
Simplified version of the app for Vercel deployment.
This file doesn't initialize all the agents and services that might
cause timeouts in a serverless environment.
"""
import os
from flask import Flask, render_template, jsonify

# Create simplified app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "vercel-deployment-key")

@app.route('/')
def index():
    """Render a simplified home page for Vercel deployment."""
    return render_template('vercel_index.html')

@app.route('/api/health')
def health():
    """Simple health check endpoint."""
    return jsonify({
        "status": "healthy",
        "environment": "vercel",
        "timestamp": os.environ.get("VERCEL_URL", "Not deployed on Vercel")
    })

# Special route for Vercel to check deployment is working
@app.route('/_vercel/now.json')
def vercel_info():
    """Return information about the Vercel deployment."""
    return jsonify({
        "version": 2,
        "public": True,
        "functions": {"api/*.py": {"memory": 1024}}
    })

# For local development
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)