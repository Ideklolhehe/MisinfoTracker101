"""
Ultra-minimal version of the app for Vercel deployment.
This is the simplest possible Flask app that can be deployed to Vercel.
"""
from flask import Flask, render_template

# Create ultra-simplified app
app = Flask(__name__)

@app.route('/')
def index():
    """Return a simple HTML response."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>CIVILIAN - Speed Insights</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <!-- Vercel Speed Insights -->
        <script defer src="https://cdn.vercel-insights.com/v1/speed-insights/script.js"></script>
    </head>
    <body style="font-family: sans-serif; max-width: 600px; margin: 0 auto; padding: 20px; text-align: center;">
        <h1>CIVILIAN</h1>
        <p>Sovereign Machine Intelligence for Information Analysis</p>
        <p>Speed Insights is now active.</p>
        <p style="margin-top: 40px; color: #666;">The full system is running on Replit.</p>
    </body>
    </html>
    """

# For local development
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)