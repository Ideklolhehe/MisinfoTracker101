from vercel_app import app

# This file is used by Vercel to find the Flask application
# We're using the simplified vercel_app instead of the full main app
# to avoid initialization timeouts in the serverless environment
if __name__ == "__main__":
    app.run()