# CIVILIAN Deployment Guide

This guide provides instructions for deploying the complete CIVILIAN system independently.

## Required Files

The CIVILIAN system consists of the following core components:

### Core Application Files
- `main.py` - Main application entry point
- `app.py` - Flask application setup
- `models.py` - Database models
- `config.py` - Configuration settings
- `replit_auth.py` - Authentication system

### Directory Structure
- `agents/` - AI agent modules
- `api/` - API endpoints
- `data_sources/` - Data source modules
- `evidence/` - Evidence storage modules
- `routes/` - Web routes
- `scripts/` - Utility scripts
- `services/` - Core services
- `static/` - Static assets (CSS, JS, images)
- `templates/` - HTML templates
- `utils/` - Utility functions

### Deployment Configuration
- `requirements.txt` - Python dependencies
- `wsgi.py` - WSGI server configuration
- `vercel.json` - (Optional) Vercel deployment configuration

## Deployment Steps

1. **Database Setup**
   - Set up a PostgreSQL database
   - Configure the connection via the `DATABASE_URL` environment variable

2. **Environment Variables**
   - Set the following essential variables:
     - `DATABASE_URL` - PostgreSQL connection string
     - `SESSION_SECRET` - Secret key for session management
     - `OPENAI_API_KEY` - For AI analysis
     - `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_PHONE_NUMBER` - For SMS alerts

3. **Installation**
   - Install dependencies: `pip install -r requirements.txt`

4. **Database Initialization**
   - Run `python main.py` to initialize the database schema

5. **Running the Application**
   - Development: `python main.py`
   - Production: Use Gunicorn with `gunicorn --bind 0.0.0.0:5000 --reuse-port --reload main:app`

## Production Deployment Options

### Option 1: Dedicated Server/VPS
1. Clone the repository
2. Set up a reverse proxy (Nginx/Apache)
3. Configure Gunicorn as a service
4. Set up PostgreSQL database

### Option 2: Container-based Deployment
1. Use the included Dockerfile
2. Build the container: `docker build -t civilian .`
3. Run with proper environment variables

### Option 3: Vercel Deployment
For Vercel deployment of the full system, extensive optimization would be needed due to serverless constraints.

## Monitoring and Maintenance

1. Configure logging to track system performance
2. Set up regular database backups
3. Monitor agent performance and narrative analysis

## Security Considerations

1. Keep API keys secure and rotate them regularly
2. Use HTTPS for all connections
3. Consider IP whitelisting for admin access
4. Enable database encryption for sensitive data

---

For issues or questions, please contact the system administrator.