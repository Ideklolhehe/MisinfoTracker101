# CIVILIAN System Export Guide

This guide provides a comprehensive overview of all essential files needed to deploy the CIVILIAN system independently. Use this as a checklist to ensure all necessary components are included in your deployment.

## Core Application Files

### Primary Files
- `main.py` - Main application entry point
- `app.py` - Flask application and database configuration
- `models.py` - Database schema and models
- `config.py` - Configuration settings and environment variables
- `replit_auth.py` - Authentication system
- `wsgi.py` - WSGI server configuration

### Initialization Scripts
- `init_open_sources.py` - Configures initial data sources
- `recreate_db.py` - Database recreation utility (use with caution)

## Core Directories and Components

### Agents
The multi-agent system that powers CIVILIAN:
- `agents/__init__.py`
- `agents/agent_factory.py` - Factory pattern for agent creation
- `agents/base_agent.py` - Base agent class
- `agents/analyzer_agent.py` - Analyzes narratives
- `agents/detector_agent.py` - Detects misinformation
- `agents/counter_agent.py` - Generates counter-narratives
- `agents/multi_agent_coordinator.py` - Coordinates agent interactions

### Data Sources
Components for ingesting various data sources:
- `data_sources/__init__.py`
- `data_sources/source_base.py` - Base data source class
- `data_sources/rss_source.py` - RSS feed monitoring
- `data_sources/web_source_manager.py` - Web scraping manager
- `data_sources/darkweb_source.py` - Dark web monitoring
- `data_sources/telegram_source.py` - Telegram monitoring
- `data_sources/twitter_source.py` - Twitter monitoring
- `data_sources/youtube_source.py` - YouTube monitoring

### Services
Core services that provide functionality to the agents:
- `services/__init__.py`
- `services/narrative_network.py` - Network analysis of narratives
- `services/complexity_analyzer.py` - Complexity analysis
- `services/complexity_scheduler.py` - Scheduler for analysis tasks
- `services/counter_narrative_service.py` - Counter-narrative generation
- `services/ipfs_service.py` - IPFS integration
- `services/predictive_modeling.py` - Predictive modeling
- `services/time_series_analyzer.py` - Time series analysis
- `services/verification_service.py` - Content verification
- `services/web_scraping_service.py` - Web scraping
- `services/alert_system.py` - Alert notifications
- `services/comparative_analysis_service.py` - Comparative analysis
- `services/decentralized_publishing.py` - Decentralized publishing
- `services/export_service.py` - Data export functionality

### Routes
Web routes and API endpoints:
- `routes/__init__.py`
- `routes/dashboard.py` - Main dashboard
- `routes/api.py` - API endpoints
- `routes/home.py` - Home page
- `routes/adversarial.py` - Adversarial testing
- `routes/agents.py` - Agent management
- `routes/alerts.py` - Alerts and notifications
- `routes/clusters.py` - Narrative clustering
- `routes/comparative.py` - Comparative analysis
- `routes/complexity.py` - Complexity analysis
- `routes/counter_narrative.py` - Counter-narrative management
- `routes/data_sources.py` - Data source management
- `routes/decentralized_publishing.py` - Decentralized publishing
- `routes/evidence.py` - Evidence management
- `routes/network.py` - Network visualization
- `routes/prediction.py` - Prediction visualization
- `routes/time_series.py` - Time series visualization
- `routes/verification.py` - Content verification
- `routes/web_scraping.py` - Web scraping interface

### Utils
Utility functions used throughout the system:
- `utils/__init__.py`
- `utils/text_processor.py` - Text processing utilities
- `utils/feed_parser.py` - Feed parsing utilities
- `utils/stream_processor.py` - Stream processing
- `utils/clustering.py` - Clustering algorithms
- `utils/time_series.py` - Time series utilities
- `utils/vector_store.py` - Vector database interface
- `utils/vector_search.py` - Vector similarity search
- `utils/ai_processor.py` - AI processing utilities
- `utils/web_scraper.py` - Web scraping utilities
- `utils/secleds.py` - SeCLEDS algorithm implementation
- `utils/metrics.py` - Metrics calculation
- `utils/data_scaling.py` - Data scaling utilities
- `utils/encryption.py` - Encryption utilities
- `utils/sms_service.py` - SMS notification service
- `utils/app_context.py` - Flask application context utilities
- `utils/app_context_fix.py` - Application context fixes

### Storage
Storage-related components:
- `storage/__init__.py`
- `storage/evidence_store.py` - Evidence storage
- `storage/graph_store.py` - Graph database interface
- `storage/ipfs_evidence_store.py` - IPFS evidence storage

## Templates and Static Assets
- `templates/` - All HTML templates
  - `templates/layout.html` - Base layout
  - `templates/dashboard/` - Dashboard templates
  - `templates/network/` - Network visualization templates
  - `templates/counter/` - Counter-narrative templates
  - Various other template subdirectories
- `static/` - Static assets
  - `static/css/` - CSS stylesheets
  - `static/js/` - JavaScript files
  - `static/img/` - Images

## Configuration Files
- `config/` - Configuration directory
  - `config/open_news_sources.json` - Open news source configuration
  - `config/fact_check_sources.json` - Fact-checking source configuration
  - `config/open_data_sources.json` - Open data source configuration
  - `config/specialized_monitoring.json` - Specialized monitoring configuration
  - `config/advanced_data_sources.json` - Advanced data source configuration

## Deployment Files
- `.env.example` - Example environment variables
- `deployment_requirements.txt` - Python dependencies
- `docker-compose.yml` - Docker Compose configuration
- `Dockerfile` - Docker configuration

## Required Directories (empty in a new deployment)
- `evidence/` - For evidence storage
- `graph_exports/` - For network analysis exports
- `storage/web_cache/` - For web scraping cache

## Database Schema

### Primary Models:
1. `User` - User authentication and profiles
2. `DataSource` - Information sources
3. `DetectedNarrative` - Misinformation narratives
4. `NarrativeInstance` - Individual occurrences of narratives
5. `BeliefNode` and `BeliefEdge` - Graph representation
6. `CounterMessage` - Counter-narratives
7. `EvidenceRecord` - Immutable evidence records
8. `MisinformationEvent` - Source reliability tracking
9. `PublishedContent` - Decentralized publishing records

## Environment Variables
Required environment variables:
```
# Database
DATABASE_URL

# Security
SESSION_SECRET

# API Keys
OPENAI_API_KEY

# Notifications
TWILIO_ACCOUNT_SID
TWILIO_AUTH_TOKEN
TWILIO_PHONE_NUMBER
RECIPIENT_PHONE_NUMBER
```

## Installation and Deployment
1. Create a Python environment
2. Install dependencies: `pip install -r deployment_requirements.txt`
3. Configure environment variables
4. Initialize database: `python main.py`
5. Run the application:
   - Development: `python main.py`
   - Production: `gunicorn --bind 0.0.0.0:5000 --workers=4 --timeout=120 --reuse-port --reload main:app`

## Containerized Deployment
1. Build the Docker image: `docker build -t civilian .`
2. Run with Docker Compose: `docker-compose up -d`

## Important Notes
- The system requires a PostgreSQL database
- IPFS should be available for evidence storage
- API keys should be securely stored as environment variables
- The system is designed to run continuously, with multiple agents operating asynchronously