# CIVILIAN: Complete Deployment Package

## System Overview
CIVILIAN is an advanced AI-powered platform for comprehensive misinformation detection and counteraction. It leverages multi-agent intelligence to analyze and map global information manipulation networks with sophisticated technological strategies.

## Core Technologies
- Python-based multi-agent AI system
- NetworkX for advanced network analysis
- Machine learning misinformation detection
- Real-time data processing and narrative mapping
- Advanced natural language processing
- Modular Flask-based scalable architecture
- Temporal clustering and visualization technologies

## Complete Deployment Instructions

### Step 1: Clone the Repository
Clone the entire repository to your deployment environment.

### Step 2: Configure Environment Variables
Create a `.env` file or set up environment variables with the following essential configurations:

```
# Database Configuration
DATABASE_URL=postgresql://username:password@hostname:port/database

# Security
SESSION_SECRET=your-secure-session-key

# API Keys
OPENAI_API_KEY=your-openai-api-key

# Twilio Configuration (for SMS notifications)
TWILIO_ACCOUNT_SID=your-twilio-sid
TWILIO_AUTH_TOKEN=your-twilio-token
TWILIO_PHONE_NUMBER=your-twilio-phone-number
RECIPIENT_PHONE_NUMBER=target-phone-number

# Optional Configurations
DEBUG=False
LOG_LEVEL=INFO
DETECTION_THRESHOLD=0.75
SIMILARITY_THRESHOLD=0.85
```

### Step 3: Set Up the Database
1. Create a PostgreSQL database
2. The system will automatically initialize the schema on first run

### Step 4: Install Dependencies
Install all required Python packages:

```bash
pip install -r requirements.txt
```

### Step 5: Initialize the System
Run the initialization script to set up initial data sources and configurations:

```bash
python main.py
```

### Step 6: Production Deployment
For production deployment, use Gunicorn as the WSGI server:

```bash
gunicorn --bind 0.0.0.0:5000 --workers=4 --timeout=120 --reuse-port --reload main:app
```

### System Architecture

The core components of the CIVILIAN system include:

1. **Multi-Agent System**
   - Located in the `agents/` directory
   - Includes analyzer, detector, and counter-narrative generation agents

2. **Data Sources**
   - Located in the `data_sources/` directory
   - Handles various information sources including RSS feeds and web scraping

3. **Services**
   - Located in the `services/` directory
   - Core services like narrative network analysis, complexity analysis, and predictive modeling

4. **Routes**
   - Located in the `routes/` directory
   - Web routes for dashboards, API endpoints, and user interfaces

5. **Evidence Storage**
   - Located in the `evidence/` directory
   - IPFS-based immutable evidence storage for narrative tracking

6. **Templates and Static Assets**
   - Located in the `templates/` and `static/` directories
   - User interface components and styling

### Security Considerations

1. **API Key Protection**: Ensure all API keys are securely stored as environment variables
2. **Database Security**: Use strong passwords and consider encryption for sensitive data
3. **Access Control**: Implement proper authentication and authorization
4. **Updates**: Regularly update dependencies to patch security vulnerabilities

### Monitoring and Maintenance

1. Monitor logs for errors and system performance
2. Set up regular database backups
3. Track resource usage and scale as needed
4. Regularly check information sources for availability and accuracy

### Troubleshooting

If you encounter issues during deployment:

1. Check database connectivity and credentials
2. Verify environment variables are correctly set
3. Review logs for specific error messages
4. Ensure dependencies are correctly installed
5. Check file permissions for necessary directories

For detailed API documentation and development information, refer to the code documentation and comments throughout the codebase.

---

© CIVILIAN - Sovereign Machine Intelligence for Information Analysis