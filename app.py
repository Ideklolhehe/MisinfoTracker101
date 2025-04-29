import os
import logging
import uuid
from urllib.parse import urlencode

from flask import Flask, g, session, redirect, render_template, request, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, current_user, login_user, logout_user
from sqlalchemy.orm import DeclarativeBase
from werkzeug.middleware.proxy_fix import ProxyFix
from flask_dance.consumer import OAuth2ConsumerBlueprint, oauth_authorized, oauth_error
from flask_dance.consumer.storage import BaseStorage
from sqlalchemy.exc import NoResultFound
from werkzeug.local import LocalProxy
import jwt

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Database setup
class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)

# Create the app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "default-dev-key")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Configure database
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", "sqlite:///civilian.db")
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Initialize database
db.init_app(app)

# Initialize Flask-Login
login_manager = LoginManager(app)
login_manager.login_view = "replit_auth.login"

@login_manager.user_loader
def load_user(user_id):
    from models import User
    return User.query.get(user_id)

# Import and initialize routes after app is initialized
with app.app_context():
    # Import models
    import models  # noqa: F401
    
    # Import replit auth
    from replit_auth import make_replit_blueprint, require_login
    
    # Register auth blueprint
    app.register_blueprint(make_replit_blueprint(), url_prefix="/auth")
    
    # Make session permanent
    @app.before_request
    def make_session_permanent():
        session.permanent = True
    
    # Import route blueprints
    from routes.dashboard import dashboard_bp
    from routes.api import api_bp
    from routes.data_sources import data_sources_bp
    from routes.adversarial import adversarial_bp
    from routes.verification import verification_bp
    from routes.home import home_bp
    from routes.profile import profile_bp
    
    # Import new route blueprints
    from routes.api_credentials import api_credentials_bp
    from routes.rss_feeds import rss_feeds_bp
    from routes.agents import agents_bp
    from routes.dev_auth import dev_auth_bp  # Development-only
    from routes.evidence import evidence_bp
    from routes.complexity import complexity_bp
    
    # Import advanced analysis blueprints
    from routes.time_series import time_series_bp
    from routes.alerts import alerts_bp
    from routes.clusters import clusters_bp
    from routes.prediction import prediction_bp
    from routes.network import network_bp
    
    # Import counter-narrative and comparative analysis blueprints
    from routes.counter_narrative import counter_narrative_bp
    from routes.comparative import comparative_bp
    
    # Import web scraping blueprint
    from routes.web_scraping import web_scraping_bp
    
    # Register route blueprints
    app.register_blueprint(dashboard_bp, url_prefix='/dashboard')
    app.register_blueprint(api_bp, url_prefix='/api')
    app.register_blueprint(data_sources_bp, url_prefix='/data-sources')
    app.register_blueprint(adversarial_bp, url_prefix='/adversarial')
    app.register_blueprint(verification_bp, url_prefix='/verify')
    app.register_blueprint(api_credentials_bp, url_prefix='/api-credentials')
    app.register_blueprint(rss_feeds_bp, url_prefix='/rss-feeds')
    app.register_blueprint(evidence_bp, url_prefix='/evidence')
    app.register_blueprint(complexity_bp)  # No URL prefix needed since routes include '/complexity'
    app.register_blueprint(agents_bp)
    app.register_blueprint(profile_bp)
    app.register_blueprint(home_bp)
    app.register_blueprint(dev_auth_bp)  # Development-only
    
    # Register advanced analysis blueprints
    app.register_blueprint(time_series_bp)  # Routes include '/time_series'
    app.register_blueprint(alerts_bp)  # Routes include '/alerts'
    app.register_blueprint(clusters_bp)  # Routes include '/clusters'
    app.register_blueprint(prediction_bp)  # Routes include '/prediction'
    app.register_blueprint(network_bp)  # Routes include '/network'
    
    # Register counter-narrative and comparative analysis blueprints
    app.register_blueprint(counter_narrative_bp)  # Routes include '/counter-narrative'
    app.register_blueprint(comparative_bp)  # Routes include '/comparative'
    
    # Register web scraping blueprint
    app.register_blueprint(web_scraping_bp)  # Routes include '/web-scraping'
    
    # Initialize external API clients
    from services.api_credential_manager import APICredentialManager
    from services.external_api_initializer import ExternalAPIInitializer
    
    # Check API credentials and initialize clients
    api_status = APICredentialManager.get_all_credential_status()
    for api_name, available in api_status.items():
        if available:
            logger.info(f"{api_name.capitalize()} API credentials are available")
        else:
            logger.warning(f"{api_name.capitalize()} API credentials are missing")
    
    # Initialize API clients
    app.config['API_CLIENTS'] = {}
    
    # Try to initialize OpenAI client (required for verification service)
    if api_status.get('openai', False):
        app.config['API_CLIENTS']['openai'] = ExternalAPIInitializer.init_openai_client()
        if app.config['API_CLIENTS']['openai']:
            logger.info("OpenAI API client initialized successfully")
        else:
            logger.error("Failed to initialize OpenAI API client")
    
    # Try to initialize YouTube client
    if api_status.get('youtube', False):
        app.config['API_CLIENTS']['youtube'] = ExternalAPIInitializer.init_youtube_client()
        if app.config['API_CLIENTS']['youtube']:
            logger.info("YouTube API client initialized successfully")
        else:
            logger.error("Failed to initialize YouTube API client")
    
    # Try to initialize Twitter client
    if api_status.get('twitter', False):
        app.config['API_CLIENTS']['twitter'] = ExternalAPIInitializer.init_twitter_client()
        if app.config['API_CLIENTS']['twitter']:
            logger.info("Twitter API client initialized successfully")
        else:
            logger.error("Failed to initialize Twitter API client")
    
    # Try to initialize Telegram client
    if api_status.get('telegram', False):
        app.config['API_CLIENTS']['telegram'] = ExternalAPIInitializer.init_telegram_client()
        if app.config['API_CLIENTS']['telegram']:
            logger.info("Telegram API client initialized successfully")
        else:
            logger.error("Failed to initialize Telegram API client")
    
    # Try to initialize Tor client for dark web monitoring
    if api_status.get('dark_web', False):
        app.config['API_CLIENTS']['tor'] = ExternalAPIInitializer.init_tor_client()
        if app.config['API_CLIENTS']['tor']:
            logger.info("Tor client initialized successfully")
        else:
            logger.error("Failed to initialize Tor client")
    
    # Create database tables
    db.create_all()

    @app.errorhandler(403)
    def forbidden(e):
        return render_template("403.html"), 403

    @app.errorhandler(404)
    def not_found(e):
        return render_template("404.html"), 404

    @app.errorhandler(500)
    def server_error(e):
        return render_template("500.html"), 500
    
    logger.info("CIVILIAN application initialized")
