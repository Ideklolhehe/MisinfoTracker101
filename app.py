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
    
    # Register route blueprints
    app.register_blueprint(dashboard_bp, url_prefix='/dashboard')
    app.register_blueprint(api_bp, url_prefix='/api')
    app.register_blueprint(data_sources_bp, url_prefix='/data-sources')
    app.register_blueprint(adversarial_bp, url_prefix='/adversarial')
    app.register_blueprint(verification_bp, url_prefix='/verify')
    app.register_blueprint(profile_bp)
    app.register_blueprint(home_bp)
    
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
