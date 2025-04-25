import os

class Config:
    """Base configuration for the CIVILIAN application."""
    # Application settings
    DEBUG = os.environ.get('DEBUG', 'True').lower() == 'true'
    TESTING = False
    SECRET_KEY = os.environ.get('SESSION_SECRET', 'default-dev-key')
    
    # Database settings
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL', 'sqlite:///civilian.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # NLP settings
    SUPPORTED_LANGUAGES = ['en', 'es']  # English and Spanish to start
    DETECTION_THRESHOLD = 0.75  # Confidence threshold for misinformation detection
    
    # Redis settings for caching and message queue
    REDIS_URL = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
    
    # API rate limits
    TWITTER_API_RATE_LIMIT = 60  # Requests per 15-min window
    TELEGRAM_API_RATE_LIMIT = 30  # Requests per second
    
    # Ingestion settings
    INGESTION_INTERVAL = 120  # Seconds between ingestion cycles
    MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10MB max upload size
    
    # Vector database settings
    VECTOR_DIMENSION = 768  # BERT embedding dimension
    SIMILARITY_THRESHOLD = 0.85  # Threshold for similarity matching
    
    # Agent settings
    DETECTOR_REFRESH_INTERVAL = 300  # Seconds between model refreshes
    ANALYZER_BATCH_SIZE = 100  # Number of items to analyze in a batch
    
    # Logging settings
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'DEBUG')
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    SQLALCHEMY_TRACK_MODIFICATIONS = True

class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    TESTING = False
    
    # In production, use stronger security settings
    SESSION_COOKIE_SECURE = True
    REMEMBER_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True

class TestingConfig(Config):
    """Testing configuration."""
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'

# Active configuration
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}

active_config = config[os.environ.get('FLASK_ENV', 'default')]
