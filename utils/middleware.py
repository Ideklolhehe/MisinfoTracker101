"""
Middleware components for the CIVILIAN system.
Provides secure CORS configuration and other Flask middleware.
"""

import logging
from typing import List, Dict, Any, Optional

from flask import Flask, request, g
from flask_cors import CORS
import time

from utils.environment import CORS_ORIGINS, API_RATE_LIMIT
from utils.metrics import MetricsMiddleware, api_counter, api_latency_summary

# Configure module logger
logger = logging.getLogger(__name__)


def configure_cors(app: Flask) -> None:
    """
    Configure CORS for the Flask application.
    
    Args:
        app: Flask application
    """
    logger.info(f"Configuring CORS with origins: {CORS_ORIGINS}")
    
    # Configure CORS based on allowed origins
    resources = {
        r"/api/*": {"origins": CORS_ORIGINS},
        r"/auth/*": {"origins": CORS_ORIGINS},
    }
    
    # For open endpoints like healthcheck
    if "*" not in CORS_ORIGINS:
        resources[r"/health"] = {"origins": "*"}
        resources[r"/metrics"] = {"origins": "*"}
        
    CORS(app, resources=resources, supports_credentials=True)
    
    logger.info("CORS configured successfully")


class RequestMiddleware:
    """
    Middleware for request processing and logging.
    """
    
    def __init__(self, app: Flask):
        """
        Initialize the middleware.
        
        Args:
            app: Flask application
        """
        self.app = app
        self._init_before_request(app)
        self._init_after_request(app)
        self._init_teardown_request(app)
        
        logger.info("Request middleware initialized")
        
    def _init_before_request(self, app: Flask) -> None:
        """
        Configure before-request handlers.
        
        Args:
            app: Flask application
        """
        @app.before_request
        def set_request_start_time() -> None:
            """Store the request start time."""
            g.start_time = time.time()
            g.request_id = request.headers.get("X-Request-ID", f"req-{time.time()}")
            
        @app.before_request
        def log_request() -> None:
            """Log basic request information."""
            logger.debug(f"Request: {request.method} {request.path} [{g.request_id}]")
            
    def _init_after_request(self, app: Flask) -> None:
        """
        Configure after-request handlers.
        
        Args:
            app: Flask application
        """
        @app.after_request
        def add_security_headers(response):
            """Add security headers to responses."""
            # Security headers
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "SAMEORIGIN"
            response.headers["X-XSS-Protection"] = "1; mode=block"
            response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
            
            # Content Security Policy
            csp = "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline';"
            csp += " img-src 'self' data:; font-src 'self' data:; connect-src 'self';"
            response.headers["Content-Security-Policy"] = csp
            
            # CORS headers are managed by flask-cors
            return response
            
        @app.after_request
        def log_response(response):
            """Log response information and timing."""
            duration = time.time() - g.get("start_time", time.time())
            status_code = response.status_code
            
            # Record metrics
            endpoint = request.endpoint or request.path
            api_counter.labels(
                endpoint=endpoint,
                method=request.method,
                status=status_code
            ).inc()
            
            api_latency_summary.labels(endpoint=endpoint).observe(duration)
            
            logger.debug(
                f"Response: {request.method} {request.path} - "
                f"{status_code} ({duration:.3f}s) [{g.request_id}]"
            )
            
            return response
            
    def _init_teardown_request(self, app: Flask) -> None:
        """
        Configure teardown-request handlers.
        
        Args:
            app: Flask application
        """
        @app.teardown_request
        def log_exception(exception=None):
            """Log exceptions during request handling."""
            if exception:
                logger.exception(
                    f"Exception during request: {request.method} {request.path} "
                    f"[{g.get('request_id', 'unknown')}]"
                )


def configure_middleware(app: Flask) -> None:
    """
    Configure all middleware for the Flask application.
    
    Args:
        app: Flask application
    """
    # Configure security middleware
    configure_cors(app)
    
    # Configure request middleware
    RequestMiddleware(app)
    
    # Configure metrics middleware
    MetricsMiddleware(app)
    
    logger.info("All middleware configured successfully")