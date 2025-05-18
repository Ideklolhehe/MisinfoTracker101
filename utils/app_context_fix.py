"""
Fixes for application context issues in CIVILIAN system.
This module provides wrapper functions to ensure proper
Flask application context is maintained across threads.
"""

import logging
import threading
import functools
from typing import Callable, Any, Optional
from flask import Flask, current_app, has_app_context

# Configure logger
logger = logging.getLogger(__name__)

# Global application reference
_flask_app = None

def set_global_app(app: Flask) -> None:
    """Set the global Flask app instance for context management."""
    global _flask_app
    _flask_app = app
    logger.info(f"Set global Flask application: {app.name}")

def get_global_app() -> Optional[Flask]:
    """Get the global Flask app instance."""
    if has_app_context():
        return current_app
    return _flask_app

def with_app_context(func: Callable) -> Callable:
    """
    Decorator to ensure a function runs with proper application context.
    
    Args:
        func: The function to decorate
        
    Returns:
        Decorated function with app context handling
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # If already in app context, just run the function
        if has_app_context():
            return func(*args, **kwargs)
            
        # Get the app instance
        app = get_global_app()
        if app is None:
            logger.error(f"No Flask application available for {func.__name__}")
            raise RuntimeError(
                f"No Flask application available for {func.__name__}. "
                "Make sure set_global_app() is called during initialization."
            )
            
        # Run with app context
        with app.app_context():
            return func(*args, **kwargs)
            
    return wrapper

class AppContextThread(threading.Thread):
    """Thread class that ensures Flask application context is available."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.app = get_global_app()
        if self.app is None:
            logger.error("No Flask application available for thread")
            raise RuntimeError(
                "No Flask application available for thread. "
                "Make sure set_global_app() is called during initialization."
            )
    
    def run(self):
        """Run thread function with application context."""
        with self.app.app_context():
            super().run()

def run_with_app_context(func: Callable, *args: Any, **kwargs: Any) -> Any:
    """
    Run a function with application context.
    
    Args:
        func: Function to run
        *args: Arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        Result of the function
    """
    app = get_global_app()
    if app is None:
        logger.error(f"No Flask application available for {func.__name__}")
        raise RuntimeError(
            f"No Flask application available for {func.__name__}. "
            "Make sure set_global_app() is called during initialization."
        )
        
    with app.app_context():
        return func(*args, **kwargs)

def create_context_thread(target: Callable, *args: Any, **kwargs: Any) -> threading.Thread:
    """
    Create a thread that runs with application context.
    
    Args:
        target: Function to run in the thread
        *args: Arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        Thread instance that will run with app context
    """
    app = get_global_app()
    if app is None:
        logger.error("No Flask application available for thread creation")
        raise RuntimeError(
            "No Flask application available for thread creation. "
            "Make sure set_global_app() is called during initialization."
        )
    
    def run_target_with_context():
        with app.app_context():
            target(*args, **kwargs)
    
    thread = threading.Thread(target=run_target_with_context)
    return thread