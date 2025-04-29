"""
Application context utilities for the CIVILIAN system.
Provides decorators and context managers for Flask application context.
"""

import logging
import functools
import threading
from typing import Callable, Any, Optional, TypeVar, cast

from flask import Flask, current_app, has_app_context, has_request_context

# Configure module logger
logger = logging.getLogger(__name__)

# Current application reference
_app_instance: Optional[Flask] = None

# Type variables for function signatures
F = TypeVar('F', bound=Callable[..., Any])
T = TypeVar('T')


def ensure_app_context(func_or_app=None):
    """
    Decorator to ensure a function runs within an application context.
    Can be used with or without an app parameter.
    
    Args:
        func_or_app: Function to decorate or Flask app instance
        
    Returns:
        Decorated function or decorator function
    """
    # If used as @ensure_app_context(app)
    if isinstance(func_or_app, Flask):
        app = func_or_app
        
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if has_app_context():
                    return func(*args, **kwargs)
                with app.app_context():
                    return func(*args, **kwargs)
            return wrapper
        return decorator
    
    # If used as @ensure_app_context
    if callable(func_or_app):
        func = func_or_app
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if has_app_context():
                return func(*args, **kwargs)
            
            app = get_current_app()
            if app is None:
                logger.error(f"No application context available for {func.__name__}")
                raise RuntimeError(
                    f"No application context available for {func.__name__}. "
                    "Make sure set_current_app() was called."
                )
                
            with app.app_context():
                return func(*args, **kwargs)
                
        return wrapper
    
    # If used without arguments @ensure_app_context()
    return lambda func: ensure_app_context(func)


def set_current_app(app: Flask) -> None:
    """
    Set the current application instance.
    
    Args:
        app: Flask application instance
    """
    global _app_instance
    _app_instance = app
    logger.info(f"Set current application instance: {app.name}")


def get_current_app() -> Optional[Flask]:
    """
    Get the current application instance.
    
    Returns:
        Flask application instance or None
    """
    if has_app_context():
        return current_app
    return _app_instance


def with_app_context(func: F) -> F:
    """
    Decorator to ensure a function runs within an application context.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if has_app_context():
            # Already in app context
            return func(*args, **kwargs)
            
        app = get_current_app()
        if app is None:
            logger.error(f"No application context available for {func.__name__}")
            raise RuntimeError(
                f"No application context available for {func.__name__}. "
                "Make sure set_current_app() was called."
            )
            
        with app.app_context():
            return func(*args, **kwargs)
            
    return cast(F, wrapper)


class AppContextThread(threading.Thread):
    """
    Thread that runs with Flask application context.
    """
    
    def __init__(self, *args: Any, app: Optional[Flask] = None, **kwargs: Any):
        """
        Initialize the thread.
        
        Args:
            *args: Thread arguments
            app: Flask application instance
            **kwargs: Thread keyword arguments
        """
        super().__init__(*args, **kwargs)
        self.app = app or get_current_app()
        if self.app is None:
            raise ValueError(
                "No Flask application instance available. "
                "Make sure set_current_app() was called or provide app parameter."
            )
            
    def run(self) -> None:
        """Run the thread with application context."""
        with self.app.app_context():
            super().run()


def with_app_context_async(func: Callable[..., T]) -> Callable[..., threading.Thread]:
    """
    Run a function asynchronously in its own thread with application context.
    
    Args:
        func: Function to run
        
    Returns:
        Function that returns a thread
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> threading.Thread:
        app = get_current_app()
        if app is None:
            logger.error(f"No application context available for {func.__name__}")
            raise RuntimeError(
                f"No application context available for {func.__name__}. "
                "Make sure set_current_app() was called."
            )
            
        def run_with_context() -> None:
            with app.app_context():
                func(*args, **kwargs)
                
        thread = threading.Thread(target=run_with_context)
        thread.daemon = True
        thread.start()
        return thread
        
    return wrapper


def get_app_context():
    """
    Context manager for Flask application context.
    
    Usage:
        with get_app_context():
            # Code that requires application context
            
    Returns:
        Flask application context
    """
    app = get_current_app()
    if app is None:
        logger.error("No application context available")
        raise RuntimeError(
            "No application context available. "
            "Make sure set_current_app() was called."
        )
    
    return app.app_context()