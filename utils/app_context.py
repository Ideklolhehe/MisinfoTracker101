"""
Application context helpers for CIVILIAN background threads.

These utilities help ensure Flask application context is properly propagated
to background threads that need database access.
"""

import functools
import threading
from flask import current_app, has_app_context
from app import app

_app_context_local = threading.local()

def has_thread_app_context():
    """Check if the current thread has an application context."""
    return hasattr(_app_context_local, 'context') and _app_context_local.context is not None

def get_thread_app_context():
    """Get the application context for the current thread."""
    if has_thread_app_context():
        return _app_context_local.context
    return None

def create_thread_app_context():
    """Create a new application context for the current thread."""
    if not has_thread_app_context():
        _app_context_local.context = app.app_context()
        _app_context_local.context.push()
    return _app_context_local.context

def destroy_thread_app_context():
    """Destroy the application context for the current thread."""
    if has_thread_app_context():
        _app_context_local.context.pop()
        _app_context_local.context = None

def ensure_app_context(func):
    """Decorator to ensure a function has an application context.
    
    This decorator checks if there's already an app context and creates one
    if needed. It's designed to work with both regular functions and methods.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Check if already in app context
        need_context = not has_app_context() and not has_thread_app_context()
        
        if need_context:
            # Create new app context
            ctx = create_thread_app_context()
            try:
                # Run function in context
                return func(*args, **kwargs)
            finally:
                # Clean up
                destroy_thread_app_context()
        else:
            # Already has context, just run function
            return func(*args, **kwargs)
    
    return wrapper

def with_app_context(func):
    """Legacy decorator name for backward compatibility."""
    return ensure_app_context(func)