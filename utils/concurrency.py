"""
Concurrency utilities for CIVILIAN.

This module provides tools to help with concurrency, such as running functions in 
separate threads or processes.
"""

import threading
import functools
from typing import Callable, Any

def run_in_thread(func: Callable) -> Callable:
    """
    Decorator that runs the decorated function in a separate thread.
    
    Args:
        func: The function to run in a thread
        
    Returns:
        Wrapper function that starts a thread
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> threading.Thread:
        thread = threading.Thread(target=func, args=args, kwargs=kwargs)
        thread.daemon = True  # Daemon threads are killed when the main program exits
        thread.start()
        return thread
    return wrapper