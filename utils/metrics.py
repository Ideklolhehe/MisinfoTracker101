"""
Metrics utilities for recording and tracking system performance.
This module provides decorators and functions to measure and record
various metrics about the CIVILIAN system.
"""

import logging
import time
import functools
from typing import Callable, Dict, Any, Optional

# Initialize logger
logger = logging.getLogger(__name__)

class Gauge:
    """A simple gauge metric that tracks a single value."""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.value = 0.0
        
    def set(self, value: float) -> None:
        """Set the gauge to a specific value."""
        self.value = value
        
    def inc(self, amount: float = 1.0) -> None:
        """Increment the gauge by the given amount."""
        self.value += amount
        
    def dec(self, amount: float = 1.0) -> None:
        """Decrement the gauge by the given amount."""
        self.value -= amount

class Counter:
    """A simple counter metric that can only increase."""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.value = 0
        
    def inc(self, amount: int = 1) -> None:
        """Increment the counter by the given amount."""
        self.value += amount

def record_execution_time(func: Callable) -> Callable:
    """
    Decorator to record the execution time of a function.
    This is a simplified version for development use.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Log the execution time
        logger.debug(f"{func.__name__} executed in {execution_time:.4f} seconds")
        
        return result
    return wrapper

def time_operation(operation_name: str) -> Callable:
    """
    Decorator factory to time an operation with a custom name.
    This is a simplified version for development use.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Log the execution time
            logger.debug(f"{operation_name} executed in {execution_time:.4f} seconds")
            
            return result
        return wrapper
    return decorator