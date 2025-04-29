"""
Concurrency utilities for the CIVILIAN system.
Provides utilities for thread management, thread-safe operations, and asynchronous execution.
"""

import concurrent.futures
import functools
import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union, cast

from utils.app_context import get_app_context, get_current_app

# Configure module logger
logger = logging.getLogger(__name__)

# Type variables for function signatures
F = TypeVar('F', bound=Callable[..., Any])
T = TypeVar('T')

# Global thread pool executor for background tasks
_thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=10)

# Thread-local storage for task context
_thread_local = threading.local()


def run_in_thread(func: Callable[..., T]) -> Callable[..., concurrent.futures.Future[T]]:
    """
    Run a function in a separate thread from the thread pool.
    
    Args:
        func: Function to run
        
    Returns:
        Decorated function that returns a Future
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> concurrent.futures.Future[T]:
        future = _thread_pool.submit(func, *args, **kwargs)
        return future
    
    return wrapper


def run_in_thread_with_app_context(func: Callable[..., T]) -> Callable[..., concurrent.futures.Future[T]]:
    """
    Run a function in a separate thread with application context.
    
    Args:
        func: Function to run
        
    Returns:
        Decorated function that returns a Future
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> concurrent.futures.Future[T]:
        app = get_current_app()
        if app is None:
            logger.error(f"No application context available for {func.__name__}")
            raise RuntimeError(
                f"No application context available for {func.__name__}. "
                "Make sure set_current_app() was called."
            )
        
        def run_with_context() -> T:
            with app.app_context():
                return func(*args, **kwargs)
        
        future = _thread_pool.submit(run_with_context)
        return future
    
    return wrapper


def periodic_task(interval: int, start_immediately: bool = False, daemon: bool = True):
    """
    Decorator to run a function periodically at a specified interval.
    
    Args:
        interval: Interval in seconds
        start_immediately: Whether to start the task immediately
        daemon: Whether the thread should be a daemon thread
        
    Returns:
        Decorator function
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> threading.Thread:
            app = get_current_app()
            
            def task_runner() -> None:
                if not start_immediately:
                    time.sleep(interval)
                
                while not getattr(_thread_local, 'stop_requested', False):
                    try:
                        if app:
                            with app.app_context():
                                func(*args, **kwargs)
                        else:
                            func(*args, **kwargs)
                    except Exception as e:
                        logger.error(f"Error in periodic task {func.__name__}: {str(e)}")
                    
                    time.sleep(interval)
            
            thread = threading.Thread(target=task_runner)
            thread.daemon = daemon
            thread.name = f"PeriodicTask-{func.__name__}"
            thread.start()
            
            # Store the thread reference
            setattr(_thread_local, f"{func.__name__}_thread", thread)
            
            return thread
        
        # Add control functions
        def stop() -> None:
            setattr(_thread_local, 'stop_requested', True)
        
        wrapper.stop = stop  # type: ignore
        
        return cast(F, wrapper)
    
    return decorator


class ThreadSafeDict(Dict[Any, Any]):
    """Thread-safe dictionary implementation."""
    
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._lock = threading.RLock()
        super().__init__(*args, **kwargs)
    
    def __getitem__(self, key: Any) -> Any:
        with self._lock:
            return super().__getitem__(key)
    
    def __setitem__(self, key: Any, value: Any) -> None:
        with self._lock:
            super().__setitem__(key, value)
    
    def __delitem__(self, key: Any) -> None:
        with self._lock:
            super().__delitem__(key)
    
    def get(self, key: Any, default: Any = None) -> Any:
        with self._lock:
            return super().get(key, default)
    
    def update(self, *args: Any, **kwargs: Any) -> None:
        with self._lock:
            super().update(*args, **kwargs)
    
    def pop(self, key: Any, default: Any = None) -> Any:
        with self._lock:
            return super().pop(key, default)
    
    def clear(self) -> None:
        with self._lock:
            super().clear()


class BackgroundTaskManager:
    """Manager for background tasks."""
    
    def __init__(self, max_workers: int = 10) -> None:
        """
        Initialize the task manager.
        
        Args:
            max_workers: Maximum number of worker threads
        """
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.tasks: Dict[str, concurrent.futures.Future[Any]] = {}
        self._lock = threading.RLock()
    
    def submit(self, task_id: str, func: Callable[..., T], *args: Any, **kwargs: Any) -> concurrent.futures.Future[T]:
        """
        Submit a task to the executor.
        
        Args:
            task_id: Unique identifier for the task
            func: Function to execute
            *args: Arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Future for the task
        """
        with self._lock:
            # Cancel existing task if it exists
            if task_id in self.tasks and not self.tasks[task_id].done():
                self.tasks[task_id].cancel()
            
            app = get_current_app()
            
            if app:
                # Run with application context
                def run_with_context() -> T:
                    with app.app_context():
                        return func(*args, **kwargs)
                
                future = self.executor.submit(run_with_context)
            else:
                # Run without application context
                future = self.executor.submit(func, *args, **kwargs)
            
            self.tasks[task_id] = future
            return future
    
    def get_task(self, task_id: str) -> Optional[concurrent.futures.Future[Any]]:
        """
        Get a task by ID.
        
        Args:
            task_id: Task identifier
            
        Returns:
            Future for the task or None if not found
        """
        with self._lock:
            return self.tasks.get(task_id)
    
    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a task.
        
        Args:
            task_id: Task identifier
            
        Returns:
            True if the task was cancelled, False otherwise
        """
        with self._lock:
            if task_id in self.tasks and not self.tasks[task_id].done():
                return self.tasks[task_id].cancel()
            return False
    
    def shutdown(self, wait: bool = True) -> None:
        """
        Shutdown the executor.
        
        Args:
            wait: Whether to wait for tasks to complete
        """
        self.executor.shutdown(wait=wait)
        with self._lock:
            self.tasks.clear()


# Global task manager instance
task_manager = BackgroundTaskManager()


def submit_background_task(task_id: str, func: Callable[..., T], *args: Any, **kwargs: Any) -> concurrent.futures.Future[T]:
    """
    Submit a task to the global task manager.
    
    Args:
        task_id: Unique identifier for the task
        func: Function to execute
        *args: Arguments for the function
        **kwargs: Keyword arguments for the function
        
    Returns:
        Future for the task
    """
    return task_manager.submit(task_id, func, *args, **kwargs)