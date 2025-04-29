"""
Concurrency utilities for the CIVILIAN system.

This module provides functions and decorators for managing concurrency,
including thread management for async operations.
"""

import threading
import logging
import functools
import time
from typing import Any, Callable, Dict, List, Optional

# Configure logging
logger = logging.getLogger(__name__)


def run_in_thread(func: Callable) -> Callable:
    """
    Decorator to run a function in a separate thread.
    
    Args:
        func: The function to run in a thread
        
    Returns:
        Wrapper function that starts a thread
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        thread = threading.Thread(target=func, args=args, kwargs=kwargs)
        thread.daemon = True
        thread.start()
        return thread
    return wrapper


class ThreadPool:
    """
    Simple thread pool implementation for managing multiple concurrent tasks.
    """
    
    def __init__(self, max_workers: int = 5, name: str = "ThreadPool"):
        """
        Initialize the thread pool.
        
        Args:
            max_workers: Maximum number of concurrent workers
            name: Name for this thread pool (for logging)
        """
        self.max_workers = max_workers
        self.name = name
        self.active_workers = 0
        self.queue = []
        self.lock = threading.Lock()
        self.workers = []
        self.results = {}
        self.active = True
        
        # Start the manager thread
        self.manager_thread = threading.Thread(target=self._manage_workers)
        self.manager_thread.daemon = True
        self.manager_thread.start()
        
        logger.debug(f"ThreadPool '{name}' initialized with {max_workers} workers")
    
    def submit(self, func: Callable, *args, **kwargs) -> str:
        """
        Submit a function to be executed by the thread pool.
        
        Args:
            func: Function to execute
            *args: Arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            Task ID that can be used to retrieve the result
        """
        task_id = str(hash(func.__name__ + str(time.time())))
        
        with self.lock:
            self.queue.append({
                'id': task_id,
                'func': func,
                'args': args,
                'kwargs': kwargs,
                'status': 'queued',
                'submitted_at': time.time()
            })
            logger.debug(f"Task {task_id} ({func.__name__}) submitted to ThreadPool '{self.name}'")
        
        return task_id
    
    def get_result(self, task_id: str, timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        Get the result of a task.
        
        Args:
            task_id: ID of the task to get result for
            timeout: Maximum time to wait for the result (None = wait indefinitely)
            
        Returns:
            Dictionary with task information including result and status
        """
        start_time = time.time()
        
        while timeout is None or (time.time() - start_time) < timeout:
            with self.lock:
                if task_id in self.results and self.results[task_id]['status'] != 'running':
                    return self.results[task_id]
            
            time.sleep(0.1)
        
        # If we get here, we timed out
        return {
            'id': task_id,
            'status': 'timeout',
            'result': None,
            'error': 'Timeout waiting for result'
        }
    
    def wait_all(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for all submitted tasks to complete.
        
        Args:
            timeout: Maximum time to wait (None = wait indefinitely)
            
        Returns:
            True if all tasks completed, False if timeout occurred
        """
        start_time = time.time()
        
        while timeout is None or (time.time() - start_time) < timeout:
            with self.lock:
                if len(self.queue) == 0 and self.active_workers == 0:
                    return True
            
            time.sleep(0.1)
        
        # If we get here, we timed out
        return False
    
    def shutdown(self, wait: bool = True) -> None:
        """
        Shutdown the thread pool.
        
        Args:
            wait: Whether to wait for all tasks to complete
        """
        logger.debug(f"Shutting down ThreadPool '{self.name}'")
        
        self.active = False
        
        if wait:
            self.wait_all()
    
    def _manage_workers(self) -> None:
        """Worker management thread that starts workers as needed."""
        while self.active:
            with self.lock:
                # Start new workers if needed
                while self.active_workers < self.max_workers and self.queue:
                    task = self.queue.pop(0)
                    task['status'] = 'running'
                    self.results[task['id']] = {
                        'id': task['id'],
                        'status': 'running',
                        'result': None,
                        'started_at': time.time()
                    }
                    
                    worker = threading.Thread(
                        target=self._worker_thread,
                        args=(task,)
                    )
                    worker.daemon = True
                    self.workers.append(worker)
                    self.active_workers += 1
                    worker.start()
                    
                    logger.debug(f"Started worker for task {task['id']} ({task['func'].__name__})")
            
            time.sleep(0.1)
    
    def _worker_thread(self, task: Dict[str, Any]) -> None:
        """
        Worker thread to execute a task.
        
        Args:
            task: Task dictionary with function and arguments
        """
        result = None
        error = None
        
        try:
            result = task['func'](*task['args'], **task['kwargs'])
            status = 'completed'
        except Exception as e:
            error = str(e)
            status = 'error'
            logger.exception(f"Error executing task {task['id']} ({task['func'].__name__}): {e}")
        
        with self.lock:
            self.active_workers -= 1
            self.results[task['id']] = {
                'id': task['id'],
                'status': status,
                'result': result,
                'error': error,
                'started_at': self.results[task['id']]['started_at'],
                'completed_at': time.time(),
                'duration': time.time() - self.results[task['id']]['started_at']
            }
            
            logger.debug(f"Task {task['id']} ({task['func'].__name__}) {status}")


class RateLimiter:
    """
    Rate limiter for controlling the frequency of operations.
    """
    
    def __init__(self, operations_per_second: float):
        """
        Initialize the rate limiter.
        
        Args:
            operations_per_second: Maximum operations per second
        """
        self.min_interval = 1.0 / operations_per_second
        self.last_operation_time = 0
        self.lock = threading.Lock()
    
    def wait(self) -> None:
        """
        Wait until the rate limit allows another operation.
        """
        with self.lock:
            current_time = time.time()
            elapsed = current_time - self.last_operation_time
            
            if elapsed < self.min_interval:
                sleep_time = self.min_interval - elapsed
                time.sleep(sleep_time)
            
            self.last_operation_time = time.time()


def with_rate_limit(operations_per_second: float) -> Callable:
    """
    Decorator to apply rate limiting to a function.
    
    Args:
        operations_per_second: Maximum operations per second
        
    Returns:
        Decorator function
    """
    limiter = RateLimiter(operations_per_second)
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            limiter.wait()
            return func(*args, **kwargs)
        return wrapper
    
    return decorator


class AsyncTask:
    """
    Class for managing an asynchronous task with progress tracking.
    """
    
    def __init__(self, task_id: str, name: str, total_steps: int = 100):
        """
        Initialize the asynchronous task.
        
        Args:
            task_id: Unique identifier for the task
            name: Human-readable name for the task
            total_steps: Total number of steps in the task
        """
        self.task_id = task_id
        self.name = name
        self.total_steps = total_steps
        self.current_step = 0
        self.status = 'pending'
        self.start_time = None
        self.end_time = None
        self.result = None
        self.error = None
        self.progress_message = ''
        self.lock = threading.Lock()
    
    def start(self) -> None:
        """
        Mark the task as started.
        """
        with self.lock:
            self.status = 'running'
            self.start_time = time.time()
    
    def update_progress(self, step: int, message: str = '') -> None:
        """
        Update the progress of the task.
        
        Args:
            step: Current step (0 to total_steps)
            message: Optional progress message
        """
        with self.lock:
            self.current_step = min(step, self.total_steps)
            if message:
                self.progress_message = message
    
    def complete(self, result: Any = None) -> None:
        """
        Mark the task as completed.
        
        Args:
            result: Optional result of the task
        """
        with self.lock:
            self.status = 'completed'
            self.end_time = time.time()
            self.current_step = self.total_steps
            self.result = result
    
    def fail(self, error: str) -> None:
        """
        Mark the task as failed.
        
        Args:
            error: Error message
        """
        with self.lock:
            self.status = 'failed'
            self.end_time = time.time()
            self.error = error
    
    def get_progress(self) -> Dict[str, Any]:
        """
        Get the current progress of the task.
        
        Returns:
            Dictionary with task progress information
        """
        with self.lock:
            progress = {
                'task_id': self.task_id,
                'name': self.name,
                'status': self.status,
                'current_step': self.current_step,
                'total_steps': self.total_steps,
                'progress_percent': (self.current_step / self.total_steps) * 100 if self.total_steps > 0 else 0,
                'message': self.progress_message
            }
            
            if self.start_time:
                progress['elapsed_seconds'] = time.time() - self.start_time
                
                if self.end_time:
                    progress['duration_seconds'] = self.end_time - self.start_time
                
                if self.status == 'running' and self.current_step > 0:
                    # Estimate time remaining based on progress
                    steps_per_second = self.current_step / (time.time() - self.start_time)
                    if steps_per_second > 0:
                        progress['estimated_seconds_remaining'] = (self.total_steps - self.current_step) / steps_per_second
            
            if self.status == 'completed' and self.result is not None:
                progress['result'] = self.result
                
            if self.status == 'failed' and self.error:
                progress['error'] = self.error
                
            return progress


class AsyncTaskManager:
    """
    Manager for asynchronous tasks with progress tracking.
    """
    
    def __init__(self):
        """
        Initialize the task manager.
        """
        self.tasks = {}
        self.lock = threading.Lock()
    
    def create_task(self, name: str, total_steps: int = 100) -> str:
        """
        Create a new asynchronous task.
        
        Args:
            name: Human-readable name for the task
            total_steps: Total number of steps in the task
            
        Returns:
            Task ID
        """
        task_id = str(hash(name + str(time.time())))
        
        with self.lock:
            self.tasks[task_id] = AsyncTask(task_id, name, total_steps)
            
        return task_id
    
    def get_task(self, task_id: str) -> Optional[AsyncTask]:
        """
        Get a task by ID.
        
        Args:
            task_id: ID of the task to get
            
        Returns:
            AsyncTask object or None if not found
        """
        with self.lock:
            return self.tasks.get(task_id)
    
    def get_progress(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the progress of a task.
        
        Args:
            task_id: ID of the task to get progress for
            
        Returns:
            Dictionary with task progress information or None if task not found
        """
        task = self.get_task(task_id)
        return task.get_progress() if task else None
    
    def get_all_tasks(self) -> List[Dict[str, Any]]:
        """
        Get information about all tasks.
        
        Returns:
            List of dictionaries with task information
        """
        with self.lock:
            return [task.get_progress() for task in self.tasks.values()]
    
    def cleanup(self, max_age_seconds: int = 3600) -> int:
        """
        Clean up old completed or failed tasks.
        
        Args:
            max_age_seconds: Maximum age of completed/failed tasks to keep
            
        Returns:
            Number of tasks removed
        """
        current_time = time.time()
        tasks_to_remove = []
        
        with self.lock:
            for task_id, task in self.tasks.items():
                if task.status in ['completed', 'failed'] and task.end_time:
                    if current_time - task.end_time > max_age_seconds:
                        tasks_to_remove.append(task_id)
            
            for task_id in tasks_to_remove:
                del self.tasks[task_id]
                
        return len(tasks_to_remove)