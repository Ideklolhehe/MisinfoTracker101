"""
Concurrency utilities for the CIVILIAN system.
Provides thread-safe data structures and processing capabilities.
"""

import threading
import logging
import time
import uuid
from typing import Dict, List, Any, Callable, TypeVar, Generic, Optional, Tuple

from utils.environment import ENABLE_THREADING

# Configure module logger
logger = logging.getLogger(__name__)

# Generic type for the resource being protected
T = TypeVar('T')

class ResourceLock(Generic[T]):
    """
    Thread-safe resource access manager with read-write lock semantics.
    Uses a reader-writer pattern to allow concurrent reads but exclusive writes.
    """
    
    def __init__(self, resource: T):
        """
        Initialize the resource lock.
        
        Args:
            resource: The resource to protect
        """
        self.resource = resource
        self._lock = threading.RLock()
        self._reader_count = 0
        self._reader_lock = threading.Lock()
        self._writer_lock = threading.Lock()
        
    def read(self) -> T:
        """
        Get read access to the resource.
        Must be used in a with statement.
        
        Returns:
            Context manager for the resource
        """
        return self.ReadContext(self)
        
    def write(self) -> T:
        """
        Get write access to the resource.
        Must be used in a with statement.
        
        Returns:
            Context manager for the resource
        """
        return self.WriteContext(self)
    
    class ReadContext:
        """Context manager for read access to the resource."""
        
        def __init__(self, lock_manager):
            self.lock_manager = lock_manager
            
        def __enter__(self):
            # If threading is disabled, simply return the resource
            if not ENABLE_THREADING:
                return self.lock_manager.resource
                
            # Increment reader count atomically
            with self.lock_manager._reader_lock:
                self.lock_manager._reader_count += 1
                if self.lock_manager._reader_count == 1:
                    # First reader acquires the writer lock
                    self.lock_manager._writer_lock.acquire()
            
            return self.lock_manager.resource
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            if not ENABLE_THREADING:
                return
                
            # Decrement reader count atomically
            with self.lock_manager._reader_lock:
                self.lock_manager._reader_count -= 1
                if self.lock_manager._reader_count == 0:
                    # Last reader releases the writer lock
                    self.lock_manager._writer_lock.release()
    
    class WriteContext:
        """Context manager for write access to the resource."""
        
        def __init__(self, lock_manager):
            self.lock_manager = lock_manager
            
        def __enter__(self):
            # If threading is disabled, simply return the resource
            if not ENABLE_THREADING:
                return self.lock_manager.resource
                
            # Acquire exclusive write lock
            self.lock_manager._writer_lock.acquire()
            
            return self.lock_manager.resource
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            if not ENABLE_THREADING:
                return
                
            # Release write lock
            self.lock_manager._writer_lock.release()


class ThreadSafeDict(Dict[str, Any]):
    """
    Thread-safe dictionary implementation.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._lock = threading.RLock() if ENABLE_THREADING else None
        
    def __getitem__(self, key):
        if self._lock:
            with self._lock:
                return super().__getitem__(key)
        return super().__getitem__(key)
        
    def __setitem__(self, key, value):
        if self._lock:
            with self._lock:
                super().__setitem__(key, value)
        else:
            super().__setitem__(key, value)
            
    def __delitem__(self, key):
        if self._lock:
            with self._lock:
                super().__delitem__(key)
        else:
            super().__delitem__(key)
            
    def get(self, key, default=None):
        if self._lock:
            with self._lock:
                return super().get(key, default)
        return super().get(key, default)
        
    def setdefault(self, key, default=None):
        if self._lock:
            with self._lock:
                return super().setdefault(key, default)
        return super().setdefault(key, default)
        
    def update(self, *args, **kwargs):
        if self._lock:
            with self._lock:
                super().update(*args, **kwargs)
        else:
            super().update(*args, **kwargs)
            
    def pop(self, key, default=None):
        if self._lock:
            with self._lock:
                return super().pop(key, default)
        return super().pop(key, default)
        
    def items(self):
        if self._lock:
            with self._lock:
                return list(super().items())
        return super().items()
        
    def keys(self):
        if self._lock:
            with self._lock:
                return list(super().keys())
        return super().keys()
        
    def values(self):
        if self._lock:
            with self._lock:
                return list(super().values())
        return super().values()
        
    def clear(self):
        if self._lock:
            with self._lock:
                super().clear()
        else:
            super().clear()
            
    def copy(self):
        if self._lock:
            with self._lock:
                return dict(super().items())
        return dict(super().items())


class BatchProcessor:
    """
    Processor for batch operations with thread safety.
    """
    
    def __init__(self, batch_size: int = 100, processing_interval: float = 0.0):
        """
        Initialize the batch processor.
        
        Args:
            batch_size: Maximum items to process in a batch
            processing_interval: Time to wait between batches (seconds)
        """
        self.batch_size = batch_size
        self.processing_interval = processing_interval
        self.items = []
        self._lock = threading.Lock() if ENABLE_THREADING else None
        
    def add_item(self, item: Any) -> None:
        """
        Add an item to the batch.
        
        Args:
            item: The item to add
        """
        if self._lock:
            with self._lock:
                self.items.append(item)
        else:
            self.items.append(item)
            
    def process_batch(self, processor_func: Callable[[List[Any]], None]) -> int:
        """
        Process a batch of items.
        
        Args:
            processor_func: Function to process the batch
            
        Returns:
            Number of items processed
        """
        batch = []
        if self._lock:
            with self._lock:
                # Get items up to batch size
                batch = self.items[:self.batch_size]
                # Remove processed items
                self.items = self.items[self.batch_size:]
        else:
            batch = self.items[:self.batch_size]
            self.items = self.items[self.batch_size:]
            
        if batch:
            processor_func(batch)
            if self.processing_interval > 0:
                time.sleep(self.processing_interval)
                
        return len(batch)
        
    def process_all(self, processor_func: Callable[[List[Any]], None]) -> int:
        """
        Process all items in batches.
        
        Args:
            processor_func: Function to process each batch
            
        Returns:
            Total number of items processed
        """
        total_processed = 0
        while True:
            processed = self.process_batch(processor_func)
            if processed == 0:
                break
            total_processed += processed
            
        return total_processed
        
    def get_pending_count(self) -> int:
        """
        Get the number of items pending processing.
        
        Returns:
            Number of pending items
        """
        if self._lock:
            with self._lock:
                return len(self.items)
        return len(self.items)
        
    def clear(self) -> None:
        """Clear all pending items."""
        if self._lock:
            with self._lock:
                self.items.clear()
        else:
            self.items.clear()


def generate_id(prefix: str = "") -> str:
    """
    Generate a unique ID string.
    
    Args:
        prefix: Optional prefix for the ID
        
    Returns:
        A unique ID string
    """
    return f"{prefix}{uuid.uuid4()}"