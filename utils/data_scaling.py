"""
Data scaling utilities for handling large volumes of real-time data.
This module provides functions for efficient data processing, caching,
and management of large datasets from internet sources.
"""

import logging
import os
import time
import json
import hashlib
import threading
import sqlite3
from typing import Dict, List, Any, Optional, Callable, Union, Generator
from datetime import datetime, timedelta
import pickle
from collections import defaultdict, deque

# Configure logger
logger = logging.getLogger(__name__)

# Constants
CACHE_DIR = "./storage/data_cache"
DB_PATH = f"{CACHE_DIR}/scaling_cache.db"
MAX_CACHE_AGE = 86400  # 24 hours in seconds
CACHE_CLEANUP_INTERVAL = 3600  # 1 hour in seconds
DEFAULT_BATCH_SIZE = 100
DEFAULT_PARALLEL_PROCESSES = 5
DEFAULT_MAX_QUEUE_SIZE = 10000

# Ensure cache directory exists
os.makedirs(CACHE_DIR, exist_ok=True)


class DataCache:
    """Thread-safe cache for data storage and retrieval."""
    
    def __init__(self, namespace: str = "default", max_size: int = 1000, ttl: int = MAX_CACHE_AGE):
        """
        Initialize the data cache.
        
        Args:
            namespace: Cache namespace for isolation
            max_size: Maximum number of items to store in memory
            ttl: Time to live in seconds for cached items
        """
        self.namespace = namespace
        self.max_size = max_size
        self.ttl = ttl
        self.cache = {}
        self.access_times = {}
        self.cache_lock = threading.Lock()
        
        # Initialize SQLite cache if needed
        self._init_db_cache()
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_thread, daemon=True)
        self.cleanup_thread.start()
        
        logger.info(f"DataCache initialized: namespace={namespace}, max_size={max_size}")
    
    def _init_db_cache(self):
        """Initialize the SQLite cache database."""
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            # Create cache table if it doesn't exist
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS data_cache (
                    namespace TEXT,
                    key TEXT,
                    value BLOB,
                    created_at REAL,
                    expires_at REAL,
                    PRIMARY KEY (namespace, key)
                )
            ''')
            
            # Create index on expiration
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_expires_at ON data_cache (expires_at)
            ''')
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error initializing cache database: {e}")
    
    def _key_to_hash(self, key: str) -> str:
        """Convert key to a hash for storage."""
        return hashlib.md5(f"{self.namespace}:{key}".encode()).hexdigest()
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to store
            ttl: Custom TTL for this value (None for default)
            
        Returns:
            True if successful, False otherwise
        """
        if ttl is None:
            ttl = self.ttl
            
        now = time.time()
        expires_at = now + ttl
        hashed_key = self._key_to_hash(key)
        
        try:
            # Store in memory cache
            with self.cache_lock:
                self.cache[hashed_key] = value
                self.access_times[hashed_key] = now
                
                # Trim cache if needed
                if len(self.cache) > self.max_size:
                    self._trim_cache()
            
            # Store in SQLite cache
            try:
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                
                # Serialize value
                serialized = pickle.dumps(value)
                
                # Insert or replace
                cursor.execute(
                    "INSERT OR REPLACE INTO data_cache VALUES (?, ?, ?, ?, ?)",
                    (self.namespace, hashed_key, serialized, now, expires_at)
                )
                
                conn.commit()
                conn.close()
            except Exception as e:
                logger.error(f"Error storing in SQLite cache: {e}")
            
            return True
        except Exception as e:
            logger.error(f"Error setting cache value: {e}")
            return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a value from the cache.
        
        Args:
            key: Cache key
            default: Default value if key not found
            
        Returns:
            Cached value or default
        """
        hashed_key = self._key_to_hash(key)
        
        # Try memory cache first
        with self.cache_lock:
            if hashed_key in self.cache:
                self.access_times[hashed_key] = time.time()
                return self.cache[hashed_key]
        
        # Try SQLite cache
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            # Get value and check expiration
            cursor.execute(
                "SELECT value, expires_at FROM data_cache WHERE namespace = ? AND key = ?",
                (self.namespace, hashed_key)
            )
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                value_blob, expires_at = result
                
                # Check if expired
                if time.time() > expires_at:
                    self.delete(key)
                    return default
                
                # Deserialize and update memory cache
                value = pickle.loads(value_blob)
                with self.cache_lock:
                    self.cache[hashed_key] = value
                    self.access_times[hashed_key] = time.time()
                
                return value
        except Exception as e:
            logger.error(f"Error getting from SQLite cache: {e}")
        
        return default
    
    def delete(self, key: str) -> bool:
        """
        Delete a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if successful, False otherwise
        """
        hashed_key = self._key_to_hash(key)
        
        try:
            # Remove from memory cache
            with self.cache_lock:
                if hashed_key in self.cache:
                    del self.cache[hashed_key]
                if hashed_key in self.access_times:
                    del self.access_times[hashed_key]
            
            # Remove from SQLite cache
            try:
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                
                cursor.execute(
                    "DELETE FROM data_cache WHERE namespace = ? AND key = ?",
                    (self.namespace, hashed_key)
                )
                
                conn.commit()
                conn.close()
            except Exception as e:
                logger.error(f"Error deleting from SQLite cache: {e}")
            
            return True
        except Exception as e:
            logger.error(f"Error deleting cache value: {e}")
            return False
    
    def clear(self) -> bool:
        """
        Clear the entire cache for this namespace.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Clear memory cache
            with self.cache_lock:
                self.cache.clear()
                self.access_times.clear()
            
            # Clear SQLite cache
            try:
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                
                cursor.execute(
                    "DELETE FROM data_cache WHERE namespace = ?",
                    (self.namespace,)
                )
                
                conn.commit()
                conn.close()
            except Exception as e:
                logger.error(f"Error clearing SQLite cache: {e}")
            
            return True
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False
    
    def _trim_cache(self):
        """Trim the memory cache by removing least recently used items."""
        with self.cache_lock:
            # Sort by access time
            items_by_access = sorted(self.access_times.items(), key=lambda x: x[1])
            
            # Remove oldest items until under max size
            items_to_remove = len(self.cache) - self.max_size
            for i in range(items_to_remove):
                if i < len(items_by_access):
                    key = items_by_access[i][0]
                    if key in self.cache:
                        del self.cache[key]
                    if key in self.access_times:
                        del self.access_times[key]
    
    def _cleanup_thread(self):
        """Background thread to clean up expired items."""
        while True:
            try:
                # Sleep before first cleanup
                time.sleep(CACHE_CLEANUP_INTERVAL)
                
                # Clean up memory cache
                now = time.time()
                with self.cache_lock:
                    expired_keys = []
                    for key, access_time in self.access_times.items():
                        if now - access_time > self.ttl:
                            expired_keys.append(key)
                    
                    for key in expired_keys:
                        if key in self.cache:
                            del self.cache[key]
                        if key in self.access_times:
                            del self.access_times[key]
                
                # Clean up SQLite cache
                try:
                    conn = sqlite3.connect(DB_PATH)
                    cursor = conn.cursor()
                    
                    cursor.execute(
                        "DELETE FROM data_cache WHERE expires_at < ?",
                        (now,)
                    )
                    
                    deleted_count = cursor.rowcount
                    conn.commit()
                    conn.close()
                    
                    if deleted_count > 0:
                        logger.debug(f"Cleaned up {deleted_count} expired items from SQLite cache")
                except Exception as e:
                    logger.error(f"Error cleaning up SQLite cache: {e}")
            except Exception as e:
                logger.error(f"Error in cache cleanup thread: {e}")


class BatchProcessor:
    """Process data in batches for efficient handling of large datasets."""
    
    def __init__(self, batch_size: int = DEFAULT_BATCH_SIZE, parallel: int = DEFAULT_PARALLEL_PROCESSES):
        """
        Initialize the batch processor.
        
        Args:
            batch_size: Size of each batch
            parallel: Number of parallel processes/threads
        """
        self.batch_size = batch_size
        self.parallel = parallel
        logger.info(f"BatchProcessor initialized: batch_size={batch_size}, parallel={parallel}")
    
    def process(self, items: List[Any], process_func: Callable[[Any], Any], callback: Optional[Callable[[List[Any]], None]] = None) -> List[Any]:
        """
        Process items in batches.
        
        Args:
            items: List of items to process
            process_func: Function to apply to each item
            callback: Optional callback function for each completed batch
            
        Returns:
            List of processed items
        """
        results = []
        total_items = len(items)
        processed_count = 0
        
        # Process in batches
        for i in range(0, total_items, self.batch_size):
            batch = items[i:i+self.batch_size]
            batch_results = []
            
            # Process the batch in parallel if enabled
            if self.parallel > 1 and len(batch) > 1:
                threads = []
                lock = threading.Lock()
                
                def process_item(item):
                    try:
                        result = process_func(item)
                        with lock:
                            batch_results.append(result)
                    except Exception as e:
                        logger.error(f"Error processing item in batch: {e}")
                
                # Create and start threads
                for item in batch:
                    thread = threading.Thread(target=process_item, args=(item,))
                    threads.append(thread)
                    thread.start()
                
                # Wait for all threads to complete
                for thread in threads:
                    thread.join()
            else:
                # Process sequentially
                for item in batch:
                    try:
                        result = process_func(item)
                        batch_results.append(result)
                    except Exception as e:
                        logger.error(f"Error processing item in batch: {e}")
            
            # Add batch results to overall results
            results.extend(batch_results)
            processed_count += len(batch)
            
            # Call callback if provided
            if callback is not None:
                try:
                    callback(batch_results)
                except Exception as e:
                    logger.error(f"Error in batch callback: {e}")
                    
            logger.debug(f"Processed batch: {processed_count}/{total_items} items")
        
        return results


class StreamBuffer:
    """Efficient buffer for streaming data processing."""
    
    def __init__(self, max_size: int = DEFAULT_MAX_QUEUE_SIZE):
        """
        Initialize the stream buffer.
        
        Args:
            max_size: Maximum buffer size before blocking
        """
        self.buffer = deque(maxlen=max_size)
        self.buffer_lock = threading.Lock()
        self.not_empty = threading.Condition(self.buffer_lock)
        self.not_full = threading.Condition(self.buffer_lock)
        self.max_size = max_size
        self.is_closed = False
        logger.info(f"StreamBuffer initialized: max_size={max_size}")
    
    def put(self, item: Any, block: bool = True, timeout: Optional[float] = None) -> bool:
        """
        Add an item to the buffer.
        
        Args:
            item: Item to add
            block: Whether to block if buffer is full
            timeout: Maximum time to block in seconds
            
        Returns:
            True if successful, False if buffer full or closed
        """
        if self.is_closed:
            return False
            
        with self.buffer_lock:
            if len(self.buffer) >= self.max_size:
                if not block:
                    return False
                    
                # Wait for space in buffer
                if timeout is not None:
                    end_time = time.time() + timeout
                    remaining = timeout
                    
                    while len(self.buffer) >= self.max_size and remaining > 0:
                        self.not_full.wait(remaining)
                        if self.is_closed:
                            return False
                        remaining = end_time - time.time()
                        
                    if len(self.buffer) >= self.max_size:
                        return False
                else:
                    while len(self.buffer) >= self.max_size:
                        self.not_full.wait()
                        if self.is_closed:
                            return False
            
            # Add item to buffer
            self.buffer.append(item)
            self.not_empty.notify()
            return True
    
    def get(self, block: bool = True, timeout: Optional[float] = None) -> Optional[Any]:
        """
        Get an item from the buffer.
        
        Args:
            block: Whether to block if buffer is empty
            timeout: Maximum time to block in seconds
            
        Returns:
            Item from buffer or None if empty or timeout
        """
        with self.buffer_lock:
            if not self.buffer:
                if not block:
                    return None
                    
                # Wait for item in buffer
                if timeout is not None:
                    end_time = time.time() + timeout
                    remaining = timeout
                    
                    while not self.buffer and remaining > 0 and not self.is_closed:
                        self.not_empty.wait(remaining)
                        remaining = end_time - time.time()
                        
                    if not self.buffer:
                        return None
                else:
                    while not self.buffer and not self.is_closed:
                        self.not_empty.wait()
                        
                    if not self.buffer:
                        return None
            
            # Get item from buffer
            item = self.buffer.popleft()
            self.not_full.notify()
            return item
    
    def close(self):
        """Close the buffer, preventing new items from being added."""
        with self.buffer_lock:
            self.is_closed = True
            self.not_empty.notify_all()
            self.not_full.notify_all()
    
    def __iter__(self):
        """Iterate over items in the buffer."""
        return self
    
    def __next__(self):
        """Get next item from the buffer."""
        item = self.get()
        if item is None and self.is_closed:
            raise StopIteration
        return item
    
    def stream(self, timeout: Optional[float] = None) -> Generator[Any, None, None]:
        """
        Stream items from the buffer.
        
        Args:
            timeout: Maximum time to wait for each item
            
        Yields:
            Items from the buffer
        """
        while not self.is_closed or len(self.buffer) > 0:
            item = self.get(block=True, timeout=timeout)
            if item is not None:
                yield item
            elif self.is_closed:
                break


class DataScaler:
    """Manager class for data scaling operations."""
    
    def __init__(self):
        """Initialize the data scaler."""
        self.caches = {}
        self.batch_processor = BatchProcessor()
        self.stream_buffers = {}
        logger.info("DataScaler initialized")
    
    def get_cache(self, namespace: str = "default", max_size: int = 1000, ttl: int = MAX_CACHE_AGE) -> DataCache:
        """
        Get a cache instance for a namespace.
        
        Args:
            namespace: Cache namespace
            max_size: Maximum items in memory cache
            ttl: Time to live for cached items
            
        Returns:
            DataCache instance
        """
        if namespace not in self.caches:
            self.caches[namespace] = DataCache(namespace=namespace, max_size=max_size, ttl=ttl)
        return self.caches[namespace]
    
    def get_batch_processor(self, batch_size: int = DEFAULT_BATCH_SIZE, parallel: int = DEFAULT_PARALLEL_PROCESSES) -> BatchProcessor:
        """
        Get a batch processor with custom settings.
        
        Args:
            batch_size: Size of each batch
            parallel: Number of parallel processes
            
        Returns:
            BatchProcessor instance
        """
        if batch_size != self.batch_processor.batch_size or parallel != self.batch_processor.parallel:
            return BatchProcessor(batch_size=batch_size, parallel=parallel)
        return self.batch_processor
    
    def get_stream_buffer(self, name: str = "default", max_size: int = DEFAULT_MAX_QUEUE_SIZE) -> StreamBuffer:
        """
        Get a stream buffer instance.
        
        Args:
            name: Buffer name
            max_size: Maximum buffer size
            
        Returns:
            StreamBuffer instance
        """
        if name not in self.stream_buffers:
            self.stream_buffers[name] = StreamBuffer(max_size=max_size)
        return self.stream_buffers[name]
    
    def close_stream(self, name: str):
        """
        Close a stream buffer.
        
        Args:
            name: Buffer name
        """
        if name in self.stream_buffers:
            self.stream_buffers[name].close()


# Create a singleton instance
data_scaler = DataScaler()