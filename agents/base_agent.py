"""
Base agent implementation for the CIVILIAN multi-agent system.
Defines core functionality shared by all specialized agent types.
"""

import logging
import time
import threading
import os
import json
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List

from utils.app_context import ensure_app_context
from models import SystemLog
from app import db

logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    """Base class for all CIVILIAN agents."""
    
    def __init__(self, agent_type: str, refresh_interval: int = 300):
        """Initialize the base agent.
        
        Args:
            agent_type: Type of agent (e.g., 'detector', 'analyzer', 'counter')
            refresh_interval: Seconds between processing cycles
        """
        self.agent_type = agent_type
        self.running = False
        self.thread = None
        self.refresh_interval = int(os.environ.get(f'{agent_type.upper()}_REFRESH_INTERVAL', refresh_interval))
        self._initialized_at = time.time()
        self._last_cycle_start = 0
        self._last_cycle_end = 0
        self._cycle_count = 0
        self._successful_cycles = 0
        self._error_count = 0
        
        logger.info(f"{self.__class__.__name__} initialized")
    
    def start(self):
        """Start the agent in a background thread."""
        if self.running:
            logger.warning(f"{self.__class__.__name__} is already running")
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._run_loop)
        self.thread.daemon = True
        self.thread.start()
        
        logger.info(f"{self.__class__.__name__} started")
        
    def stop(self):
        """Stop the agent."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5.0)
        logger.info(f"{self.__class__.__name__} stopped")
    
    def _run_loop(self):
        """Main processing loop that runs in background thread."""
        while self.running:
            try:
                # Track cycle metrics
                self._last_cycle_start = time.time()
                self._cycle_count += 1
                
                # Log start of cycle
                logger.debug(f"Starting {self.agent_type} cycle")
                
                # Run the agent's processing cycle
                self._process_cycle()
                
                # Update cycle metrics
                self._last_cycle_end = time.time()
                self._successful_cycles += 1
                
                # Wait for next cycle
                cycle_duration = self._last_cycle_end - self._last_cycle_start
                logger.debug(f"{self.agent_type} cycle complete in {cycle_duration:.2f}s, sleeping for {self.refresh_interval} seconds")
                time.sleep(self.refresh_interval)
                
            except Exception as e:
                self._error_count += 1
                logger.error(f"Error in {self.agent_type} loop: {e}")
                # Log error to database
                self._log_error(f"{self.agent_type}_loop", str(e))
                time.sleep(30)  # Short sleep on error
    
    @abstractmethod
    def _process_cycle(self):
        """Process a single cycle of the agent's operation.
        
        This method must be implemented by all agent subclasses.
        """
        pass
    
    @ensure_app_context
    def _log_error(self, operation: str, message: str):
        """Log an error to the database."""
        try:
            log_entry = SystemLog(
                log_type="error",
                component=f"{self.agent_type}_agent",
                message=f"Error in {operation}: {message}"
            )
            with db.session.begin():
                db.session.add(log_entry)
        except Exception:
            # Just log to console if database logging fails
            logger.error(f"Failed to log error to database: {message}")
    
    @ensure_app_context
    def _log_info(self, message: str, metadata: Optional[Dict[str, Any]] = None):
        """Log an informational message to the database."""
        try:
            log_entry = SystemLog(
                log_type="info",
                component=f"{self.agent_type}_agent",
                message=message,
                meta_data=json.dumps(metadata) if metadata else None
            )
            with db.session.begin():
                db.session.add(log_entry)
        except Exception:
            # Just log to console if database logging fails
            logger.info(f"[DB Log] {message}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics and performance metrics."""
        uptime = time.time() - self._initialized_at
        
        stats = {
            "agent_type": self.agent_type,
            "uptime": uptime,
            "cycle_count": self._cycle_count,
            "successful_cycles": self._successful_cycles,
            "error_count": self._error_count,
            "refresh_interval": self.refresh_interval
        }
        
        # Add last cycle time if we've completed at least one cycle
        if self._last_cycle_end > 0:
            stats["last_cycle_duration"] = self._last_cycle_end - self._last_cycle_start
            stats["last_cycle_completed_at"] = self._last_cycle_end
        
        return stats