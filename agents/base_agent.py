"""
Base agent for the CIVILIAN multi-agent system.
Provides common functionality for all agent types.
"""

import logging
import time
import threading
import json
from typing import Dict, Any, Optional
from datetime import datetime

from app import db
from models import SystemLog
from utils.app_context import AppContextThread, with_app_context

logger = logging.getLogger(__name__)

class BaseAgent:
    """Base class for all CIVILIAN agents."""
    
    def __init__(self, agent_type: str, refresh_interval: int = 300):
        """Initialize the base agent.
        
        Args:
            agent_type: Type of agent ('detector', 'analyzer', 'counter', etc.)
            refresh_interval: Time between agent cycles in seconds (default: 300)
        """
        self.agent_type = agent_type
        self.refresh_interval = refresh_interval
        self.is_running = False
        self.thread = None
        
        # Stats tracking
        self.cycle_count = 0
        self.error_count = 0
        self.last_cycle_time = None
        self.last_cycle_duration = None
        self.last_error = None
        
        logger.info(f"{self.__class__.__name__} initialized")
    
    def start(self):
        """Start the agent's processing thread."""
        if self.is_running:
            logger.warning(f"{self.__class__.__name__} is already running")
            return
            
        self.is_running = True
        # Use AppContextThread instead of standard Thread to maintain Flask context
        self.thread = AppContextThread(target=self._run, daemon=True)
        self.thread.start()
        
        logger.info(f"{self.__class__.__name__} started")
    
    def stop(self):
        """Stop the agent's processing thread."""
        if not self.is_running:
            logger.warning(f"{self.__class__.__name__} is not running")
            return
            
        self.is_running = False
        
        if self.thread and self.thread.is_alive():
            try:
                self.thread.join(timeout=5.0)  # Wait for thread to finish, with timeout
            except Exception as e:
                logger.error(f"Error stopping {self.__class__.__name__}: {e}")
        
        logger.info(f"{self.__class__.__name__} stopped")
    
    def _run(self):
        """Run the agent's processing loop."""
        while self.is_running:
            try:
                self.last_cycle_time = datetime.utcnow()
                
                # Log cycle start
                logger.debug(f"Starting {self.agent_type} cycle")
                
                # Track cycle time
                start_time = time.time()
                
                # Process the cycle
                self._process_cycle()
                
                # Track stats
                self.cycle_count += 1
                self.last_cycle_duration = time.time() - start_time
                
                # Log cycle completion
                logger.debug(f"{self.agent_type} cycle complete, sleeping for {self.refresh_interval} seconds")
                
            except Exception as e:
                self.error_count += 1
                self.last_error = str(e)
                logger.error(f"Error in {self.agent_type} cycle: {e}")
                self._log_error("cycle_error", str(e))
            
            # Sleep until next cycle
            for _ in range(int(self.refresh_interval / 5)):
                if not self.is_running:
                    break
                time.sleep(5)
    
    def _process_cycle(self):
        """Process a single agent cycle.
        
        This method should be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses must implement _process_cycle")
    
    def _log_info(self, operation: str, details: Optional[Dict[str, Any]] = None):
        """Log an informational event to the system log.
        
        Args:
            operation: The operation being performed
            details: Optional details as a dictionary
        """
        try:
            log_entry = SystemLog(
                log_type="info",
                component=f"{self.agent_type}_agent",
                message=f"{operation} completed successfully",
                meta_data=json.dumps(details) if details else None
            )
            db.session.add(log_entry)
            db.session.commit()
        except Exception as e:
            logger.error(f"Error logging info: {e}")
    
    def _log_error(self, operation: str, error_message: str, details: Optional[Dict[str, Any]] = None):
        """Log an error event to the system log.
        
        Args:
            operation: The operation being performed
            error_message: Error message
            details: Optional details as a dictionary
        """
        try:
            if not details:
                details = {}
            details["error"] = error_message
            
            log_entry = SystemLog(
                log_type="error",
                component=f"{self.agent_type}_agent",
                message=f"Error during {operation}",
                meta_data=json.dumps(details)
            )
            db.session.add(log_entry)
            db.session.commit()
        except Exception as e:
            logger.error(f"Error logging error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for this agent.
        
        Returns:
            Dictionary with agent statistics
        """
        return {
            "agent_type": self.agent_type,
            "running": self.is_running,
            "cycle_count": self.cycle_count,
            "error_count": self.error_count,
            "last_cycle_time": self.last_cycle_time.isoformat() if self.last_cycle_time else None,
            "last_cycle_duration": self.last_cycle_duration,
            "last_error": self.last_error
        }