"""
Multi-agent coordinator for the CIVILIAN system.
Manages the creation, configuration, and cooperation between agents.
"""

import logging
import json
from typing import Dict, Any, List, Optional

from utils.text_processor import TextProcessor
from utils.vector_store import VectorStore
from utils.ai_processor import AIProcessor
from agents.agent_factory import AgentFactory
from models import SystemLog
from app import db

logger = logging.getLogger(__name__)

class MultiAgentCoordinator:
    """Coordinator for the CIVILIAN multi-agent system."""
    
    def __init__(self):
        """Initialize the multi-agent coordinator."""
        self.text_processor = TextProcessor()
        self.vector_store = VectorStore()
        self.ai_processor = AIProcessor()
        self.agent_factory = AgentFactory(self.text_processor, self.vector_store, self.ai_processor)
        self.running = False
        
        logger.info("MultiAgentCoordinator initialized")
    
    def initialize_agents(self):
        """Initialize all required agents."""
        # Create the core agents
        self.agent_factory.create_agent('detector')
        self.agent_factory.create_agent('analyzer')
        self.agent_factory.create_agent('counter')
        
        # Log the initialization
        try:
            log_entry = SystemLog(
                log_type="info",
                component="multi_agent_coordinator",
                message="Initialized CIVILIAN multi-agent system",
                meta_data=json.dumps({
                    "agent_count": len(self.agent_factory.get_all_agents()),
                    "agents": [agent.agent_type for agent in self.agent_factory.get_all_agents()]
                })
            )
            db.session.add(log_entry)
            db.session.commit()
        except Exception as e:
            logger.error(f"Error logging agent initialization: {e}")
        
        logger.info(f"Initialized {len(self.agent_factory.get_all_agents())} agents")
    
    def start_all_agents(self):
        """Start all registered agents."""
        if self.running:
            logger.warning("Multi-agent system is already running")
            return
        
        self.agent_factory.start_all_agents()
        self.running = True
        
        logger.info("Multi-agent system started")
    
    def stop_all_agents(self):
        """Stop all registered agents."""
        if not self.running:
            logger.warning("Multi-agent system is not running")
            return
        
        self.agent_factory.stop_all_agents()
        self.running = False
        
        logger.info("Multi-agent system stopped")
    
    def get_agent_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all agents.
        
        Returns:
            Dictionary mapping agent types to their statistics
        """
        return self.agent_factory.get_agent_stats()
    
    def get_system_summary(self) -> Dict[str, Any]:
        """Get a summary of the multi-agent system.
        
        Returns:
            Dictionary with system summary information
        """
        stats = self.get_agent_stats()
        
        total_cycles = sum(s.get('cycle_count', 0) for s in stats.values())
        total_errors = sum(s.get('error_count', 0) for s in stats.values())
        
        summary = {
            "status": "running" if self.running else "stopped",
            "agent_count": len(stats),
            "total_cycles": total_cycles,
            "total_errors": total_errors,
            "agents": list(stats.keys()),
            "agent_details": stats
        }
        
        return summary