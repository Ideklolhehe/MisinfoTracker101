"""
Agent factory for the CIVILIAN multi-agent system.
Manages creation and access to different types of agents.
"""

import logging
from typing import Dict, Any, List, Optional, Type

from utils.text_processor import TextProcessor
from utils.vector_store import VectorStore
from utils.ai_processor import AIProcessor
from agents.base_agent import BaseAgent
from agents.detector_agent_v2 import DetectorAgent
from agents.analyzer_agent_v2 import AnalyzerAgent
from agents.counter_agent_v2 import CounterAgent

logger = logging.getLogger(__name__)

class AgentFactory:
    """Factory for creating and managing agents in the CIVILIAN system."""
    
    def __init__(self, text_processor: TextProcessor, vector_store: VectorStore, ai_processor: AIProcessor):
        """Initialize the agent factory.
        
        Args:
            text_processor: Text processing utility
            vector_store: Vector storage for embeddings
            ai_processor: AI processing utility
        """
        self.text_processor = text_processor
        self.vector_store = vector_store
        self.ai_processor = ai_processor
        self._agents = {}
        
        logger.info("AgentFactory initialized")
    
    def create_agent(self, agent_type: str) -> BaseAgent:
        """Create an agent of the specified type.
        
        Args:
            agent_type: Type of agent to create ('detector', 'analyzer', or 'counter')
            
        Returns:
            The created agent instance
            
        Raises:
            ValueError: If an invalid agent type is specified
        """
        if agent_type in self._agents:
            logger.warning(f"Agent of type {agent_type} already exists, returning existing instance")
            return self._agents[agent_type]
            
        agent = None
        
        if agent_type == 'detector':
            agent = DetectorAgent(self.text_processor, self.vector_store)
            
        elif agent_type == 'analyzer':
            agent = AnalyzerAgent(self.text_processor)
            
        elif agent_type == 'counter':
            agent = CounterAgent(self.text_processor)
            
        else:
            raise ValueError(f"Invalid agent type: {agent_type}")
            
        # Store and return the agent
        self._agents[agent_type] = agent
        logger.info(f"Created agent of type {agent_type}")
        
        return agent
    
    def get_agent(self, agent_type: str) -> Optional[BaseAgent]:
        """Get an existing agent by type.
        
        Args:
            agent_type: Type of agent to retrieve
            
        Returns:
            The agent instance, or None if not found
        """
        return self._agents.get(agent_type)
    
    def get_all_agents(self) -> List[BaseAgent]:
        """Get all registered agents.
        
        Returns:
            List of all agent instances
        """
        return list(self._agents.values())
    
    def start_all_agents(self) -> None:
        """Start all registered agents."""
        for agent_type, agent in self._agents.items():
            if not agent.is_running:
                agent.start()
                logger.info(f"Started agent {agent_type}")
    
    def stop_all_agents(self) -> None:
        """Stop all registered agents."""
        for agent_type, agent in self._agents.items():
            if agent.is_running:
                agent.stop()
                logger.info(f"Stopped agent {agent_type}")
    
    def get_agent_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all agents.
        
        Returns:
            Dictionary mapping agent types to statistics
        """
        stats = {}
        
        for agent_type, agent in self._agents.items():
            agent_stats = {
                'running': agent.is_running,
                'cycle_count': agent.cycle_count,
                'error_count': agent.error_count,
                'last_cycle_duration': agent.last_cycle_duration,
                'last_error': agent.last_error
            }
            
            stats[agent_type] = agent_stats
            
        return stats