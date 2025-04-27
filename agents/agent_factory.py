"""
Agent factory for the CIVILIAN multi-agent system.
Manages agent instantiation, configuration, and coordination.
"""

import logging
from typing import Dict, List, Any, Optional, Type

from utils.text_processor import TextProcessor
from utils.vector_store import VectorStore
from utils.ai_processor import AIProcessor
from agents.base_agent import BaseAgent
from agents.analyzer_agent import AnalyzerAgent
from agents.detector_agent import DetectorAgent
from agents.counter_agent import CounterAgent

logger = logging.getLogger(__name__)

class AgentFactory:
    """Factory class for creating and managing CIVILIAN agents."""
    
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
        self.agents: Dict[str, BaseAgent] = {}
        
        logger.info("AgentFactory initialized")
    
    def create_agent(self, agent_type: str, **kwargs) -> BaseAgent:
        """Create a specific type of agent.
        
        Args:
            agent_type: Type of agent to create ('analyzer', 'detector', 'counter', etc.)
            **kwargs: Additional arguments to pass to the agent constructor
            
        Returns:
            The created agent instance
            
        Raises:
            ValueError: If the agent type is not supported
        """
        # Check for existing agent of this type
        if agent_type in self.agents:
            logger.info(f"Agent of type {agent_type} already exists, returning existing instance")
            return self.agents[agent_type]
        
        # Create a new agent based on the type
        if agent_type == 'analyzer':
            agent = AnalyzerAgent(self.text_processor)
        elif agent_type == 'detector':
            agent = DetectorAgent(self.text_processor, self.vector_store)
        elif agent_type == 'counter':
            agent = CounterAgent(self.text_processor)
        else:
            raise ValueError(f"Unsupported agent type: {agent_type}")
        
        # Store the agent reference
        self.agents[agent_type] = agent
        logger.info(f"Created new {agent_type} agent")
        
        return agent
    
    def start_all_agents(self):
        """Start all registered agents."""
        for agent_type, agent in self.agents.items():
            agent.start()
        
        logger.info(f"Started {len(self.agents)} agents")
    
    def stop_all_agents(self):
        """Stop all registered agents."""
        for agent_type, agent in self.agents.items():
            agent.stop()
        
        logger.info(f"Stopped {len(self.agents)} agents")
    
    def get_agent(self, agent_type: str) -> Optional[BaseAgent]:
        """Get an agent by type.
        
        Args:
            agent_type: Type of agent to retrieve
            
        Returns:
            The agent instance or None if not found
        """
        return self.agents.get(agent_type)
    
    def get_all_agents(self) -> List[BaseAgent]:
        """Get all registered agents.
        
        Returns:
            List of all agent instances
        """
        return list(self.agents.values())
    
    def get_agent_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all agents.
        
        Returns:
            Dictionary mapping agent types to their statistics
        """
        return {agent_type: agent.get_stats() for agent_type, agent in self.agents.items()}