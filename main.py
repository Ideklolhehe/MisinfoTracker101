from app import app  # noqa: F401
import logging
import os
from utils.text_processor import TextProcessor
from utils.vector_store import VectorStore
from utils.ai_processor import AIProcessor
from utils.web_scraper import WebScraper
from agents.detector_agent import DetectorAgent
from agents.analyzer_agent import AnalyzerAgent
from agents.counter_agent import CounterAgent
from data_sources.twitter_source import TwitterSource
from data_sources.telegram_source import TelegramSource
from data_sources.rss_source import RSSSource
from storage.evidence_store import EvidenceStore
from storage.graph_store import GraphStore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('CIVILIAN')

# Application components
text_processor = None
vector_store = None
ai_processor = None
web_scraper = None
detector_agent = None
analyzer_agent = None
counter_agent = None
twitter_source = None
telegram_source = None
rss_source = None
evidence_store = None
graph_store = None

def initialize_app():
    """Initialize application components."""
    global text_processor, vector_store, ai_processor, web_scraper
    global detector_agent, analyzer_agent, counter_agent
    global twitter_source, telegram_source, rss_source
    global evidence_store, graph_store
    
    # Check if already initialized
    if detector_agent is not None:
        return
        
    # Initialize utility components
    text_processor = TextProcessor()
    vector_store = VectorStore()
    ai_processor = AIProcessor()
    web_scraper = WebScraper()
    
    # Initialize storage components
    evidence_store = EvidenceStore()
    graph_store = GraphStore()
    
    # Initialize agent components
    detector_agent = DetectorAgent(text_processor, vector_store)
    analyzer_agent = AnalyzerAgent(text_processor)
    counter_agent = CounterAgent(text_processor)
    
    # Initialize data sources
    twitter_source = TwitterSource()
    telegram_source = TelegramSource()
    rss_source = RSSSource()
    
    # Check for API keys and display appropriate messages
    if not ai_processor.openai_available and not ai_processor.anthropic_available:
        logger.warning("Advanced AI capabilities are unavailable. Set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variables to enable them.")
    
    # Start background agents
    detector_agent.start()
    analyzer_agent.start()
    counter_agent.start()
    
    # Start data sources
    twitter_source.start()
    telegram_source.start()
    rss_source.start()
    
    logger.info("CIVILIAN application initialized")

# Initialize the app when imported
with app.app_context():
    initialize_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
