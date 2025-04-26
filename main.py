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
from data_sources.youtube_source import YouTubeSource
from data_sources.darkweb_source import DarkWebSource
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
youtube_source = None
darkweb_source = None
evidence_store = None
graph_store = None

def configure_open_sources():
    """Configure open data and news sources for the CIVILIAN system."""
    import json
    from pathlib import Path
    from models import DataSource
    from app import db
    
    def load_json_config(filepath):
        """Load a JSON configuration file."""
        try:
            if Path(filepath).exists():
                with open(filepath, 'r') as f:
                    return json.load(f)
            else:
                logger.warning(f"Config file not found: {filepath}")
                return None
        except Exception as e:
            logger.error(f"Error loading config file {filepath}: {e}")
            return None
    
    # Configure RSS news sources
    def add_rss_sources(sources_dict, prefix=""):
        """Add RSS sources from a dictionary of categories and feeds."""
        count = 0
        
        for category, feeds in sources_dict.items():
            logger.info(f"Adding {len(feeds)} {category} RSS feeds...")
            
            for feed in feeds:
                try:
                    # Create source name with optional prefix
                    name = f"{prefix}{feed['name']} ({feed['category']})"
                    
                    # Check if source already exists
                    existing = DataSource.query.filter_by(name=name).first()
                    if existing:
                        logger.info(f"Source '{name}' already exists, skipping.")
                        continue
                    
                    # Create config
                    config = {'feeds': [feed['url']]}
                    
                    # Create the source
                    source = DataSource(
                        name=name,
                        source_type='rss',
                        config=json.dumps(config),
                        is_active=True
                    )
                    
                    # Add to database
                    db.session.add(source)
                    count += 1
                
                except Exception as e:
                    logger.error(f"Error adding source {feed.get('name', 'unknown')}: {e}")
                    
        # Commit all changes
        if count > 0:
            try:
                db.session.commit()
                logger.info(f"Added {count} RSS sources")
            except Exception as e:
                logger.error(f"Error committing RSS sources: {e}")
                db.session.rollback()
        else:
            logger.info("No new RSS sources added")
    
    # Add Twitter monitoring sources
    def add_twitter_sources():
        """Add Twitter sources for monitoring misinformation."""
        # Define monitoring queries
        misinfo_queries = [
            {
                "name": "COVID-19 Misinformation Monitor",
                "queries": [
                    "covid hoax", "covid conspiracy",
                    "vaccine microchip", "5G covid", "pandemic fake"
                ]
            },
            {
                "name": "Climate Change Misinformation Monitor",
                "queries": [
                    "climate hoax", "climate change fake",
                    "global warming myth", "climate scientists lying"
                ]
            },
            {
                "name": "Election Misinformation Monitor",
                "queries": [
                    "election rigged", "election stolen",
                    "voter fraud widespread", "voting machines hacked"
                ]
            },
            {
                "name": "Fact-Check Accounts",
                "queries": [
                    "from:Snopes", "from:PolitiFact", 
                    "from:FactCheck.org", "from:APFactCheck", 
                    "from:FullFact"
                ]
            }
        ]
        
        count = 0
        for source_config in misinfo_queries:
            try:
                name = source_config["name"]
                
                # Check if source already exists
                existing = DataSource.query.filter_by(
                    name=name, 
                    source_type='twitter'
                ).first()
                
                if existing:
                    logger.info(f"Twitter source '{name}' already exists, skipping.")
                    continue
                
                # Create the source
                source = DataSource(
                    name=name,
                    source_type='twitter',
                    config=json.dumps({'queries': source_config["queries"]}),
                    is_active=True
                )
                
                # Add to database
                db.session.add(source)
                count += 1
            
            except Exception as e:
                logger.error(f"Error adding Twitter source {source_config.get('name', 'unknown')}: {e}")
        
        # Commit all changes
        if count > 0:
            try:
                db.session.commit()
                logger.info(f"Added {count} Twitter sources")
            except Exception as e:
                logger.error(f"Error committing Twitter sources: {e}")
                db.session.rollback()
        else:
            logger.info("No new Twitter sources added")
    
    try:
        # Load configuration files
        news_sources = load_json_config("config/open_news_sources.json")
        fact_check_sources = load_json_config("config/fact_check_sources.json")
        open_data_sources = load_json_config("config/open_data_sources.json")
        specialized_sources = load_json_config("config/specialized_monitoring.json")
        advanced_data_sources = load_json_config("config/advanced_data_sources.json")
        
        # Add sources to database with rate limiting
        import time
        
        # Optimized batch-add function
        def batch_add_sources(sources_dict, source_type, prefix="", batch_size=10):
            """Add sources in batches to improve performance."""
            total_count = 0
            current_batch = []
            
            # Flatten the sources list
            all_sources = []
            for category, items in sources_dict.items():
                for item in items:
                    item['_category'] = category  # Store category for reference
                    all_sources.append(item)
            
            # Process in batches
            for i, source in enumerate(all_sources):
                try:
                    # Create source name with optional prefix
                    name = f"{prefix}{source['name']} ({source['category']})"
                    
                    # Check if source already exists
                    existing = DataSource.query.filter_by(name=name).first()
                    if existing:
                        logger.info(f"Source '{name}' already exists, skipping.")
                        continue
                    
                    # Create the source configuration based on type
                    if source_type == 'rss':
                        config = {'feeds': [source['url']]}
                        if 'description' in source:
                            config['description'] = source['description']
                    else:
                        config = source.get('config', {})
                    
                    # Create the source
                    new_source = DataSource(
                        name=name,
                        source_type=source_type,
                        config=json.dumps(config),
                        is_active=True
                    )
                    
                    # Add to database
                    db.session.add(new_source)
                    current_batch.append(name)
                    total_count += 1
                    
                    # Commit when batch size reached or at the end
                    if len(current_batch) >= batch_size or i == len(all_sources) - 1:
                        try:
                            db.session.commit()
                            logger.info(f"Added batch of {len(current_batch)} {source_type} sources")
                            current_batch = []
                            
                            # Add a short delay to prevent rate limiting on database
                            time.sleep(0.1)
                        except Exception as e:
                            logger.error(f"Error committing batch: {e}")
                            db.session.rollback()
                
                except Exception as e:
                    logger.error(f"Error adding source {source.get('name', 'unknown')}: {e}")
            
            return total_count
        
        # Add core news sources
        sources_added = 0
        if news_sources:
            logger.info("Adding open news sources...")
            sources_added += batch_add_sources(news_sources, 'rss')
        
        # Add fact checking sources
        if fact_check_sources:
            logger.info("Adding fact-checking sources...")
            sources_added += batch_add_sources(fact_check_sources, 'rss', "FactCheck: ")
        
        # Add open data sources
        if open_data_sources:
            logger.info("Adding open data sources...")
            sources_added += batch_add_sources(open_data_sources, 'rss', "OpenData: ")
        
        # Add specialized monitoring sources
        if specialized_sources:
            logger.info("Adding specialized monitoring sources...")
            # Add each specialized category with a specific prefix
            for category in specialized_sources:
                if category == "security_intelligence":
                    sources_added += batch_add_sources({category: specialized_sources[category]}, 'rss', "Security: ")
                elif category == "academic_research":
                    sources_added += batch_add_sources({category: specialized_sources[category]}, 'rss', "Research: ")
                elif category == "misinformation_tracking":
                    sources_added += batch_add_sources({category: specialized_sources[category]}, 'rss', "MisInfo: ")
                elif category == "multilingual_sources":
                    sources_added += batch_add_sources({category: specialized_sources[category]}, 'rss', "Lang: ")
                else:
                    sources_added += batch_add_sources({category: specialized_sources[category]}, 'rss')
        
        # Add advanced data sources
        if advanced_data_sources:
            logger.info("Adding advanced data sources...")
            # Add each advanced category with a specific prefix
            for category in advanced_data_sources:
                if category == "media_monitoring":
                    sources_added += batch_add_sources({category: advanced_data_sources[category]}, 'rss', "Media: ")
                elif category == "scientific_datasets":
                    sources_added += batch_add_sources({category: advanced_data_sources[category]}, 'rss', "Dataset: ")
                elif category == "specialized_databases":
                    sources_added += batch_add_sources({category: advanced_data_sources[category]}, 'rss', "Database: ")
                elif category == "elections_monitoring":
                    sources_added += batch_add_sources({category: advanced_data_sources[category]}, 'rss', "Elections: ")
                else:
                    sources_added += batch_add_sources({category: advanced_data_sources[category]}, 'rss', "Advanced: ")
        
        logger.info(f"Total RSS sources added: {sources_added}")
        
        # Add Twitter sources
        add_twitter_sources()
        
        # Add advanced Telegram monitoring
        try:
            # Define focused Telegram entities to monitor
            focused_telegram_entities = [
                {
                    "name": "Information Security Monitoring",
                    "entities": [
                        "@TheHackersNews",
                        "@CISAgov",
                        "@MalwareHunterTeam",
                        "@vxunderground"
                    ]
                },
                {
                    "name": "Health & Science Monitoring",
                    "entities": [
                        "@WHO",
                        "@CDCgov",
                        "@NatureNews",
                        "@ScienceNews"
                    ]
                },
                {
                    "name": "Global Politics Monitoring",
                    "entities": [
                        "@UN",
                        "@POTUS",
                        "@Politico",
                        "@BBCBreaking"
                    ]
                }
            ]
            
            # Add to database in batch mode (if credentials are present)
            count = 0
            for source_config in focused_telegram_entities:
                name = source_config["name"]
                
                # Check if source already exists
                existing = DataSource.query.filter_by(
                    name=name, 
                    source_type='telegram'
                ).first()
                
                if existing:
                    logger.info(f"Telegram source '{name}' already exists, skipping.")
                    continue
                
                # Create the source
                source = DataSource(
                    name=name,
                    source_type='telegram',
                    config=json.dumps({'entities': source_config["entities"]}),
                    is_active=True
                )
                
                # Add to database
                db.session.add(source)
                count += 1
            
            # Commit all changes
            if count > 0:
                db.session.commit()
                logger.info(f"Added {count} Telegram sources")
        except Exception as e:
            logger.error(f"Error setting up Telegram sources: {e}")
            db.session.rollback()
        
        logger.info("Open sources configuration completed")
    
    except Exception as e:
        logger.error(f"Error configuring open sources: {e}")

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
    youtube_source = YouTubeSource()
    darkweb_source = DarkWebSource()
    
    # Configure open sources
    configure_open_sources()
    
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
    youtube_source.start()
    darkweb_source.start()
    
    logger.info("CIVILIAN application initialized")

# All blueprints are registered in app.py now

# Initialize the app when imported
with app.app_context():
    initialize_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
