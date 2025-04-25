"""
Dark Web data source module for the CIVILIAN system.
This module handles ingestion of content from Dark Web sites (.onion domains).
"""

import json
import logging
import threading
import time
import re
import socket
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse, urljoin

import requests
import socks
from stem import Signal
from stem.control import Controller
from stem.connection import MissingPassword, PasswordAuthFailed
from bs4 import BeautifulSoup
from sqlalchemy.exc import SQLAlchemyError

from app import db, app
from models import DataSource, NarrativeInstance, SystemLog

logger = logging.getLogger(__name__)

# Default Tor SOCKS ports and control port
DEFAULT_SOCKS_PORT = 9050
DEFAULT_CONTROL_PORT = 9051

class DarkWebSource:
    """Data source connector for Dark Web sites."""
    
    def __init__(self):
        """Initialize the Dark Web data source."""
        self._running = False
        self._thread = None
        self._tor_password = None
        self._session = None
        self._user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; rv:109.0) Gecko/20100101 Firefox/115.0",
            "Mozilla/5.0 (Windows NT 10.0; rv:102.0) Gecko/20100101 Firefox/102.0",
            "Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/115.0",
            "Mozilla/5.0 (X11; Linux x86_64; rv:102.0) Gecko/20100101 Firefox/102.0"
        ]
        
        # Try to initialize the Tor session
        try:
            self._initialize_session()
        except Exception as e:
            logger.warning(f"Tor session initialization failed: {e}")
        
        logger.info("DarkWebSource initialized")
    
    def _initialize_session(self):
        """Initialize a requests session with Tor as a proxy."""
        import os
        
        # Get Tor password if provided
        self._tor_password = os.environ.get('TOR_CONTROL_PASSWORD')
        
        # Create a session with SOCKS proxy settings for Tor
        self._session = requests.Session()
        self._session.proxies = {
            'http': f'socks5h://127.0.0.1:{DEFAULT_SOCKS_PORT}',
            'https': f'socks5h://127.0.0.1:{DEFAULT_SOCKS_PORT}'
        }
        
        # Test the Tor connection
        try:
            # Test if we can connect to check.torproject.org
            response = self._session.get("https://check.torproject.org", timeout=15)
            
            if "Congratulations. This browser is configured to use Tor." in response.text:
                logger.info("Tor connection successful")
                return
            else:
                raise ValueError("Tor connection test failed: Not using Tor network")
        
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Tor connection test failed: {e}")
    
    def _renew_tor_identity(self):
        """Request a new Tor identity (circuit)."""
        try:
            with Controller.from_port(port=DEFAULT_CONTROL_PORT) as controller:
                if self._tor_password:
                    controller.authenticate(password=self._tor_password)
                else:
                    # Try without password (may work in development environments)
                    try:
                        controller.authenticate()
                    except (MissingPassword, PasswordAuthFailed):
                        logger.warning("Tor control requires a password but none was provided")
                        return False
                
                # Send NEWNYM signal to get a new identity
                controller.signal(Signal.NEWNYM)
                logger.debug("Tor identity renewed")
                return True
        
        except Exception as e:
            logger.error(f"Error renewing Tor identity: {e}")
            return False
    
    def start(self):
        """Start monitoring Dark Web sites in a background thread."""
        if self._running:
            logger.warning("DarkWebSource monitoring already running")
            return
        
        if not self._session:
            logger.error("Tor session not initialized, cannot start monitoring")
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._run_monitoring_loop, daemon=True)
        self._thread.start()
        logger.info("DarkWebSource monitoring started")
    
    def stop(self):
        """Stop monitoring Dark Web sites."""
        if not self._running:
            logger.warning("DarkWebSource monitoring not running")
            return
        
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)  # Wait up to 3 seconds
            logger.info("DarkWebSource monitoring stopped")
    
    def _run_monitoring_loop(self):
        """Main monitoring loop that runs in background thread."""
        from config import Config
        
        while self._running:
            try:
                logger.debug("Starting Dark Web monitoring cycle")
                
                with app.app_context():
                    # Get active Dark Web sources from database
                    sources = self._get_active_sources()
                    
                    if not sources:
                        logger.debug("No active Dark Web sources defined")
                    else:
                        # Process each source
                        for source in sources:
                            if not self._running:
                                break
                            
                            try:
                                # Extract onion sites from config
                                config = json.loads(source.config) if source.config else {}
                                sites = config.get('sites', [])
                                
                                # Process each onion site
                                for site in sites:
                                    if not self._running:
                                        break
                                    
                                    site_url = site.get('url')
                                    site_type = site.get('type', 'forum')
                                    max_pages = site.get('max_pages', 5)
                                    
                                    if site_type == 'forum':
                                        self._scrape_forum(source.id, site_url, max_pages, site)
                                    elif site_type == 'market':
                                        self._scrape_market(source.id, site_url, max_pages, site)
                                    else:
                                        self._scrape_generic(source.id, site_url, max_pages, site)
                                    
                                    # Renew Tor identity periodically
                                    self._renew_tor_identity()
                                    
                                    # Sleep between sites to avoid overloading Tor
                                    time.sleep(random.uniform(5, 15))
                            
                            except Exception as e:
                                self._log_error("process_source", f"Error processing source {source.id}: {e}")
                        
                        # Update last ingestion time
                        self._update_ingestion_time(sources)
                
                # Sleep until next cycle
                for _ in range(int(Config.INGESTION_INTERVAL)):
                    if not self._running:
                        break
                    time.sleep(2)  # Check if still running every 2 seconds
            
            except Exception as e:
                with app.app_context():
                    self._log_error("monitoring_loop", f"Error in Dark Web monitoring loop: {e}")
                time.sleep(60)  # Shorter interval after error
    
    def _get_active_sources(self) -> List[DataSource]:
        """Get active Dark Web data sources from the database."""
        try:
            return DataSource.query.filter_by(
                source_type='darkweb',
                is_active=True
            ).all()
        except Exception as e:
            self._log_error("get_sources", f"Error fetching Dark Web sources: {e}")
            return []
    
    def _update_ingestion_time(self, sources: List[DataSource]):
        """Update the last ingestion timestamp for sources."""
        current_time = datetime.utcnow()
        try:
            for source in sources:
                source.last_ingestion = current_time
            db.session.commit()
        except Exception as e:
            self._log_error("update_ingestion_time", f"Error updating ingestion time: {e}")
            db.session.rollback()
    
    def _scrape_forum(self, source_id: int, forum_url: str, max_pages: int, site_config: Dict[str, Any]):
        """Scrape a Dark Web forum for content."""
        logger.debug(f"Scraping Dark Web forum: {forum_url}")
        
        try:
            base_url = forum_url.rstrip('/')
            
            # Apply site-specific scraping configuration
            thread_selector = site_config.get('thread_selector', 'a.thread-title')
            thread_link_attr = site_config.get('thread_link_attr', 'href')
            content_selector = site_config.get('content_selector', '.post-content')
            
            # Get forum index page
            response = self._fetch_url(base_url)
            if not response:
                logger.warning(f"Failed to fetch forum index: {base_url}")
                return
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find thread links
            thread_links = []
            for thread in soup.select(thread_selector):
                link = thread.get(thread_link_attr)
                if link:
                    # Create absolute URL
                    if not link.startswith('http'):
                        link = urljoin(base_url, link)
                    thread_links.append(link)
            
            # Limit the number of threads to process
            thread_links = thread_links[:min(len(thread_links), max_pages)]
            
            # Process each thread
            for thread_url in thread_links:
                if not self._running:
                    break
                
                # Get thread page
                thread_response = self._fetch_url(thread_url)
                if not thread_response:
                    logger.warning(f"Failed to fetch thread: {thread_url}")
                    continue
                
                thread_soup = BeautifulSoup(thread_response.text, 'html.parser')
                
                # Extract thread title
                title = thread_soup.title.text if thread_soup.title else "Unknown Thread"
                
                # Extract thread content
                posts = thread_soup.select(content_selector)
                content = "\n".join([post.get_text(strip=True) for post in posts])
                
                # Create a narrative instance for detection
                data = {
                    'title': title,
                    'url': thread_url,
                    'content': content,
                    'post_count': len(posts),
                    'site_type': 'forum',
                    'timestamp': datetime.utcnow().isoformat()
                }
                
                instance = NarrativeInstance(
                    source_id=source_id,
                    content=title + "\n" + content[:1000],  # First 1000 chars for analysis
                    meta_data=json.dumps(data),
                    url=thread_url
                )
                
                # Add to database
                db.session.add(instance)
                
                # Sleep between threads to avoid overloading the site
                time.sleep(random.uniform(3, 8))
            
            # Commit changes
            db.session.commit()
            logger.debug(f"Processed {len(thread_links)} threads from forum {forum_url}")
        
        except Exception as e:
            self._log_error("scrape_forum", f"Error scraping forum {forum_url}: {e}")
            db.session.rollback()
    
    def _scrape_market(self, source_id: int, market_url: str, max_pages: int, site_config: Dict[str, Any]):
        """Scrape a Dark Web marketplace for content."""
        logger.debug(f"Scraping Dark Web marketplace: {market_url}")
        
        try:
            base_url = market_url.rstrip('/')
            
            # Apply site-specific scraping configuration
            listing_selector = site_config.get('listing_selector', '.product-listing')
            listing_link_attr = site_config.get('listing_link_attr', 'href')
            title_selector = site_config.get('title_selector', '.product-title')
            description_selector = site_config.get('description_selector', '.product-description')
            price_selector = site_config.get('price_selector', '.product-price')
            
            # Get marketplace index page
            response = self._fetch_url(base_url)
            if not response:
                logger.warning(f"Failed to fetch marketplace index: {base_url}")
                return
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find listing links
            listing_links = []
            for listing in soup.select(listing_selector):
                link_element = listing.find('a')
                if link_element and link_element.get(listing_link_attr):
                    link = link_element.get(listing_link_attr)
                    # Create absolute URL
                    if not link.startswith('http'):
                        link = urljoin(base_url, link)
                    listing_links.append(link)
            
            # Limit the number of listings to process
            listing_links = listing_links[:min(len(listing_links), max_pages)]
            
            # Process each listing
            for listing_url in listing_links:
                if not self._running:
                    break
                
                # Get listing page
                listing_response = self._fetch_url(listing_url)
                if not listing_response:
                    logger.warning(f"Failed to fetch listing: {listing_url}")
                    continue
                
                listing_soup = BeautifulSoup(listing_response.text, 'html.parser')
                
                # Extract listing details
                title_elem = listing_soup.select_one(title_selector)
                title = title_elem.get_text(strip=True) if title_elem else "Unknown Listing"
                
                description_elem = listing_soup.select_one(description_selector)
                description = description_elem.get_text(strip=True) if description_elem else ""
                
                price_elem = listing_soup.select_one(price_selector)
                price = price_elem.get_text(strip=True) if price_elem else "Unknown Price"
                
                # Create a narrative instance for detection
                data = {
                    'title': title,
                    'description': description,
                    'price': price,
                    'url': listing_url,
                    'site_type': 'market',
                    'timestamp': datetime.utcnow().isoformat()
                }
                
                instance = NarrativeInstance(
                    source_id=source_id,
                    content=title + "\n" + description,
                    meta_data=json.dumps(data),
                    url=listing_url
                )
                
                # Add to database
                db.session.add(instance)
                
                # Sleep between listings to avoid overloading the site
                time.sleep(random.uniform(3, 8))
            
            # Commit changes
            db.session.commit()
            logger.debug(f"Processed {len(listing_links)} listings from marketplace {market_url}")
        
        except Exception as e:
            self._log_error("scrape_market", f"Error scraping marketplace {market_url}: {e}")
            db.session.rollback()
    
    def _scrape_generic(self, source_id: int, site_url: str, max_pages: int, site_config: Dict[str, Any]):
        """Scrape a generic Dark Web site for content."""
        logger.debug(f"Scraping generic Dark Web site: {site_url}")
        
        try:
            base_url = site_url.rstrip('/')
            
            # Apply site-specific scraping configuration
            content_selector = site_config.get('content_selector', 'body')
            link_selector = site_config.get('link_selector', 'a')
            exclude_patterns = site_config.get('exclude_patterns', [])
            
            # Track visited pages
            visited_urls = set()
            to_visit = [base_url]
            
            # Breadth-first crawl
            pages_processed = 0
            while to_visit and pages_processed < max_pages and self._running:
                # Get next URL to visit
                current_url = to_visit.pop(0)
                
                # Skip if already visited
                if current_url in visited_urls:
                    continue
                
                # Skip URLs matching exclude patterns
                if any(re.search(pattern, current_url) for pattern in exclude_patterns):
                    continue
                
                # Add to visited set
                visited_urls.add(current_url)
                
                # Fetch page
                response = self._fetch_url(current_url)
                if not response:
                    logger.warning(f"Failed to fetch page: {current_url}")
                    continue
                
                # Process page
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract content
                content_elem = soup.select_one(content_selector)
                content = content_elem.get_text(strip=True) if content_elem else ""
                
                # Create a narrative instance for detection
                data = {
                    'title': soup.title.text if soup.title else "Unknown Page",
                    'url': current_url,
                    'content': content,
                    'site_type': 'generic',
                    'timestamp': datetime.utcnow().isoformat()
                }
                
                instance = NarrativeInstance(
                    source_id=source_id,
                    content=content[:1000],  # First 1000 chars for analysis
                    meta_data=json.dumps(data),
                    url=current_url
                )
                
                # Add to database
                db.session.add(instance)
                pages_processed += 1
                
                # Find links to follow
                for link in soup.select(link_selector):
                    href = link.get('href')
                    if href:
                        # Skip external links, anchors, etc.
                        if href.startswith('#') or href.startswith('javascript:'):
                            continue
                        
                        # Create absolute URL
                        absolute_url = urljoin(base_url, href)
                        
                        # Only follow links on the same domain
                        if urlparse(absolute_url).netloc == urlparse(base_url).netloc:
                            if absolute_url not in visited_urls and absolute_url not in to_visit:
                                to_visit.append(absolute_url)
                
                # Sleep between pages to avoid overloading the site
                time.sleep(random.uniform(3, 8))
            
            # Commit changes
            db.session.commit()
            logger.debug(f"Processed {pages_processed} pages from site {site_url}")
        
        except Exception as e:
            self._log_error("scrape_generic", f"Error scraping site {site_url}: {e}")
            db.session.rollback()
    
    def _fetch_url(self, url: str, max_retries: int = 3, timeout: int = 30) -> Optional[requests.Response]:
        """Fetch a URL using Tor, with retries."""
        retries = 0
        while retries < max_retries:
            try:
                # Select a random user agent
                headers = {
                    'User-Agent': random.choice(self._user_agents)
                }
                
                # Make the request
                response = self._session.get(
                    url,
                    headers=headers,
                    timeout=timeout
                )
                
                # Check if successful
                if response.status_code == 200:
                    return response
                else:
                    logger.warning(f"Failed to fetch {url}: HTTP {response.status_code}")
            
            except (requests.exceptions.RequestException, socket.error) as e:
                logger.warning(f"Error fetching {url}: {e}")
            
            # Increment retry counter
            retries += 1
            
            # Renew Tor identity if we're going to retry
            if retries < max_retries:
                self._renew_tor_identity()
                # Wait before retrying
                time.sleep(random.uniform(5, 15))
        
        return None
    
    def create_source(self, name: str, config: Dict[str, Any]) -> Optional[int]:
        """Create a new Dark Web data source in the database.
        
        Args:
            name: Name for the data source
            config: Configuration dictionary with site parameters
            
        Returns:
            source_id: ID of the created source, or None on error
        """
        try:
            # Validate config
            if not isinstance(config, dict) or 'sites' not in config:
                logger.error("Invalid config: must contain 'sites' list")
                return None
            
            if not isinstance(config['sites'], list):
                logger.error("Invalid config: 'sites' must be a list")
                return None
            
            # Validate each site
            for site in config['sites']:
                if not isinstance(site, dict) or 'url' not in site:
                    logger.error("Invalid site config: must contain 'url'")
                    return None
                
                # Validate URL is an onion address
                url = site.get('url', '')
                if not url.endswith('.onion') and not url.startswith('http://') and not url.endswith('.onion/'):
                    logger.error(f"Invalid onion URL: {url}")
                    return None
            
            with app.app_context():
                # Create a new data source
                source = DataSource(
                    name=name,
                    source_type='darkweb',
                    config=json.dumps(config),
                    is_active=True,
                    created_at=datetime.utcnow()
                )
                
                # Add to database
                db.session.add(source)
                db.session.commit()
                
                logger.info(f"Created Dark Web source: {name} (ID: {source.id})")
                return source.id
        
        except Exception as e:
            with app.app_context():
                db.session.rollback()
                self._log_error("create_source", f"Error creating Dark Web source: {e}")
            return None
    
    def _log_error(self, operation: str, message: str):
        """Log an error to the database."""
        try:
            log = SystemLog(
                timestamp=datetime.utcnow(),
                log_type='error',
                component='darkweb_source',
                message=f"{operation}: {message}"
            )
            db.session.add(log)
            db.session.commit()
        except SQLAlchemyError:
            logger.error(f"Failed to log error to database: {message}")
            db.session.rollback()
        except Exception as e:
            logger.error(f"Error logging to database: {e}")