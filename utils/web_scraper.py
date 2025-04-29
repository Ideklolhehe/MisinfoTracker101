import os
import logging
import urllib.parse
import requests
import json
import re
import time
import concurrent.futures
from typing import Dict, List, Optional, Any, Tuple, Set
from bs4 import BeautifulSoup
import feedparser
import trafilatura
from datetime import datetime

logger = logging.getLogger(__name__)

class WebScraper:
    """Utility class for scraping web content."""
    
    def __init__(self):
        """Initialize the web scraper."""
        self.headers = {
            'User-Agent': 'CIVILIAN Web Scraper/1.0 (Research; Analysis)',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        self.request_delay = 1.0  # Delay between requests in seconds
        self.last_request_time = 0
    
    def _rate_limit(self):
        """Apply rate limiting to prevent overwhelming servers."""
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        
        if elapsed < self.request_delay:
            time.sleep(self.request_delay - elapsed)
        
        self.last_request_time = time.time()
    
    def scrape_url(self, url: str, extract_links: bool = False) -> Dict:
        """
        Scrape content from a URL.
        
        Args:
            url: URL to scrape
            extract_links: Whether to extract links from the page
            
        Returns:
            Dict containing the scraped content
        """
        logger.info(f"Scraping URL: {url}")
        self._rate_limit()
        
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            html_content = response.text
            
            # Extract content using trafilatura for better text extraction
            content = trafilatura.extract(html_content)
            
            # Use BeautifulSoup for additional metadata and link extraction
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Get the title
            title = soup.title.string if soup.title else ''
            
            # Extract links if requested
            links = []
            if extract_links:
                links = self._extract_links(soup, url)
            
            # Get metadata
            metadata = self._extract_metadata(soup)
            
            result = {
                'url': url,
                'title': title.strip() if title else '',
                'content': content or '',
                'html': html_content,
                'headers': dict(response.headers),
                'links': links,
                'metadata': metadata,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            return result
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error scraping URL {url}: {str(e)}")
            raise
    
    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> List[Dict]:
        """Extract links from a BeautifulSoup object."""
        links = []
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            
            # Skip empty links and non-HTTP links
            if not href or href.startswith('#') or href.startswith('javascript:'):
                continue
            
            # Convert relative URLs to absolute
            absolute_url = urllib.parse.urljoin(base_url, href)
            
            # Get link text
            link_text = a_tag.get_text().strip()
            
            links.append({
                'url': absolute_url,
                'text': link_text if link_text else None,
                'rel': a_tag.get('rel', None),
                'title': a_tag.get('title', None)
            })
        
        return links
    
    def _extract_metadata(self, soup: BeautifulSoup) -> Dict:
        """Extract metadata from a BeautifulSoup object."""
        metadata = {}
        
        # Extract meta tags
        for meta in soup.find_all('meta'):
            name = meta.get('name') or meta.get('property')
            content = meta.get('content')
            
            if name and content:
                metadata[name] = content
        
        # Extract schema.org structured data
        for script in soup.find_all('script', {'type': 'application/ld+json'}):
            try:
                json_data = json.loads(script.string)
                metadata['structured_data'] = metadata.get('structured_data', []) + [json_data]
            except (json.JSONDecodeError, TypeError):
                pass
        
        return metadata
    
    def crawl(self, start_url: str, max_pages: int = 5, same_domain_only: bool = True) -> Dict[str, Dict]:
        """
        Crawl a website starting from a URL.
        
        Args:
            start_url: Starting URL for the crawl
            max_pages: Maximum number of pages to crawl
            same_domain_only: Whether to only crawl pages on the same domain
            
        Returns:
            Dict of URL -> content mappings
        """
        logger.info(f"Starting crawl from {start_url}, max_pages={max_pages}")
        
        # Parse the start URL to get the domain
        parsed_url = urllib.parse.urlparse(start_url)
        base_domain = parsed_url.netloc
        
        # Set up tracking variables
        visited_urls = set()
        to_visit = [start_url]
        results = {}
        
        while to_visit and len(results) < max_pages:
            current_url = to_visit.pop(0)
            
            if current_url in visited_urls:
                continue
            
            visited_urls.add(current_url)
            
            try:
                # Scrape the current URL
                content = self.scrape_url(current_url, extract_links=True)
                results[current_url] = content
                
                # Add links to the to_visit list
                for link in content.get('links', []):
                    link_url = link.get('url')
                    
                    if not link_url:
                        continue
                    
                    # Skip if we've already visited or queued this URL
                    if link_url in visited_urls or link_url in to_visit:
                        continue
                    
                    # Skip if not on the same domain and same_domain_only is True
                    if same_domain_only:
                        link_domain = urllib.parse.urlparse(link_url).netloc
                        if link_domain != base_domain:
                            continue
                    
                    # Add to the to_visit list
                    to_visit.append(link_url)
                
            except Exception as e:
                logger.error(f"Error crawling {current_url}: {str(e)}")
        
        logger.info(f"Crawling completed: {len(results)} pages crawled")
        return results
    
    def search(self, query: str, engine: str = 'bing', limit: int = 10) -> List[Dict]:
        """
        Search the web for a query.
        
        Args:
            query: Search query
            engine: Search engine to use ('bing', 'google', 'duckduckgo')
            limit: Maximum number of results to return
            
        Returns:
            List of search result dicts
        """
        logger.info(f"Searching for '{query}' using {engine}, limit={limit}")
        
        if engine.lower() == 'bing':
            return self._search_bing(query, limit)
        elif engine.lower() == 'google':
            return self._search_google(query, limit)
        elif engine.lower() == 'duckduckgo':
            return self._search_duckduckgo(query, limit)
        else:
            raise ValueError(f"Unsupported search engine: {engine}")
    
    def _search_bing(self, query: str, limit: int = 10) -> List[Dict]:
        """Search using Bing."""
        # This is a simplified implementation that scrapes the results page
        # In a production environment, you would use the Bing API
        
        url = f"https://www.bing.com/search?q={urllib.parse.quote(query)}"
        
        try:
            self._rate_limit()
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            results = []
            
            # Extract search results
            for result in soup.select('.b_algo')[:limit]:
                title_elem = result.select_one('h2 a')
                if not title_elem:
                    continue
                
                title = title_elem.get_text().strip()
                link = title_elem.get('href', '')
                
                # Get the snippet
                snippet_elem = result.select_one('.b_caption p')
                snippet = snippet_elem.get_text().strip() if snippet_elem else ''
                
                results.append({
                    'title': title,
                    'url': link,
                    'snippet': snippet,
                    'source': 'bing',
                    'date': ''  # Bing doesn't consistently provide dates in the search results
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching Bing for '{query}': {str(e)}")
            return []
    
    def _search_google(self, query: str, limit: int = 10) -> List[Dict]:
        """Search using Google."""
        # This is a simplified implementation that scrapes the results page
        # In a production environment, you would use the Google Search API
        
        url = f"https://www.google.com/search?q={urllib.parse.quote(query)}"
        
        try:
            self._rate_limit()
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            results = []
            
            # Extract search results
            for result in soup.select('.g')[:limit]:
                title_elem = result.select_one('h3')
                if not title_elem:
                    continue
                
                title = title_elem.get_text().strip()
                
                link_elem = result.select_one('a')
                link = link_elem.get('href', '') if link_elem else ''
                
                # Clean up the link (Google prefixes links with /url?q=)
                if link.startswith('/url?q='):
                    link = link.split('/url?q=')[1].split('&')[0]
                    link = urllib.parse.unquote(link)
                
                # Get the snippet
                snippet_elem = result.select_one('.VwiC3b')
                snippet = snippet_elem.get_text().strip() if snippet_elem else ''
                
                # Get date if available
                date = ''
                date_elem = result.select_one('.MUxGbd.wuQ4Ob.WZ8Tjf')
                if date_elem:
                    date_text = date_elem.get_text().strip()
                    date_match = re.search(r'\d{1,2} [A-Za-z]{3} \d{4}', date_text)
                    if date_match:
                        date = date_match.group(0)
                
                results.append({
                    'title': title,
                    'url': link,
                    'snippet': snippet,
                    'source': 'google',
                    'date': date
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching Google for '{query}': {str(e)}")
            return []
    
    def _search_duckduckgo(self, query: str, limit: int = 10) -> List[Dict]:
        """Search using DuckDuckGo."""
        # This is a simplified implementation using the HTML version
        
        url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
        
        try:
            self._rate_limit()
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            results = []
            
            # Extract search results
            for result in soup.select('.result')[:limit]:
                title_elem = result.select_one('.result__title a')
                if not title_elem:
                    continue
                
                title = title_elem.get_text().strip()
                link = title_elem.get('href', '')
                
                # DuckDuckGo uses redirects, extract the actual URL
                if link.startswith('/'):
                    parsed = urllib.parse.urlparse(link)
                    query_params = urllib.parse.parse_qs(parsed.query)
                    if 'uddg' in query_params:
                        link = query_params['uddg'][0]
                    
                # Get the snippet
                snippet_elem = result.select_one('.result__snippet')
                snippet = snippet_elem.get_text().strip() if snippet_elem else ''
                
                results.append({
                    'title': title,
                    'url': link,
                    'snippet': snippet,
                    'source': 'duckduckgo',
                    'date': ''  # DuckDuckGo doesn't consistently provide dates
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching DuckDuckGo for '{query}': {str(e)}")
            return []
    
    def parse_rss(self, feed_url: str) -> List[Dict]:
        """
        Parse an RSS feed.
        
        Args:
            feed_url: URL of the RSS feed
            
        Returns:
            List of feed entries as dictionaries
        """
        logger.info(f"Parsing RSS feed: {feed_url}")
        
        try:
            self._rate_limit()
            feed = feedparser.parse(feed_url)
            
            entries = []
            for entry in feed.entries:
                # Extract relevant fields
                entry_dict = {
                    'title': entry.get('title', ''),
                    'link': entry.get('link', ''),
                    'summary': entry.get('summary', ''),
                    'published': entry.get('published', ''),
                    'published_parsed': entry.get('published_parsed'),
                    'author': entry.get('author', ''),
                    'tags': [tag.get('term', '') for tag in entry.get('tags', [])]
                }
                
                # If content is available, add it
                if 'content' in entry:
                    # Some feeds have multiple content elements, use the first one
                    entry_dict['content'] = entry.content[0].value if entry.content else ''
                
                entries.append(entry_dict)
            
            return entries
            
        except Exception as e:
            logger.error(f"Error parsing RSS feed {feed_url}: {str(e)}")
            return []