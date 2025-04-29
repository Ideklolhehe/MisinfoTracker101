import os
import json
import logging
import datetime
import threading
import time
from typing import Dict, List, Optional, Any, Union

from app import db, app
from models import WebSource, WebSourceJob, WebSourceJobStatus, ContentItem, DetectedNarrative
from utils.web_scraper import WebScraper
from utils.text_processor import TextProcessor
from data_sources.web_source_manager import WebSourceManager
from utils.concurrency import run_in_thread

logger = logging.getLogger(__name__)

class WebScrapingService:
    """Service for managing web scraping operations."""
    
    def __init__(self):
        """Initialize web scraping service."""
        self.web_scraper = WebScraper()
        self.web_source_manager = WebSourceManager()
        self.text_processor = TextProcessor()
        self.job_queue = []  # Queue for job IDs to be processed
        self.processing_thread = None
        self.stop_requested = False
        
        # Start the worker thread for processing jobs
        self._start_worker_thread()
        
    def _start_worker_thread(self):
        """Start a worker thread to process jobs."""
        self.stop_requested = False
        self.processing_thread = threading.Thread(
            target=self._process_job_queue,
            daemon=True
        )
        self.processing_thread.start()
        logger.info("Web scraping job processing thread started")
        
    def _process_job_queue(self):
        """Process jobs from the queue in a background thread."""
        # Use application context to ensure database access works
        with app.app_context():
            while not self.stop_requested:
                if self.job_queue:
                    job_id = self.job_queue.pop(0)
                    try:
                        self._process_job(job_id)
                    except Exception as e:
                        logger.error(f"Error processing job {job_id}: {str(e)}")
                        self._update_job_status(job_id, WebSourceJobStatus.FAILED.value, error_message=str(e))
                else:
                    # Sleep a short time to prevent CPU spinning
                    time.sleep(0.5)
    
    def _process_job(self, job_id: int):
        """Process a job based on its type."""
        job = WebSourceJob.query.get(job_id)
        
        if not job:
            logger.error(f"Job with ID {job_id} not found")
            return
        
        # Update job status to running
        self._update_job_status(job_id, WebSourceJobStatus.RUNNING.value)
        
        if job.source_id:
            # This is a job for a specific source
            source = WebSource.query.get(job.source_id)
            if not source:
                self._update_job_status(job_id, WebSourceJobStatus.FAILED.value, 
                                        error_message="Source not found")
                return
                
            # Process based on source type
            if source.source_type == 'web_page':
                result = self._process_web_page_job(job, source)
            elif source.source_type == 'web_crawl':
                result = self._process_web_crawl_job(job, source)
            elif source.source_type == 'web_search':
                result = self._process_web_search_job(job, source)
            elif source.source_type == 'rss':
                result = self._process_rss_job(job, source)
            else:
                self._update_job_status(job_id, WebSourceJobStatus.FAILED.value, 
                                        error_message=f"Unsupported source type: {source.source_type}")
                return
        else:
            # This is a standalone job (scan or search)
            if job.job_type == 'scan':
                result = self.process_scan_job(job_id)
            elif job.job_type == 'search':
                result = self.process_search_job(job_id)
            else:
                self._update_job_status(job_id, WebSourceJobStatus.FAILED.value, 
                                        error_message=f"Unsupported job type: {job.job_type}")
                return
        
        # If we got here and the job status isn't already set, mark it as complete
        if job.status != WebSourceJobStatus.FAILED.value:
            self._update_job_status(job_id, WebSourceJobStatus.COMPLETED.value, results=result)
        
        return result
    
    def _update_job_status(self, job_id: int, status: str, 
                           error_message: Optional[str] = None, 
                           results: Optional[Dict] = None):
        """Update the status of a job."""
        job = WebSourceJob.query.get(job_id)
        if not job:
            logger.error(f"Job with ID {job_id} not found")
            return
        
        job.status = status
        
        if status == WebSourceJobStatus.RUNNING.value:
            job.started_at = datetime.datetime.utcnow()
        
        if status in [WebSourceJobStatus.COMPLETED.value, WebSourceJobStatus.FAILED.value]:
            job.completed_at = datetime.datetime.utcnow()
        
        if error_message:
            job.error_message = error_message
        
        if results:
            job.set_results(results)
        
        try:
            db.session.commit()
        except Exception as e:
            logger.error(f"Error updating job status: {str(e)}")
            db.session.rollback()
    
    def queue_job(self, job_id: int) -> bool:
        """Add a job to the processing queue."""
        self.job_queue.append(job_id)
        logger.info(f"Job {job_id} added to processing queue")
        return True
    
    def process_scan_job(self, job_id: int) -> Dict:
        """Process a web page scan job."""
        job = WebSourceJob.query.get(job_id)
        if not job:
            raise ValueError(f"Job with ID {job_id} not found")
        
        metadata = job.get_meta_data()
        url = metadata.get('url')
        extract_links = metadata.get('extract_links', False)
        extract_content = metadata.get('extract_content', True)
        analyze_credibility = metadata.get('analyze_credibility', False)
        
        if not url:
            raise ValueError("URL is required")
        
        try:
            # Get content using the web scraper
            content = self.web_scraper.scrape_url(url, extract_links=extract_links)
            
            # Process the content based on options
            result = {
                'url': url,
                'title': content.get('title', ''),
                'timestamp': datetime.datetime.utcnow().isoformat(),
            }
            
            if extract_content:
                result['content'] = content.get('content', '')
                result['content_length'] = len(content.get('content', ''))
            
            if extract_links and 'links' in content:
                result['links'] = content.get('links', [])
                result['link_count'] = len(content.get('links', []))
            
            if analyze_credibility:
                credibility_score = self._analyze_credibility(url, content)
                result['credibility'] = credibility_score
                
            # Store the content in the database
            self._store_content(url, content, job.id)
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing scan job {job_id}: {str(e)}")
            raise
    
    def process_search_job(self, job_id: int) -> Dict:
        """Process a web search job."""
        job = WebSourceJob.query.get(job_id)
        if not job:
            raise ValueError(f"Job with ID {job_id} not found")
        
        metadata = job.get_meta_data()
        query = metadata.get('query')
        search_engine = metadata.get('search_engine', 'bing')
        limit = metadata.get('limit', 10)
        
        if not query:
            raise ValueError("Search query is required")
        
        try:
            # Perform the search using the web scraper
            search_results = self.web_scraper.search(query, engine=search_engine, limit=limit)
            
            # Process the results
            results = []
            for result in search_results:
                results.append({
                    'title': result.get('title', ''),
                    'url': result.get('url', ''),
                    'snippet': result.get('snippet', ''),
                    'source': result.get('source', ''),
                    'date': result.get('date', '')
                })
                
                # Store the search result in the database
                self._store_search_result(result, job.id)
            
            return {
                'query': query,
                'search_engine': search_engine,
                'result_count': len(results),
                'results': results,
                'timestamp': datetime.datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error processing search job {job_id}: {str(e)}")
            raise
    
    def _process_web_page_job(self, job: WebSourceJob, source: WebSource) -> Dict:
        """Process a web page source job."""
        source_config = source.get_config()
        extract_links = source_config.get('extract_links', False)
        
        # Create a metadata dict for the scan job
        metadata = {
            'url': source.url,
            'extract_links': extract_links,
            'extract_content': True,
            'analyze_credibility': False
        }
        
        # Update job metadata with source information
        job_meta = job.get_meta_data() or {}
        job_meta.update(metadata)
        job.set_meta_data(job_meta)
        db.session.commit()
        
        # Reuse the scan job processing
        result = self.process_scan_job(job.id)
        
        # Update the source's last ingestion timestamp
        source.last_ingestion = datetime.datetime.utcnow()
        db.session.commit()
        
        return result
    
    def _process_web_crawl_job(self, job: WebSourceJob, source: WebSource) -> Dict:
        """Process a web crawl source job."""
        source_config = source.get_config()
        max_pages = source_config.get('max_pages', 5)
        same_domain_only = source_config.get('same_domain_only', True)
        
        try:
            # Crawl the website
            crawl_results = self.web_scraper.crawl(
                start_url=source.url,
                max_pages=max_pages,
                same_domain_only=same_domain_only
            )
            
            # Store each page in the database
            processed_pages = []
            for page_url, content in crawl_results.items():
                self._store_content(page_url, content, job.id)
                
                processed_pages.append({
                    'url': page_url,
                    'title': content.get('title', ''),
                    'content_length': len(content.get('content', '')),
                    'links': len(content.get('links', []))
                })
            
            # Update the source's last ingestion timestamp
            source.last_ingestion = datetime.datetime.utcnow()
            db.session.commit()
            
            # Return the results
            return {
                'start_url': source.url,
                'pages_crawled': len(crawl_results),
                'max_pages': max_pages,
                'same_domain_only': same_domain_only,
                'processed_pages': processed_pages,
                'timestamp': datetime.datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error processing web crawl job for source {source.id}: {str(e)}")
            raise
    
    def _process_web_search_job(self, job: WebSourceJob, source: WebSource) -> Dict:
        """Process a web search source job."""
        source_config = source.get_config()
        search_term = source_config.get('search_term', '')
        search_engine = source_config.get('search_engine', 'bing')
        limit = source_config.get('limit', 10)
        
        # If no search term is provided, use the domain from the URL
        if not search_term:
            from urllib.parse import urlparse
            parsed_url = urlparse(source.url)
            search_term = parsed_url.netloc
        
        # Create a metadata dict for the search job
        metadata = {
            'query': search_term,
            'search_engine': search_engine,
            'limit': limit
        }
        
        # Update job metadata with source information
        job_meta = job.get_meta_data() or {}
        job_meta.update(metadata)
        job.set_meta_data(job_meta)
        db.session.commit()
        
        # Reuse the search job processing
        result = self.process_search_job(job.id)
        
        # Update the source's last ingestion timestamp
        source.last_ingestion = datetime.datetime.utcnow()
        db.session.commit()
        
        return result
    
    def _process_rss_job(self, job: WebSourceJob, source: WebSource) -> Dict:
        """Process an RSS feed source job."""
        source_config = source.get_config()
        
        try:
            # Process the RSS feed
            feed_entries = self.web_scraper.parse_rss(source.url)
            
            # Store each entry in the database
            processed_entries = []
            for entry in feed_entries:
                self._store_rss_entry(entry, job.id, source.id)
                
                processed_entries.append({
                    'title': entry.get('title', ''),
                    'link': entry.get('link', ''),
                    'published': entry.get('published', ''),
                    'summary_length': len(entry.get('summary', ''))
                })
            
            # Update the source's last ingestion timestamp
            source.last_ingestion = datetime.datetime.utcnow()
            db.session.commit()
            
            # Return the results
            return {
                'feed_url': source.url,
                'entries_processed': len(feed_entries),
                'entries': processed_entries,
                'timestamp': datetime.datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error processing RSS job for source {source.id}: {str(e)}")
            raise
    
    def _store_content(self, url: str, content: Dict, job_id: int) -> None:
        """Store web content in the database."""
        try:
            # Create a content item
            content_item = ContentItem(
                title=content.get('title', ''),
                content=content.get('content', ''),
                source='web_scraping',
                url=url,
                published_date=datetime.datetime.utcnow(),
                content_type='web_page',
                is_processed=False
            )
            
            # Set metadata with job information
            metadata = {
                'job_id': job_id,
                'headers': content.get('headers', {}),
                'links': content.get('links', []),
                'html_content': content.get('html', '')
            }
            content_item.set_meta_data(metadata)
            
            db.session.add(content_item)
            db.session.commit()
            
            logger.info(f"Stored web content for URL: {url}")
            
        except Exception as e:
            logger.error(f"Error storing web content: {str(e)}")
            db.session.rollback()
    
    def _store_search_result(self, result: Dict, job_id: int) -> None:
        """Store search result in the database."""
        try:
            # Create a content item
            content_item = ContentItem(
                title=result.get('title', ''),
                content=result.get('snippet', ''),
                source='web_search',
                url=result.get('url', ''),
                published_date=datetime.datetime.now(),
                content_type='search_result',
                is_processed=False
            )
            
            # Set metadata with job information
            metadata = {
                'job_id': job_id,
                'source': result.get('source', ''),
                'date': result.get('date', '')
            }
            content_item.set_meta_data(metadata)
            
            db.session.add(content_item)
            db.session.commit()
            
            logger.info(f"Stored search result for URL: {result.get('url', '')}")
            
        except Exception as e:
            logger.error(f"Error storing search result: {str(e)}")
            db.session.rollback()
    
    def _store_rss_entry(self, entry: Dict, job_id: int, source_id: int) -> None:
        """Store RSS entry in the database."""
        try:
            # Create a content item
            content_item = ContentItem(
                title=entry.get('title', ''),
                content=entry.get('summary', ''),
                source='rss',
                url=entry.get('link', ''),
                published_date=entry.get('published_parsed') or datetime.datetime.utcnow(),
                content_type='rss_entry',
                is_processed=False
            )
            
            # Set metadata with job information
            metadata = {
                'job_id': job_id,
                'source_id': source_id,
                'author': entry.get('author', ''),
                'categories': entry.get('tags', [])
            }
            content_item.set_meta_data(metadata)
            
            db.session.add(content_item)
            db.session.commit()
            
            logger.info(f"Stored RSS entry for URL: {entry.get('link', '')}")
            
        except Exception as e:
            logger.error(f"Error storing RSS entry: {str(e)}")
            db.session.rollback()
    
    def _analyze_credibility(self, url: str, content: Dict) -> Dict:
        """Analyze the credibility of a web page."""
        # Implement credibility analysis using heuristics, NLP, and other techniques
        # This is a placeholder that returns a basic analysis
        
        # Domain-based checks
        from urllib.parse import urlparse
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        
        # Simplified analysis (in a real implementation, this would be more sophisticated)
        metrics = {
            'domain_credibility': 0.0,
            'content_quality': 0.0,
            'information_diversity': 0.0,
            'source_citations': 0.0,
            'overall_score': 0.0
        }
        
        # Check for known credible domains
        credible_domains = [
            'reuters.com', 'apnews.com', 'bbc.co.uk', 'bbc.com',
            'nytimes.com', 'wsj.com', 'washingtonpost.com',
            'economist.com', 'nature.com', 'science.org',
            'nasa.gov', 'nih.gov', 'cdc.gov', 'who.int'
        ]
        
        for credible_domain in credible_domains:
            if credible_domain in domain:
                metrics['domain_credibility'] = 0.9
                break
        else:
            # If not in the list, use some heuristics
            if '.gov' in domain:
                metrics['domain_credibility'] = 0.85
            elif '.edu' in domain:
                metrics['domain_credibility'] = 0.8
            elif '.org' in domain:
                metrics['domain_credibility'] = 0.7
            else:
                # Default for unknown domains
                metrics['domain_credibility'] = 0.5
        
        # Content quality (this would use NLP in a real implementation)
        content_text = content.get('content', '')
        if len(content_text) > 1000:  # Longer articles tend to be more detailed
            metrics['content_quality'] = 0.7
        else:
            metrics['content_quality'] = 0.5
        
        # Information diversity (count unique words/concepts)
        words = content_text.lower().split()
        unique_words = len(set(words))
        if len(words) > 0:
            diversity = unique_words / len(words)
            metrics['information_diversity'] = min(diversity * 2, 1.0)
        
        # Source citations (count links and references)
        links = content.get('links', [])
        if len(links) > 5:  # More citations is generally better
            metrics['source_citations'] = 0.8
        elif len(links) > 0:
            metrics['source_citations'] = 0.6
        else:
            metrics['source_citations'] = 0.4
        
        # Calculate overall score (weighted average)
        weights = {
            'domain_credibility': 0.4,
            'content_quality': 0.3,
            'information_diversity': 0.15,
            'source_citations': 0.15
        }
        
        overall_score = sum(
            metrics[key] * weights[key] for key in weights
        )
        
        metrics['overall_score'] = round(overall_score, 2)
        
        return metrics
    
    def stop(self):
        """Stop the worker thread."""
        self.stop_requested = True
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
        logger.info("Web scraping job processing thread stopped")