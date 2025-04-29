"""
Data scaling utilities for the CIVILIAN system.
These utilities help prepare and scale data from internet sources for analysis.
"""

import re
import logging
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Union

# Configure logger
logger = logging.getLogger(__name__)

class DataScaler:
    """A utility class for scaling and normalizing data from various sources."""
    
    @staticmethod
    def clean_text(text: str, remove_urls: bool = True, 
                   remove_html: bool = True, lowercase: bool = False) -> str:
        """
        Clean text data by removing unwanted elements.
        
        Args:
            text: Input text to clean
            remove_urls: Whether to remove URLs
            remove_html: Whether to remove HTML tags
            lowercase: Whether to convert to lowercase
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove URLs
        if remove_urls:
            text = re.sub(r'https?://\S+', '', text)
            text = re.sub(r'www\.\S+', '', text)
        
        # Remove HTML tags
        if remove_html:
            text = re.sub(r'<.*?>', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Convert to lowercase if requested
        if lowercase:
            text = text.lower()
        
        return text
    
    @staticmethod
    def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
        """
        Extract key terms from text using simple frequency analysis.
        
        Args:
            text: Input text
            max_keywords: Maximum number of keywords to extract
            
        Returns:
            List of extracted keywords
        """
        if not text:
            return []
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation and split into words
        words = re.findall(r'\b[a-z]{3,}\b', text)
        
        # Remove common stop words (simplified list)
        stop_words = {
            'the', 'and', 'or', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'by',
            'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before',
            'after', 'above', 'below', 'from', 'up', 'down', 'this', 'that', 'these',
            'those', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
            'had', 'having', 'do', 'does', 'did', 'doing', 'but', 'if', 'because', 'as',
            'until', 'while', 'there', 'here', 'when', 'where', 'why', 'how', 'all', 'any',
            'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
            'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'can', 'will',
            'just', 'should', 'now'
        }
        
        # Filter out stop words and count frequencies
        filtered_words = [word for word in words if word not in stop_words]
        
        if not filtered_words:
            return []
        
        # Count frequencies
        word_counts = {}
        for word in filtered_words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        # Sort by frequency
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Return top keywords
        keywords = [word for word, count in sorted_words[:max_keywords]]
        
        return keywords
    
    @staticmethod
    def normalize_date(date_str: Optional[str]) -> Optional[str]:
        """
        Normalize date strings to ISO format.
        
        Args:
            date_str: Date string in various formats
            
        Returns:
            ISO-formatted date string or None if parsing fails
        """
        if not date_str:
            return None
        
        # Common date formats to try
        formats = [
            '%Y-%m-%d',
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%dT%H:%M:%SZ',
            '%Y-%m-%d %H:%M:%S',
            '%d-%m-%Y',
            '%m/%d/%Y',
            '%d/%m/%Y',
            '%B %d, %Y',
            '%d %B %Y',
            '%Y/%m/%d'
        ]
        
        # Try each format
        for fmt in formats:
            try:
                dt = datetime.strptime(date_str, fmt)
                return dt.isoformat()
            except ValueError:
                continue
        
        # If all formats fail, return original
        logger.warning(f"Could not normalize date format: {date_str}")
        return date_str
    
    @staticmethod
    def prepare_web_content(content_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare web content for analysis by normalizing and enriching data.
        
        Args:
            content_data: Dictionary containing web content data
            
        Returns:
            Processed and enriched content data
        """
        result = {}
        
        # Copy basic metadata
        result['url'] = content_data.get('url', '')
        result['title'] = content_data.get('title', '')
        result['source'] = content_data.get('site_name', '')
        result['author'] = content_data.get('author', '')
        
        # Set content type
        result['content_type'] = 'web_page'
        
        # Extract and clean content
        content = content_data.get('content', '')
        if content:
            result['content'] = DataScaler.clean_text(content)
            
            # Extract summary (first 500 characters)
            result['summary'] = DataScaler.clean_text(content[:500])
            
            # Extract keywords
            result['keywords'] = DataScaler.extract_keywords(content)
        else:
            result['content'] = ''
            result['summary'] = ''
            result['keywords'] = []
        
        # Normalize date
        published_date = content_data.get('published_date')
        if published_date:
            result['published_date'] = DataScaler.normalize_date(published_date)
        else:
            result['published_date'] = None
        
        # Add timestamp for when this data was processed
        result['processed_at'] = datetime.now().isoformat()
        
        # Add metadata field for additional information
        result['meta_data'] = {
            'domain': content_data.get('domain', ''),
            'language': content_data.get('language', 'en'),
            'content_length': len(content) if content else 0,
            'scraped_at': content_data.get('scraped_at', datetime.now().isoformat())
        }
        
        return result
    
    @staticmethod
    def format_for_detection(web_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format web data for the detector agent.
        
        Args:
            web_data: Processed web content data
            
        Returns:
            Data formatted for detection system
        """
        detection_data = {
            'title': web_data.get('title', ''),
            'content': web_data.get('content', ''),
            'source': web_data.get('source', 'web'),
            'url': web_data.get('url', ''),
            'published_date': web_data.get('published_date'),
            'meta_data': json.dumps(web_data.get('meta_data', {})),
            'content_type': 'web'
        }
        
        return detection_data
    
    @staticmethod
    def score_relevance(content: str, keywords: List[str], threshold: float = 0.01) -> float:
        """
        Score the relevance of content based on keywords.
        
        Args:
            content: Text content to score
            keywords: List of keywords to search for
            threshold: Minimum relevance score
            
        Returns:
            Relevance score between 0.0 and 1.0
        """
        if not content or not keywords:
            return 0.0
        
        content_lower = content.lower()
        
        # Count keyword occurrences
        keyword_count = 0
        for keyword in keywords:
            keyword_lower = keyword.lower()
            keyword_count += content_lower.count(keyword_lower)
        
        # Calculate density (keywords per word)
        total_words = len(content.split())
        if total_words == 0:
            return 0.0
            
        density = keyword_count / total_words
        
        # Normalize score between 0 and 1
        score = min(1.0, density * 10)  # Scale up for better distribution
        
        # Apply threshold
        if score < threshold:
            return 0.0
            
        return score
    
    @staticmethod
    def batch_process(data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process a batch of data items.
        
        Args:
            data_list: List of data items to process
            
        Returns:
            List of processed data items
        """
        results = []
        
        for data in data_list:
            try:
                # Prepare data based on content type
                if data.get('content_type') == 'web_page' or 'url' in data:
                    processed = DataScaler.prepare_web_content(data)
                else:
                    # Skip unknown data types
                    logger.warning(f"Unknown content type in data: {data.get('content_type')}")
                    continue
                
                results.append(processed)
                
            except Exception as e:
                logger.error(f"Error processing data item: {e}")
                continue
        
        return results