"""
Text processing utilities for the CIVILIAN system.
Provides tools for text cleaning, analysis, and extraction.
"""

import logging
import re
import string
from typing import Dict, List, Optional, Set, Tuple, Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

class TextProcessor:
    """Utility class for processing and analyzing text content."""
    
    def __init__(self):
        """Initialize the text processor."""
        # Define common stopwords for cleaning
        self.stopwords = {
            'the', 'and', 'is', 'in', 'it', 'to', 'that', 'of', 'for', 'on', 'with', 
            'as', 'this', 'by', 'an', 'are', 'at', 'be', 'but', 'or', 'from', 'not', 
            'what', 'all', 'were', 'when', 'we', 'there', 'can', 'no', 'has', 'just',
            'into', 'your', 'some', 'them', 'more', 'will', 'which', 'their', 'about',
            'now', 'out', 'up', 'also', 'been', 'if', 'so', 'was', 'like', 'they',
            'would', 'then', 'other', 'who', 'because', 'many', 'one', 'you', 'should'
        }
    
    def clean_text(self, text: str, remove_stopwords: bool = False) -> str:
        """
        Clean text by removing punctuation, extra whitespace, and optionally stopwords.
        
        Args:
            text: Text to clean
            remove_stopwords: Whether to remove stopwords
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove stopwords if requested
        if remove_stopwords:
            words = text.split()
            words = [word for word in words if word not in self.stopwords]
            text = ' '.join(words)
        
        return text
    
    def extract_keywords(self, text: str, top_n: int = 10) -> List[Tuple[str, int]]:
        """
        Extract keywords from text based on frequency.
        
        Args:
            text: Text to extract keywords from
            top_n: Number of top keywords to return
            
        Returns:
            List of (keyword, frequency) tuples
        """
        if not text:
            return []
        
        # Clean the text and remove stopwords
        cleaned_text = self.clean_text(text, remove_stopwords=True)
        
        # Tokenize and count word frequencies
        words = cleaned_text.split()
        word_count = {}
        
        for word in words:
            if len(word) > 2:  # Skip very short words
                word_count[word] = word_count.get(word, 0) + 1
        
        # Sort by frequency and return top N
        sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
        return sorted_words[:top_n]
    
    def extract_sentences(self, text: str) -> List[str]:
        """
        Extract sentences from text.
        
        Args:
            text: Text to extract sentences from
            
        Returns:
            List of sentences
        """
        if not text:
            return []
        
        # Simple sentence splitting using regex
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text)
        
        # Clean up sentences
        sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
        
        return sentences
    
    def extract_domain(self, url: str) -> Optional[str]:
        """
        Extract domain from URL.
        
        Args:
            url: URL to extract domain from
            
        Returns:
            Domain or None if URL is invalid
        """
        try:
            parsed_url = urlparse(url)
            domain = parsed_url.netloc
            
            # Remove www. prefix if present
            if domain.startswith('www.'):
                domain = domain[4:]
                
            return domain
            
        except Exception as e:
            logger.error(f"Error extracting domain from URL {url}: {str(e)}")
            return None
    
    def summarize_text(self, text: str, max_sentences: int = 3) -> str:
        """
        Create a simple extractive summary of text.
        
        Args:
            text: Text to summarize
            max_sentences: Maximum number of sentences in the summary
            
        Returns:
            Text summary
        """
        if not text:
            return ""
        
        # Extract sentences
        sentences = self.extract_sentences(text)
        
        if not sentences:
            return ""
        
        # If we have fewer sentences than max_sentences, return the whole text
        if len(sentences) <= max_sentences:
            return text
        
        # Score sentences based on word frequency
        word_freq = {}
        
        # Count word frequencies across all text
        for sentence in sentences:
            for word in self.clean_text(sentence, remove_stopwords=True).split():
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Score each sentence
        sentence_scores = []
        
        for sentence in sentences:
            score = 0
            words = self.clean_text(sentence, remove_stopwords=True).split()
            
            for word in words:
                score += word_freq.get(word, 0)
            
            # Normalize by sentence length to avoid bias towards longer sentences
            if words:
                score = score / len(words)
                
            sentence_scores.append((sentence, score))
        
        # Sort sentences by score and take top max_sentences
        sorted_sentences = sorted(sentence_scores, key=lambda x: x[1], reverse=True)
        top_sentences = [sentence for sentence, _ in sorted_sentences[:max_sentences]]
        
        # Reorder sentences to maintain original order
        ordered_summary = []
        
        for sentence in sentences:
            if sentence in top_sentences:
                ordered_summary.append(sentence)
                
                # Break if we've found all our top sentences
                if len(ordered_summary) == len(top_sentences):
                    break
        
        return ' '.join(ordered_summary)
    
    def detect_language(self, text: str) -> str:
        """
        Detect language of text.
        This is a simplified implementation that works for English only.
        In a real implementation, you would use a library like langdetect.
        
        Args:
            text: Text to detect language of
            
        Returns:
            Language code ('en' for English, 'unknown' otherwise)
        """
        if not text:
            return "unknown"
        
        # Simple heuristic - check for common English words
        english_markers = {'the', 'and', 'is', 'in', 'it', 'to', 'that', 'of', 'for', 'on'}
        
        words = set(self.clean_text(text.lower()).split())
        common_words = words.intersection(english_markers)
        
        # If more than 3 common English words are found, assume it's English
        if len(common_words) > 3:
            return "en"
        
        return "unknown"
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities from text.
        This is a simplified implementation that uses regular expressions.
        In a real implementation, you would use a library like spaCy.
        
        Args:
            text: Text to extract entities from
            
        Returns:
            Dictionary of entity types to lists of entities
        """
        if not text:
            return {
                "persons": [],
                "organizations": [],
                "locations": [],
                "dates": []
            }
        
        # Simple regex patterns for entity extraction
        # Note: These patterns are very basic and will miss many entities
        # and may have false positives
        
        # Person pattern - capitalized names
        person_pattern = r'(?:[A-Z][a-z]+ ){1,2}[A-Z][a-z]+'
        
        # Organization pattern - sequences of capitalized words
        org_pattern = r'(?:[A-Z][a-z]+ ){2,}(?:Inc|LLC|Ltd|Corp|Corporation|Company|Group)'
        
        # Location pattern - capitalized place names
        location_pattern = r'(?:in|at|from|to) ([A-Z][a-z]+(?: [A-Z][a-z]+)?)'
        
        # Date pattern - dates in various formats
        date_pattern = r'\b(?:\d{1,2} (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{2,4}|\d{1,2}/\d{1,2}/\d{2,4}|\d{4}-\d{2}-\d{2})\b'
        
        # Extract entities
        persons = set(re.findall(person_pattern, text))
        organizations = set(re.findall(org_pattern, text))
        locations = set([match[0] for match in re.findall(location_pattern, text)])
        dates = set(re.findall(date_pattern, text))
        
        return {
            "persons": list(persons),
            "organizations": list(organizations), 
            "locations": list(locations),
            "dates": list(dates)
        }