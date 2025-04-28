import spacy
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import re
import logging
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Download required NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

logger = logging.getLogger(__name__)

class TextProcessor:
    """Handles NLP processing for misinformation detection."""
    
    def __init__(self, languages: List[str] = ['en']):
        """Initialize the text processor with specified languages."""
        self.languages = languages
        self.nlp_models = {}
        
        # Load spaCy models for each language
        for lang in languages:
            try:
                if lang == 'en':
                    try:
                        self.nlp_models[lang] = spacy.load('en_core_web_sm')
                    except OSError:
                        # Fallback to blank model if language model not available
                        logger.warning(f"No spaCy model available for language {lang}, using blank model")
                        self.nlp_models[lang] = spacy.blank("en")
                elif lang == 'es':
                    try:
                        self.nlp_models[lang] = spacy.load('es_core_news_sm')
                    except OSError:
                        # Fallback to blank model if language model not available
                        logger.warning(f"No spaCy model available for language {lang}, using blank model")
                        self.nlp_models[lang] = spacy.blank("es")
                else:
                    logger.warning(f"Unsupported language: {lang}")
            except Exception as e:
                logger.warning(f"Failed to load spaCy model for {lang}: {str(e)}. Using NLTK fallbacks where possible.")
        
        # Initialize TF-IDF vectorizer
        self.tfidf = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 3)
        )
        
        logger.info(f"TextProcessor initialized with languages: {languages}")
    
    def detect_language(self, text: str) -> str:
        """Detect the language of the given text."""
        # Simple language detection based on available models
        # In a production system, use a proper language detection library
        
        # Default to English if detection fails
        if not text or len(text.strip()) == 0:
            return 'en'
            
        # Try to detect language with spaCy models
        max_score = 0
        detected_lang = 'en'
        
        for lang, model in self.nlp_models.items():
            doc = model(text[:100])  # Use only first 100 chars for efficiency
            score = sum(1 for token in doc if not token.is_punct and not token.is_space)
            if score > max_score:
                max_score = score
                detected_lang = lang
                
        return detected_lang
    
    def preprocess(self, text: str, lang: Optional[str] = None) -> str:
        """Preprocess text for analysis."""
        if not text:
            return ""
            
        # Detect language if not provided
        if lang is None:
            lang = self.detect_language(text)
        
        # Basic cleaning
        text = text.lower()
        text = re.sub(r'http\S+', '', text)  # Remove URLs
        text = re.sub(r'@\w+', '', text)     # Remove mentions
        text = re.sub(r'#\w+', '', text)     # Remove hashtags
        text = re.sub(r'\s+', ' ', text)     # Normalize whitespace
        
        # Language-specific processing if spaCy model is available
        if lang in self.nlp_models:
            doc = self.nlp_models[lang](text)
            # Keep only content words, remove stopwords and punctuation
            tokens = [token.lemma_ for token in doc 
                     if not token.is_stop and not token.is_punct and token.text.strip()]
            return " ".join(tokens)
        else:
            # Fallback to basic NLTK processing
            tokens = word_tokenize(text)
            stopwords = set(nltk.corpus.stopwords.words('english'))  # Default to English
            tokens = [token for token in tokens if token.isalnum() and token not in stopwords]
            return " ".join(tokens)
    
    def extract_entities(self, text: str, lang: Optional[str] = None) -> List[Dict[str, Any]]:
        """Extract named entities from text."""
        if not text:
            return []
            
        # Detect language if not provided
        if lang is None:
            lang = self.detect_language(text)
            
        if lang not in self.nlp_models:
            logger.warning(f"No spaCy model available for language {lang}")
            return []
            
        doc = self.nlp_models[lang](text)
        entities = []
        
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char
            })
            
        return entities
    
    def extract_key_phrases(self, text: str, n: int = 5) -> List[str]:
        """Extract key phrases from text using TF-IDF."""
        preprocessed = self.preprocess(text)
        if not preprocessed:
            return []
            
        # Fit and transform the text
        tfidf_matrix = self.tfidf.fit_transform([preprocessed])
        feature_names = self.tfidf.get_feature_names_out()
        
        # Get top N features with highest TF-IDF scores
        scores = zip(feature_names, tfidf_matrix.toarray()[0])
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
        
        return [phrase for phrase, score in sorted_scores[:n]]
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts."""
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Preprocess texts
        prep_text1 = self.preprocess(text1)
        prep_text2 = self.preprocess(text2)
        
        if not prep_text1 or not prep_text2:
            return 0.0
            
        # Vectorize texts
        vectors = self.tfidf.fit_transform([prep_text1, prep_text2])
        
        # Calculate cosine similarity
        similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
        
        return float(similarity)
    
    def analyze_sentiment(self, text: str, lang: Optional[str] = None) -> Dict[str, float]:
        """Analyze sentiment of text."""
        if not text:
            return {'positive': 0.0, 'neutral': 1.0, 'negative': 0.0}
            
        # Detect language if not provided
        if lang is None:
            lang = self.detect_language(text)
            
        if lang not in self.nlp_models:
            logger.warning(f"No spaCy model available for language {lang}")
            return {'positive': 0.0, 'neutral': 1.0, 'negative': 0.0}
            
        # Simple rule-based sentiment analysis
        # In a production system, use a dedicated sentiment analysis model
        doc = self.nlp_models[lang](text)
        
        positive_words = 0
        negative_words = 0
        total_words = 0
        
        # Very simple lexicon-based approach
        positive_lexicon = {'good', 'great', 'excellent', 'positive', 'amazing', 'wonderful', 'happy', 'love', 'like'}
        negative_lexicon = {'bad', 'terrible', 'awful', 'negative', 'horrible', 'sad', 'hate', 'dislike'}
        
        for token in doc:
            if token.is_alpha and not token.is_stop:
                total_words += 1
                if token.lemma_.lower() in positive_lexicon:
                    positive_words += 1
                elif token.lemma_.lower() in negative_lexicon:
                    negative_words += 1
        
        if total_words == 0:
            return {'positive': 0.0, 'neutral': 1.0, 'negative': 0.0}
            
        positive_score = positive_words / total_words
        negative_score = negative_words / total_words
        neutral_score = 1.0 - (positive_score + negative_score)
        
        return {
            'positive': positive_score,
            'neutral': neutral_score,
            'negative': negative_score
        }
        
    def extract_claim_candidates(self, text: str, lang: Optional[str] = None) -> List[str]:
        """Extract potential claims from text."""
        if not text:
            return []
            
        # Detect language if not provided
        if lang is None:
            lang = self.detect_language(text)
        
        # Split into sentences
        sentences = sent_tokenize(text)
        
        # Identify claim candidates (sentences that make assertions)
        claim_indicators = [
            'is', 'are', 'was', 'were', 'will', 'shall',  # Verbs of being
            'claim', 'state', 'report', 'show', 'reveal', 'confirm',  # Reporting verbs
            'according to', 'research', 'study', 'evidence', 'proof',  # Evidence markers
            'always', 'never', 'all', 'none', 'every', 'most',  # Generalizers
            'must', 'should', 'have to', 'need to'  # Modal verbs
        ]
        
        claim_candidates = []
        
        for sentence in sentences:
            # Simple heuristic: if the sentence contains claim indicators and is not a question
            if not sentence.strip().endswith('?'):
                if any(indicator in sentence.lower() for indicator in claim_indicators):
                    claim_candidates.append(sentence.strip())
        
        return claim_candidates
