"""
Language processing utilities for multilingual support.
Handles language detection, translation, and text preprocessing.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import re

# Language processing imports
from langdetect import detect, LangDetectException

# googletrans is optional; it can break with certain httpcore versions.
# Use lazy/guarded import and provide a safe fallback.
try:
    from googletrans import Translator as GoogleTranslator  # type: ignore
    _GOOGLETRANS_AVAILABLE = True
except Exception as _gt_exc:  # noqa: F841
    GoogleTranslator = None  # type: ignore
    _GOOGLETRANS_AVAILABLE = False
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# Configuration
from .config_loader import get_config

logger = logging.getLogger(__name__)

@dataclass
class LanguageDetectionResult:
    """Result of language detection."""
    language: str
    confidence: float
    is_supported: bool

@dataclass
class TranslationResult:
    """Result of text translation."""
    original_text: str
    translated_text: str
    source_language: str
    target_language: str
    confidence: float

class LanguageDetector:
    """Language detection with confidence scoring."""
    
    def __init__(self):
        self.config = get_config()
        self.supported_languages = self.config.get('languages.supported', ['en'])
        self.primary_language = self.config.get('languages.primary', 'en')
    
    def detect_language(self, text: str, min_length: int = 10) -> LanguageDetectionResult:
        """
        Detect language of input text.
        
        Args:
            text: Input text to analyze
            min_length: Minimum text length for detection
            
        Returns:
            LanguageDetectionResult with detected language and confidence
        """
        if len(text.strip()) < min_length:
            return LanguageDetectionResult(
                language=self.primary_language,
                confidence=0.5,
                is_supported=True
            )
        
        try:
            # Clean text for better detection
            cleaned_text = self._clean_text_for_detection(text)
            
            # Detect language
            detected_lang = detect(cleaned_text)
            
            # Calculate confidence (simplified)
            confidence = self._calculate_confidence(cleaned_text, detected_lang)
            
            # Check if language is supported
            is_supported = detected_lang in self.supported_languages
            
            return LanguageDetectionResult(
                language=detected_lang,
                confidence=confidence,
                is_supported=is_supported
            )
            
        except LangDetectException as e:
            logger.warning(f"Language detection failed: {e}")
            return LanguageDetectionResult(
                language=self.primary_language,
                confidence=0.3,
                is_supported=True
            )
    
    def _clean_text_for_detection(self, text: str) -> str:
        """Clean text for better language detection."""
        # Remove URLs, emails, and special characters
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def _calculate_confidence(self, text: str, detected_lang: str) -> float:
        """Calculate confidence score for language detection."""
        # Simple confidence calculation based on text length and common words
        base_confidence = 0.8
        
        # Reduce confidence for very short texts
        if len(text) < 50:
            base_confidence *= 0.7
        elif len(text) < 20:
            base_confidence *= 0.5
        
        # Increase confidence if we detect language-specific patterns
        confidence_boost = self._check_language_patterns(text, detected_lang)
        
        return min(1.0, base_confidence + confidence_boost)
    
    def _check_language_patterns(self, text: str, lang: str) -> float:
        """Check for language-specific patterns to boost confidence."""
        patterns = {
            'en': [r'\b(the|and|that|have|for|not|with|you|this|but|his|from|they)\b'],
            'es': [r'\b(que|de|no|se|en|un|es|su|para|con|por|son|una)\b'],
            'de': [r'\b(der|die|das|und|in|zu|den|von|sie|ist|des|sich|mit)\b'],
            'fr': [r'\b(de|le|et|à|un|il|être|et|en|avoir|que|pour|dans|ce)\b'],
            'hi': [r'[\u0900-\u097F]'],  # Devanagari script
            'zh': [r'[\u4e00-\u9fff]'],  # Chinese characters
        }
        
        if lang in patterns:
            for pattern in patterns[lang]:
                matches = len(re.findall(pattern, text.lower()))
                if matches > 2:
                    return 0.1
        
        return 0.0

class MultilingualTranslator:
    """Multilingual translation with caching and fallback."""
    
    def __init__(self):
        self.config = get_config()
        # Initialize translator only if googletrans import succeeded and enabled in config
        self.translator = None
        try:
            use_translator = self.config.get('languages.translation_service', 'googletrans') == 'googletrans'
            if use_translator and _GOOGLETRANS_AVAILABLE and GoogleTranslator is not None:
                self.translator = GoogleTranslator()
            else:
                logger.warning("googletrans unavailable or disabled; falling back to no-op translation")
        except Exception as e:
            logger.warning(f"Failed to initialize googletrans translator, fallback will be used: {e}")
        self.translation_cache: Dict[str, TranslationResult] = {}
        self.primary_language = self.config.get('languages.primary', 'en')
        
    def translate_text(
        self, 
        text: str, 
        target_language: str = None, 
        source_language: str = None
    ) -> TranslationResult:
        """
        Translate text to target language.
        
        Args:
            text: Text to translate
            target_language: Target language code (defaults to primary language)
            source_language: Source language code (auto-detect if None)
            
        Returns:
            TranslationResult with translation details
        """
        if target_language is None:
            target_language = self.primary_language
        
        # Check cache first
        cache_key = f"{text[:50]}:{source_language}:{target_language}"
        if cache_key in self.translation_cache:
            return self.translation_cache[cache_key]
        
        try:
            # Detect source language if not provided
            if source_language is None:
                detector = LanguageDetector()
                detection = detector.detect_language(text)
                source_language = detection.language
            
            # Skip translation if source and target are the same
            if source_language == target_language:
                result = TranslationResult(
                    original_text=text,
                    translated_text=text,
                    source_language=source_language,
                    target_language=target_language,
                    confidence=1.0
                )
            else:
                # Perform translation if a translator is available; otherwise, pass-through
                if self.translator is not None:
                    translation = self.translator.translate(
                        text,
                        src=source_language,
                        dest=target_language
                    )
                    result = TranslationResult(
                        original_text=text,
                        translated_text=getattr(translation, 'text', text),
                        source_language=source_language,
                        target_language=target_language,
                        confidence=getattr(translation, 'confidence', 0.8)
                    )
                else:
                    # No translator available; return original text
                    result = TranslationResult(
                        original_text=text,
                        translated_text=text,
                        source_language=source_language,
                        target_language=target_language,
                        confidence=0.0
                    )
            
            # Cache result
            self.translation_cache[cache_key] = result
            return result
            
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            # Return original text as fallback
            return TranslationResult(
                original_text=text,
                translated_text=text,
                source_language=source_language or 'unknown',
                target_language=target_language,
                confidence=0.0
            )

class TextPreprocessor:
    """Text preprocessing for multilingual content."""
    
    def __init__(self):
        self.config = get_config()
        self.stemmers = {}
        self.stopwords_cache = {}
        self._initialize_nltk_resources()
    
    def _initialize_nltk_resources(self):
        """Initialize NLTK resources for supported languages."""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
    
    def preprocess_text(
        self, 
        text: str, 
        language: str = 'en',
        remove_stopwords: bool = True,
        stem_words: bool = False,
        normalize_case: bool = True
    ) -> str:
        """
        Preprocess text for better analysis.
        
        Args:
            text: Input text
            language: Language code
            remove_stopwords: Whether to remove stopwords
            stem_words: Whether to apply stemming
            normalize_case: Whether to normalize case
            
        Returns:
            Preprocessed text
        """
        if not text or not text.strip():
            return ""
        
        # Normalize case
        if normalize_case:
            text = text.lower()
        
        # Clean text
        text = self._clean_text(text)
        
        # Tokenize
        tokens = word_tokenize(text, language=self._get_nltk_language(language))
        
        # Remove stopwords
        if remove_stopwords:
            stopwords_set = self._get_stopwords(language)
            tokens = [token for token in tokens if token not in stopwords_set]
        
        # Stem words
        if stem_words:
            stemmer = self._get_stemmer(language)
            if stemmer:
                tokens = [stemmer.stem(token) for token in tokens]
        
        return ' '.join(tokens)
    
    def _clean_text(self, text: str) -> str:
        """Clean text by removing unwanted characters."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove excessive punctuation
        text = re.sub(r'[^\w\s.-]', ' ', text)
        
        return text.strip()
    
    def _get_stopwords(self, language: str) -> set:
        """Get stopwords for language with caching."""
        if language not in self.stopwords_cache:
            try:
                # Map language codes to NLTK stopwords languages
                nltk_lang_map = {
                    'en': 'english',
                    'es': 'spanish',
                    'de': 'german',
                    'fr': 'french',
                    'it': 'italian',
                    'pt': 'portuguese',
                    'ru': 'russian',
                }
                
                nltk_lang = nltk_lang_map.get(language, 'english')
                self.stopwords_cache[language] = set(stopwords.words(nltk_lang))
            except:
                # Fallback to English stopwords
                self.stopwords_cache[language] = set(stopwords.words('english'))
        
        return self.stopwords_cache[language]
    
    def _get_stemmer(self, language: str) -> Optional[SnowballStemmer]:
        """Get stemmer for language with caching."""
        if language not in self.stemmers:
            try:
                # Map language codes to Snowball stemmer languages
                stemmer_lang_map = {
                    'en': 'english',
                    'es': 'spanish',
                    'de': 'german',
                    'fr': 'french',
                    'it': 'italian',
                    'pt': 'portuguese',
                    'ru': 'russian',
                }
                
                stemmer_lang = stemmer_lang_map.get(language)
                if stemmer_lang:
                    self.stemmers[language] = SnowballStemmer(stemmer_lang)
                else:
                    self.stemmers[language] = None
            except:
                self.stemmers[language] = None
        
        return self.stemmers[language]
    
    def _get_nltk_language(self, language: str) -> str:
        """Map language code to NLTK language for tokenization."""
        nltk_lang_map = {
            'en': 'english',
            'es': 'spanish',
            'de': 'german',
            'fr': 'french',
            'it': 'italian',
            'pt': 'portuguese',
        }
        return nltk_lang_map.get(language, 'english')

# Utility functions
def detect_language(text: str) -> LanguageDetectionResult:
    """Detect language of text."""
    detector = LanguageDetector()
    return detector.detect_language(text)

def translate_to_english(text: str, source_language: str = None) -> TranslationResult:
    """Translate text to English."""
    translator = MultilingualTranslator()
    return translator.translate_text(text, target_language='en', source_language=source_language)

def preprocess_multilingual_text(text: str, language: str = None) -> str:
    """Preprocess text with automatic language detection."""
    if language is None:
        detection = detect_language(text)
        language = detection.language
    
    preprocessor = TextPreprocessor()
    return preprocessor.preprocess_text(text, language=language)