import re
from enum import Enum
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class QueryComplexity(Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    COMPUTATIONAL = "computational"

class ComplexityClassifier:
    """
    Classifies query complexity for routing to appropriate model
    """
    
    def __init__(self):
        # Keywords and patterns for classification
        self.patterns = {
            'simple': {
                'keywords': [
                    'what', 'where', 'when', 'who', 'hours', 'open', 'close',
                    'phone', 'email', 'address', 'location', 'contact', 'price',
                    'cost', 'name', 'title'
                ],
                'patterns': [
                    r'^what is (?:the |your )?\w+\??$',
                    r'^where (?:is|are) .+\??$',
                    r'^when (?:do|does|is|are) .+\??$',
                    r'^how much (?:is|does) .+\??$'
                ],
                'max_words': 10
            },
            'moderate': {
                'keywords': [
                    'how do i', 'can i', 'what if', 'explain', 'describe',
                    'tell me about', 'process', 'steps', 'requirements',
                    'policy', 'shipping', 'return', 'warranty', 'guarantee'
                ],
                'patterns': [
                    r'^how (?:do|can|should) (?:i|we) .+\??$',
                    r'^what (?:are|is) the .+ (?:process|steps|requirements)',
                    r'^can (?:i|you|we) .+\??$',
                    r'^tell me (?:about|more)',
                    r'^explain .+$'
                ]
            },
            'complex': {
                'keywords': [
                    'compare', 'difference', 'between', 'recommend', 'best',
                    'should i', 'which one', 'help me decide', 'analyze',
                    'evaluate', 'pros and cons', 'suitable', 'appropriate',
                    'worth', 'better', 'advantages', 'disadvantages'
                ],
                'patterns': [
                    r'compare .+ (?:with|to|and) .+',
                    r'what(?:\'s| is) the difference between .+ and .+',
                    r'which .+ (?:should|would) .+ (?:choose|select|buy)',
                    r'help me (?:decide|choose) .+',
                    r'what .+ best for .+',
                    r'recommend .+ for .+'
                ]
            },
            'computational': {
                'keywords': [
                    'calculate', 'total', 'sum', 'multiply', 'divide',
                    'percentage', 'discount', 'tax', 'formula', 'equation'
                ],
                'patterns': [
                    r'\d+.*[\+\-\*\/].*\d+',
                    r'calculate .+',
                    r'what(?:\'s| is) .+ (?:total|sum|result)',
                    r'how much .+ if .+ \d+',
                    r'\d+\s*(?:percent|%)\s*(?:of|off)'
                ]
            }
        }
        
    def classify(self, query: str, context: str = "") -> QueryComplexity:
        """
        Classify the complexity of a query
        """
        query_lower = query.lower().strip()
        
        # Check for computational needs first
        if self._is_computational(query_lower):
            return QueryComplexity.COMPUTATIONAL
            
        # Check patterns in order of complexity
        for complexity_level in ['simple', 'moderate', 'complex']:
            if self._matches_complexity(query_lower, complexity_level):
                return QueryComplexity[complexity_level.upper()]
                
        # Default based on query length and structure
        word_count = len(query_lower.split())
        
        if word_count <= 5:
            return QueryComplexity.SIMPLE
        elif word_count <= 15:
            return QueryComplexity.MODERATE
        else:
            return QueryComplexity.COMPLEX
            
    def _matches_complexity(self, query: str, level: str) -> bool:
        """
        Check if query matches complexity level patterns
        """
        config = self.patterns[level]
        
        # Check keywords
        keywords_found = sum(1 for keyword in config['keywords'] if keyword in query)
        if keywords_found >= 2:  # Multiple keywords indicate this complexity
            return True
            
        # Check patterns
        for pattern in config.get('patterns', []):
            if re.search(pattern, query, re.IGNORECASE):
                return True
                
        # Check word limit for simple queries
        if level == 'simple' and 'max_words' in config:
            if len(query.split()) <= config['max_words'] and keywords_found >= 1:
                return True
                
        return False
        
    def _is_computational(self, query: str) -> bool:
        """
        Check if query requires computation
        """
        # Look for mathematical operators
        if re.search(r'[\+\-\*\/\%]', query):
            return True
            
        # Look for numbers with computation keywords
        has_numbers = bool(re.search(r'\d+', query))
        computation_keywords = ['calculate', 'total', 'sum', 'add', 'subtract', 
                              'multiply', 'divide', 'percent', 'average']
        
        if has_numbers and any(keyword in query for keyword in computation_keywords):
            return True
            
        # Look for price calculations
        if re.search(r'(?:price|cost|total).+(?:for|of|with).+\d+', query):
            return True
            
        return False
        
    def get_reasoning_requirements(self, complexity: QueryComplexity) -> Dict:
        """
        Get requirements for reasoning based on complexity
        """
        requirements = {
            QueryComplexity.SIMPLE: {
                'max_tokens': 256,
                'temperature': 0.5,
                'need_sources': True,
                'need_reasoning': False,
                'response_style': 'direct'
            },
            QueryComplexity.MODERATE: {
                'max_tokens': 512,
                'temperature': 0.7,
                'need_sources': True,
                'need_reasoning': True,
                'response_style': 'explanatory'
            },
            QueryComplexity.COMPLEX: {
                'max_tokens': 1024,
                'temperature': 0.7,
                'need_sources': True,
                'need_reasoning': True,
                'response_style': 'analytical'
            },
            QueryComplexity.COMPUTATIONAL: {
                'max_tokens': 512,
                'temperature': 0.3,
                'need_sources': True,
                'need_reasoning': True,
                'response_style': 'step_by_step'
            }
        }
        
        return requirements.get(complexity, requirements[QueryComplexity.MODERATE])