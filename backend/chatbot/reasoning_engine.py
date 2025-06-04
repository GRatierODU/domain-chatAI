import torch
from typing import Dict, List, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import json
import re
from dataclasses import dataclass
from enum import Enum
import time
import logging
from .retrieval_optimizer import OptimizedRetriever, RetrievalResult
from .complexity_classifier import ComplexityClassifier, QueryComplexity
import hashlib

logger = logging.getLogger(__name__)

@dataclass
class ReasoningResponse:
    answer: str
    reasoning: Optional[str]
    confidence: float
    sources: List[Dict]
    query_type: QueryComplexity
    processing_time: float

class ReasoningEngine:
    """
    Natural conversation engine that acts like a knowledgeable employee
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.models = {}
        self.tokenizers = {}
        self._load_models()
        
        # Query classifier
        self.complexity_classifier = ComplexityClassifier()
        
        # Response cache
        self.cache = {}
        
        # Conversation style - professional but friendly employee
        self.conversation_style = {
            "personality": "knowledgeable employee",
            "tone": "professional yet friendly",
            "knowledge_level": "complete",  # Acts like they know everything about the business
        }
        
    def _load_models(self):
        """Load reasoning models based on configuration"""
        # Simplified for testing - in production load actual models
        logger.info("Initializing reasoning engine...")
        self.models['efficient'] = None
        self.tokenizers['efficient'] = None
        
    async def answer_question(
        self,
        question: str,
        context: str,
        retriever: OptimizedRetriever,
        conversation_history: List[Dict] = []
    ) -> ReasoningResponse:
        """
        Generate response like a knowledgeable employee of the business
        """
        start_time = time.time()
        
        # Check if this is a greeting
        question_lower = question.lower().strip()
        is_greeting = any(greeting in question_lower for greeting in ['hi', 'hello', 'hey', 'howdy', 'greetings'])
        
        # Handle greetings based on conversation history
        if is_greeting:
            if len(conversation_history) == 0:
                # First greeting - introduce as employee
                answer = f"Hello! Welcome to {context}. I'm here to help you with any questions about our business, services, hours, or anything else you'd like to know. How can I assist you today?"
                return ReasoningResponse(
                    answer=answer,
                    reasoning=None,
                    confidence=1.0,
                    sources=[],  # No sources for greeting
                    query_type=QueryComplexity.SIMPLE,
                    processing_time=time.time() - start_time
                )
            else:
                # Already greeted - simple acknowledgment
                answer = "Hello again! What else can I help you with?"
                return ReasoningResponse(
                    answer=answer,
                    reasoning=None,
                    confidence=1.0,
                    sources=[],
                    query_type=QueryComplexity.SIMPLE,
                    processing_time=time.time() - start_time
                )
        
        # For all other questions, retrieve relevant information
        try:
            retrieved_info = await retriever.retrieve(
                question,
                {'conversation_history': conversation_history},
                top_k=5,
                rerank=True
            )
        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            retrieved_info = []
        
        # Extract actual content from retrieved information
        relevant_content = []
        sources = []
        
        for info in retrieved_info:
            if info.content and len(info.content.strip()) > 10:
                relevant_content.append(info.content.strip())
                # Only add source if we're actually using this content
                if info.metadata and info.metadata.get('url'):
                    source = {
                        'url': info.metadata['url'],
                        'title': info.metadata.get('title', 'Source')
                    }
                    if source not in sources:
                        sources.append(source)
        
        # Generate response based on what we found
        if relevant_content:
            # We have actual information - respond as knowledgeable employee
            answer = self._generate_knowledgeable_response(question, relevant_content, context)
            confidence = 0.95
        else:
            # No specific information found - respond helpfully
            answer = self._generate_helpful_response(question, context, conversation_history)
            sources = []  # Don't show sources when we don't have specific info
            confidence = 0.6
        
        # Determine query complexity
        complexity = self.complexity_classifier.classify(question, context)
        
        return ReasoningResponse(
            answer=answer,
            reasoning=None,
            confidence=confidence,
            sources=sources[:3],  # Limit to 3 most relevant sources
            query_type=complexity,
            processing_time=time.time() - start_time
        )
    
    def _generate_knowledgeable_response(self, question: str, content_pieces: List[str], context: str) -> str:
        """Generate response using actual knowledge like an employee would"""
        question_lower = question.lower()
        
        # Combine relevant content
        combined_content = " ".join(content_pieces[:3])  # Use top 3 most relevant pieces
        
        # Different response patterns based on question type
        if any(word in question_lower for word in ['what', 'about', 'tell me about', 'describe']):
            # General information questions
            if 'website' in question_lower or 'business' in question_lower or 'company' in question_lower:
                # They want overview
                response = f"{combined_content}"
                if len(content_pieces) > 3:
                    response += " I'd be happy to tell you more about any specific aspect that interests you."
            else:
                response = f"{combined_content}"
        
        elif any(word in question_lower for word in ['when', 'hours', 'open', 'close']):
            # Hours/timing questions
            response = f"{combined_content}"
            
        elif any(word in question_lower for word in ['where', 'location', 'address', 'find']):
            # Location questions
            response = f"{combined_content}"
            
        elif any(word in question_lower for word in ['how much', 'price', 'cost', 'fee']):
            # Pricing questions
            response = f"{combined_content}"
            
        elif any(word in question_lower for word in ['menu', 'options', 'choices', 'selection']):
            # Menu/options questions
            response = f"{combined_content}"
            
        elif any(word in question_lower for word in ['contact', 'phone', 'email', 'reach']):
            # Contact questions
            response = f"{combined_content}"
            
        elif any(word in question_lower for word in ['service', 'offer', 'provide', 'do']):
            # Services questions
            response = f"{combined_content}"
            
        else:
            # Default pattern
            response = f"{combined_content}"
        
        return response
    
    def _generate_helpful_response(self, question: str, context: str, history: List[Dict]) -> str:
        """Generate helpful response when no specific information is found"""
        question_lower = question.lower()
        
        # Check what they're asking about
        if any(word in question_lower for word in ['hours', 'open', 'close', 'when']):
            return "I don't have our hours information available at the moment. This information might be on our hours or contact page, or you could call us directly for current hours."
            
        elif any(word in question_lower for word in ['menu', 'food', 'dishes', 'eat']):
            return "I don't have our menu details available right now. You might find this on our menu page, or I'd be happy to help you with other information about our business."
            
        elif any(word in question_lower for word in ['price', 'cost', 'how much', 'fee']):
            return "I don't have specific pricing information available. For current prices, please check our pricing page or contact us directly."
            
        elif any(word in question_lower for word in ['service', 'offer', 'provide', 'what do you']):
            return f"I don't have detailed information about our specific services right now. Would you like me to help you find our services page or contact information so you can get the details you need?"
            
        else:
            # Generic helpful response
            return f"I don't have specific information about that right now. Is there something else about {context} I can help you with? I can provide information about our business, location, contact details, or other general information."
    
    def _generate_cache_key(self, question: str, context: str) -> str:
        """Generate cache key for response"""
        return hashlib.md5(f"{question}:{context}".encode()).hexdigest()