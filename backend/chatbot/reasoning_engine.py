import torch
from typing import Dict, List, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import json
import re
from dataclasses import dataclass
from enum import Enum
import time
import logging
from .retrieval_optimizer import OptimizedRetriever
from .complexity_classifier import ComplexityClassifier

logger = logging.getLogger(__name__)

class QueryComplexity(Enum):
    SIMPLE = "simple"        # Direct fact lookup
    MODERATE = "moderate"    # Some inference needed
    COMPLEX = "complex"      # Deep reasoning required
    COMPUTATIONAL = "computational"  # Calculations needed

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
    Production-ready reasoning engine with tiered model approach
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
        
    def _load_models(self):
        """
        Load reasoning models based on configuration
        """
        # Load efficient model (always)
        logger.info("Loading Phi-3 for efficient responses...")
        self._load_model(
            'efficient',
            'microsoft/Phi-3-medium-128k-instruct',
            quantization='4bit'
        )
        
        # Load secondary model if enough memory
        if self._check_available_memory() > 16:
            logger.info("Loading DeepSeek-R1 for moderate reasoning...")
            self._load_model(
                'secondary',
                'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B',
                quantization='4bit'
            )
            
        # Load primary model if enough memory
        if self._check_available_memory() > 24:
            logger.info("Loading QwQ for complex reasoning...")
            self._load_model(
                'primary',
                'Qwen/QwQ-32B-Preview',
                quantization='4bit'
            )
            
    def _load_model(self, tier: str, model_path: str, quantization: str):
        """
        Load a specific model with optimization
        """
        if quantization == '4bit':
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
        else:
            bnb_config = None
            
        self.tokenizers[tier] = AutoTokenizer.from_pretrained(model_path)
        self.models[tier] = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        
        logger.info(f"✓ {tier} model loaded successfully")
        
    async def answer_question(
        self,
        question: str,
        context: str,
        retriever: OptimizedRetriever,
        conversation_history: List[Dict] = []
    ) -> ReasoningResponse:
        """
        Main entry point for answering questions
        """
        start_time = time.time()
        
        # Check cache
        cache_key = self._generate_cache_key(question, context)
        if cache_key in self.cache:
            cached = self.cache[cache_key]
            cached.processing_time = time.time() - start_time
            return cached
            
        # Classify query complexity
        complexity = self.complexity_classifier.classify(question, context)
        
        # Retrieve relevant information
        retrieved_info = await retriever.retrieve(
            question,
            {'conversation_history': conversation_history},
            top_k=5
        )
        
        # Route to appropriate handler
        if complexity == QueryComplexity.SIMPLE:
            response = await self._handle_simple(question, retrieved_info)
        elif complexity == QueryComplexity.MODERATE:
            response = await self._handle_moderate(question, retrieved_info, context)
        elif complexity == QueryComplexity.COMPLEX:
            response = await self._handle_complex(
                question, retrieved_info, context, conversation_history
            )
        elif complexity == QueryComplexity.COMPUTATIONAL:
            response = await self._handle_computational(question, retrieved_info)
        else:
            response = await self._handle_moderate(question, retrieved_info, context)
            
        # Add processing time
        response.processing_time = time.time() - start_time
        
        # Cache response
        self.cache[cache_key] = response
        
        return response
        
    async def _handle_complex(
        self,
        question: str,
        retrieved_info: List,
        context: str,
        history: List
    ) -> ReasoningResponse:
        """
        Handle complex queries requiring deep reasoning
        """
        if 'primary' not in self.models:
            # Fallback to best available
            return await self._handle_moderate(question, retrieved_info, context)
            
        model = self.models['primary']
        tokenizer = self.tokenizers['primary']
        
        # Build comprehensive prompt
        prompt = self._build_complex_prompt(
            question,
            retrieved_info,
            context,
            history
        )
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.7,
                do_sample=True,
                top_p=0.95,
                repetition_penalty=1.1
            )
            
        response_text = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        # Parse response
        answer, reasoning = self._parse_reasoning_response(response_text)
        
        return ReasoningResponse(
            answer=answer,
            reasoning=reasoning,
            confidence=0.95,
            sources=self._extract_sources(retrieved_info),
            query_type=QueryComplexity.COMPLEX,
            processing_time=0.0
        )
        
    def _build_complex_prompt(
        self,
        question: str,
        retrieved_info: List,
        context: str,
        history: List
    ) -> str:
        """
        Build comprehensive prompt for complex reasoning
        """
        # Format retrieved information
        info_text = "\n\n".join([
            f"[Source {i+1}] {info.content}"
            for i, info in enumerate(retrieved_info[:5])
        ])
        
        # Format conversation history
        history_text = ""
        if history:
            history_text = "Previous conversation:\n"
            for turn in history[-3:]:  # Last 3 turns
                history_text += f"User: {turn['question']}\n"
                history_text += f"Assistant: {turn['answer']}\n\n"
                
        prompt = f"""You are an expert customer service AI assistant. Think step by step to provide accurate, helpful answers.

Business Context:
{context}

Retrieved Information:
{info_text}

{history_text}

Customer Question: {question}

Please think through this step by step:
1. What is the customer really asking?
2. What information is most relevant?
3. Are there any unstated needs or implications?
4. What additional context would be helpful?
5. What's the most helpful response?

Let me work through this:
"""
        
        return prompt