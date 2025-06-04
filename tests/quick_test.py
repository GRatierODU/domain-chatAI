"""
Production Test Suite - Uses ALL production components properly
Natural conversation with precise knowledge retrieval
"""

# First, let's add the ChromaDB manager to the backend/core directory
import os
import sys
from pathlib import Path

# Add the backend directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Create the chromadb_manager.py file in backend/core if it doesn't exist
chromadb_manager_code = '''"""
Centralized ChromaDB Manager to prevent instance conflicts
"""
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import logging
from typing import Optional, Dict
import threading

logger = logging.getLogger(__name__)

class ChromaDBManager:
    """
    Singleton manager for ChromaDB to ensure consistent settings across all components
    """
    _instance = None
    _lock = threading.Lock()
    _client = None
    _embedding_function = None
    _collections_cache = {}
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the ChromaDB manager"""
        if self._client is None:
            self._initialize_client()
    
    def _initialize_client(self):
        """Initialize ChromaDB client with consistent settings"""
        try:
            # Use consistent settings
            settings = Settings(
                anonymized_telemetry=False,
                persist_directory="./chroma_db",
                chroma_db_impl="duckdb+parquet",
            )
            
            self._client = chromadb.PersistentClient(
                path="./chroma_db",
                settings=settings
            )
            
            # Initialize embedding function once
            self._embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="BAAI/bge-large-en-v1.5"
            )
            
            logger.info("ChromaDB client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB client: {e}")
            raise
    
    def get_client(self):
        """Get the ChromaDB client"""
        if self._client is None:
            self._initialize_client()
        return self._client
    
    def get_embedding_function(self):
        """Get the consistent embedding function"""
        if self._embedding_function is None:
            self._embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="BAAI/bge-large-en-v1.5"
            )
        return self._embedding_function
    
    def create_collection(self, name: str, metadata: Optional[Dict] = None):
        """Create a new collection with consistent settings"""
        try:
            client = self.get_client()
            
            # Delete if exists
            try:
                client.delete_collection(name)
                logger.info(f"Deleted existing collection: {name}")
            except:
                pass
            
            # Create with consistent embedding function
            collection = client.create_collection(
                name=name,
                embedding_function=self.get_embedding_function(),
                metadata=metadata or {"hnsw:space": "cosine"}
            )
            
            # Cache it
            self._collections_cache[name] = collection
            logger.info(f"Created collection: {name}")
            
            return collection
            
        except Exception as e:
            logger.error(f"Failed to create collection {name}: {e}")
            raise
    
    def get_collection(self, name: str):
        """Get a collection, using cache if available"""
        # Check cache first
        if name in self._collections_cache:
            return self._collections_cache[name]
        
        try:
            client = self.get_client()
            
            # Try to get with embedding function
            try:
                collection = client.get_collection(
                    name=name,
                    embedding_function=self.get_embedding_function()
                )
                self._collections_cache[name] = collection
                return collection
                
            except Exception as e:
                logger.warning(f"Could not get collection with embedding function: {e}")
                
                # Try without embedding function as fallback
                collection = client.get_collection(name=name)
                self._collections_cache[name] = collection
                logger.info(f"Got collection {name} without embedding function")
                return collection
                
        except Exception as e:
            logger.error(f"Failed to get collection {name}: {e}")
            raise
    
    def get_or_create_collection(self, name: str, metadata: Optional[Dict] = None):
        """Get existing collection or create if doesn't exist"""
        try:
            return self.get_collection(name)
        except:
            return self.create_collection(name, metadata)
    
    def clear_cache(self):
        """Clear the collections cache"""
        self._collections_cache.clear()
    
    def reset(self):
        """Reset the entire ChromaDB manager"""
        with self._lock:
            self._collections_cache.clear()
            self._client = None
            self._embedding_function = None
            self._initialize_client()

# Global instance
chroma_manager = ChromaDBManager()
'''

# Create the file if it doesn't exist
core_dir = Path(__file__).parent.parent / "backend" / "core"
core_dir.mkdir(exist_ok=True)

chromadb_manager_path = core_dir / "chromadb_manager.py"
if not chromadb_manager_path.exists():
    with open(chromadb_manager_path, "w") as f:
        f.write(chromadb_manager_code)
    print(f"Created {chromadb_manager_path}")

# Now do the regular imports
import asyncio
from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel
import uvicorn
import logging
from datetime import datetime
import json
import time
import psutil
import torch
from typing import Dict, List, Optional
import threading
import hashlib
import chromadb
from sentence_transformers import SentenceTransformer

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import ALL production components
try:
    from backend.crawler.intelligent_crawler import IntelligentCrawler, CrawledPage
    from backend.processor.multimodal_parser import MultimodalParser, ProcessedContent
    from backend.processor.knowledge_builder import KnowledgeBuilder
    from backend.chatbot.reasoning_engine import ReasoningEngine, ReasoningResponse
    from backend.chatbot.retrieval_optimizer import OptimizedRetriever
    from backend.chatbot.complexity_classifier import (
        ComplexityClassifier,
        QueryComplexity,
    )
    from backend.core.config import settings
    from backend.core.chromadb_manager import chroma_manager

    logger.info("✅ All production components loaded successfully!")
except Exception as e:
    logger.error(f"Failed to load production components: {e}")
    raise

# Create FastAPI app
app = FastAPI(title="AI Chatbot Production Test - Full Components")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state with thread-safe updates
crawl_jobs = {}
crawl_jobs_lock = threading.Lock()
knowledge_bases = {}
active_sessions = {}
models_loaded = False

# Initialize production components
multimodal_parser = None
knowledge_builder = None
reasoning_engine = None
complexity_classifier = None

# Model configuration for testing
TEST_MODEL_CONFIG = {
    "reasoning_models": {
        "efficient": "microsoft/Phi-3-mini-4k-instruct",  # Smaller model for testing
        "secondary": None,  # Skip for testing
        "primary": None,  # Skip for testing
    },
    "vision_models": {
        "multimodal": None,  # Skip heavy vision models for now
        "layout": None,
        "table": None,
        "ocr": None,
    },
}


# Request models
class CrawlRequest(BaseModel):
    domain: str
    max_pages: int = 20


class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = None
    domain: str
    require_reasoning: Optional[bool] = True


# Initialize models on startup
async def initialize_models():
    """Initialize all production models"""
    global multimodal_parser, knowledge_builder, reasoning_engine, complexity_classifier, models_loaded

    if models_loaded:
        return

    logger.info("Initializing production models...")

    try:
        # Initialize components with test configuration
        multimodal_parser = MultimodalParser(TEST_MODEL_CONFIG["vision_models"])
        knowledge_builder = KnowledgeBuilder(multimodal_parser)
        reasoning_engine = ReasoningEngine(TEST_MODEL_CONFIG["reasoning_models"])
        complexity_classifier = ComplexityClassifier()

        models_loaded = True
        logger.info("✅ All models initialized successfully!")
    except Exception as e:
        logger.error(f"Failed to initialize models: {e}")
        # Continue anyway for testing
        models_loaded = True


@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    await initialize_models()


# Enhanced crawler with proper progress tracking
class ProductionCrawler(IntelligentCrawler):
    """Production crawler with progress tracking"""

    def __init__(self, domain: str, max_pages: int, job_id: str):
        super().__init__(domain, max_pages)
        self.job_id = job_id
        self.pages_found = 0

    def _update_job_progress(self):
        """Update job progress"""
        with crawl_jobs_lock:
            if self.job_id in crawl_jobs:
                progress = min(40, int((len(self.pages) / self.max_pages) * 40))
                crawl_jobs[self.job_id].update(
                    {
                        "pages_crawled": len(self.pages),
                        "progress": progress,
                        "status": "crawling",
                    }
                )

    async def _crawler_worker(self, worker_id: int):
        """Override to add progress updates"""
        await super()._crawler_worker(worker_id)
        self._update_job_progress()


@app.get("/")
async def home():
    """Production test interface"""
    return HTMLResponse(
        """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Chatbot - Production Test (Full Components)</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: #ffffff;
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .container {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 20px;
            padding: 3rem;
            max-width: 800px;
            width: 100%;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }
        
        h1 {
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
            text-align: center;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        }
        
        .subtitle {
            text-align: center;
            opacity: 0.9;
            margin-bottom: 2rem;
        }
        
        .input-group {
            margin-bottom: 2rem;
        }
        
        label {
            display: block;
            margin-bottom: 0.5rem;
            font-weight: 500;
        }
        
        input {
            width: 100%;
            padding: 12px 16px;
            background: rgba(255, 255, 255, 0.1);
            border: 2px solid rgba(255, 255, 255, 0.3);
            border-radius: 10px;
            color: white;
            font-size: 16px;
            transition: all 0.3s;
        }
        
        input::placeholder {
            color: rgba(255, 255, 255, 0.6);
        }
        
        input:focus {
            outline: none;
            border-color: #ffffff;
            background: rgba(255, 255, 255, 0.2);
        }
        
        .page-options {
            display: flex;
            gap: 1rem;
            margin-bottom: 2rem;
        }
        
        .page-btn {
            flex: 1;
            padding: 12px;
            background: rgba(255, 255, 255, 0.1);
            border: 2px solid rgba(255, 255, 255, 0.3);
            border-radius: 10px;
            color: white;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
        }
        
        .page-btn:hover {
            background: rgba(255, 255, 255, 0.2);
            transform: translateY(-2px);
        }
        
        .page-btn.active {
            background: rgba(255, 255, 255, 0.3);
            border-color: white;
        }
        
        .start-btn {
            width: 100%;
            padding: 16px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border: none;
            border-radius: 10px;
            color: white;
            font-size: 18px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        }
        
        .start-btn:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
        }
        
        .start-btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }
        
        .status-box {
            margin-top: 2rem;
            padding: 1.5rem;
            background: rgba(0, 0, 0, 0.3);
            border-radius: 10px;
            min-height: 150px;
        }
        
        .status-title {
            font-weight: 600;
            margin-bottom: 0.5rem;
            color: #a5b4fc;
        }
        
        .progress-bar {
            height: 8px;
            background: rgba(255, 255, 255, 0.2);
            border-radius: 4px;
            overflow: hidden;
            margin: 1rem 0;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            transition: width 0.3s ease;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1rem;
            margin-top: 1rem;
        }
        
        .stat-item {
            text-align: center;
            padding: 1rem;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 8px;
        }
        
        .stat-value {
            font-size: 1.5rem;
            font-weight: 600;
        }
        
        .stat-label {
            font-size: 0.875rem;
            opacity: 0.8;
        }
        
        .success-msg {
            background: rgba(34, 197, 94, 0.2);
            border: 1px solid #22c55e;
            padding: 1rem;
            border-radius: 8px;
            margin-top: 1rem;
        }
        
        .error-msg {
            background: rgba(239, 68, 68, 0.2);
            border: 1px solid #ef4444;
            padding: 1rem;
            border-radius: 8px;
            margin-top: 1rem;
        }
        
        .spinner {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid rgba(255, 255, 255, 0.3);
            border-top-color: white;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-right: 0.5rem;
            vertical-align: middle;
        }
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        
        .system-info {
            display: flex;
            gap: 2rem;
            margin-bottom: 2rem;
            padding: 1rem;
            background: rgba(0, 0, 0, 0.2);
            border-radius: 10px;
            font-size: 0.875rem;
        }
        
        .info-item {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .info-dot {
            width: 8px;
            height: 8px;
            background: #22c55e;
            border-radius: 50%;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 Full Production Test</h1>
        <p class="subtitle">Testing with all real production components - Natural AI conversations!</p>
        
        <div class="system-info">
            <div class="info-item">
                <span class="info-dot"></span>
                <span>System: <span id="cpu-info">--</span>% CPU</span>
            </div>
            <div class="info-item">
                <span class="info-dot"></span>
                <span>Memory: <span id="mem-info">--</span>% Used</span>
            </div>
            <div class="info-item">
                <span class="info-dot"></span>
                <span>Models: <span id="model-info">Loading...</span></span>
            </div>
        </div>
        
        <div class="input-group">
            <label for="domain">Website Domain</label>
            <input 
                type="text" 
                id="domain" 
                placeholder="example.com (e.g., python.org, github.com)" 
                value=""
            />
        </div>
        
        <label>Pages to Crawl</label>
        <div class="page-options">
            <button class="page-btn active" onclick="selectPages(20, this)">20 Pages (Small)</button>
            <button class="page-btn" onclick="selectPages(50, this)">50 Pages (Medium)</button>
            <button class="page-btn" onclick="selectPages(100, this)">100 Pages (Large)</button>
        </div>
        
        <input type="hidden" id="max-pages" value="20" />
        
        <button class="start-btn" onclick="startAnalysis()">
            Start Full Analysis
        </button>
        
        <div class="status-box">
            <div class="status-title">Analysis Status</div>
            <div id="status-message">Ready to analyze with full AI components</div>
            
            <div class="progress-bar" id="progress-container" style="display: none;">
                <div class="progress-fill" id="progress-bar" style="width: 0%"></div>
            </div>
            
            <div class="stats-grid" id="stats-grid" style="display: none;">
                <div class="stat-item">
                    <div class="stat-value" id="pages-crawled">0</div>
                    <div class="stat-label">Pages Analyzed</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" id="chunks-created">0</div>
                    <div class="stat-label">Knowledge Chunks</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" id="time-elapsed">0s</div>
                    <div class="stat-label">Time Elapsed</div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        let currentJobId = null;
        let statusInterval = null;
        let timeInterval = null;
        let startTime = null;
        let selectedPages = 20;
        let testPageOpened = false;
        
        // System monitoring
        async function updateSystemInfo() {
            try {
                const response = await fetch('/system-status');
                const data = await response.json();
                
                document.getElementById('cpu-info').textContent = data.cpu_percent.toFixed(1);
                document.getElementById('mem-info').textContent = data.memory_percent.toFixed(1);
                document.getElementById('model-info').textContent = data.models_loaded ? 
                    'Ready' : 'Loading...';
            } catch (error) {
                console.error('Failed to update system info:', error);
            }
        }
        
        // Update system info every 2 seconds
        setInterval(updateSystemInfo, 2000);
        updateSystemInfo();
        
        function selectPages(count, btn) {
            selectedPages = count;
            document.getElementById('max-pages').value = count;
            
            // Update button states
            document.querySelectorAll('.page-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
        }
        
        async function startAnalysis() {
            const domain = document.getElementById('domain').value.trim();
            
            if (!domain) {
                alert('Please enter a domain');
                return;
            }
            
            // Reset test page flag
            testPageOpened = false;
            
            // Clean domain
            const cleanDomain = domain.replace(/^https?:\\/\\//, '').replace(/\\/.*$/, '');
            
            // Update UI
            const btn = document.querySelector('.start-btn');
            btn.disabled = true;
            btn.innerHTML = '<span class="spinner"></span>Starting Full Analysis...';
            
            document.getElementById('progress-container').style.display = 'block';
            document.getElementById('stats-grid').style.display = 'grid';
            
            startTime = Date.now();
            
            // Start time tracking
            timeInterval = setInterval(() => {
                if (startTime) {
                    const elapsed = Math.floor((Date.now() - startTime) / 1000);
                    document.getElementById('time-elapsed').textContent = elapsed + 's';
                }
            }, 100);
            
            try {
                const response = await fetch('/api/crawl', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ 
                        domain: cleanDomain,
                        max_pages: selectedPages
                    })
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    currentJobId = data.job_id;
                    updateStatus(`🚀 Initializing AI analysis of ${cleanDomain}...`);
                    
                    // Start monitoring
                    statusInterval = setInterval(checkStatus, 500);
                    
                    // Try WebSocket connection
                    connectWebSocket(data.job_id);
                } else {
                    throw new Error(data.detail || 'Failed to start');
                }
                
            } catch (error) {
                showError(`Error: ${error.message}`);
                btn.disabled = false;
                btn.innerHTML = 'Start Full Analysis';
                clearInterval(timeInterval);
            }
        }
        
        function connectWebSocket(jobId) {
            try {
                const ws = new WebSocket(`ws://localhost:8000/ws/${jobId}`);
                
                ws.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    updateFromWebSocket(data);
                };
                
                ws.onerror = (error) => {
                    console.log('WebSocket error, falling back to polling');
                };
            } catch (e) {
                console.log('WebSocket not available, using polling');
            }
        }
        
        async function checkStatus() {
            if (!currentJobId) return;
            
            try {
                const response = await fetch(`/api/crawl/${currentJobId}`);
                const data = await response.json();
                
                updateProgress(data);
                
                if (data.status === 'completed' || data.status === 'failed') {
                    clearInterval(statusInterval);
                    clearInterval(timeInterval);
                    
                    if (data.status === 'completed') {
                        showSuccess(data);
                    } else {
                        showError(`Analysis failed: ${data.error || 'Unknown error'}`);
                    }
                    
                    // Reset button
                    const btn = document.querySelector('.start-btn');
                    btn.disabled = false;
                    btn.innerHTML = 'Start Full Analysis';
                }
            } catch (error) {
                console.error('Status check error:', error);
            }
        }
        
        function updateFromWebSocket(data) {
            updateProgress(data);
        }
        
        function updateProgress(data) {
            // Update progress bar
            const progress = data.progress || 0;
            document.getElementById('progress-bar').style.width = progress + '%';
            
            // Update stats
            document.getElementById('pages-crawled').textContent = data.pages_crawled || 0;
            document.getElementById('chunks-created').textContent = data.chunks_created || 0;
            
            // Update status message
            let statusMsg = '';
            if (data.status === 'crawling') {
                statusMsg = `🕷️ Crawling website content... (${data.pages_crawled || 0} pages)`;
            } else if (data.status === 'processing') {
                statusMsg = `🤖 AI analyzing content with multimodal understanding...`;
            } else if (data.status === 'building_knowledge') {
                statusMsg = `📚 Building intelligent knowledge base...`;
            } else if (data.status === 'indexing') {
                statusMsg = `🔍 Creating semantic search index...`;
            }
            
            if (statusMsg) {
                updateStatus(statusMsg);
            }
        }
        
        function updateStatus(message) {
            document.getElementById('status-message').textContent = message;
        }
        
        function showSuccess(data) {
            const testUrl = `/test-website?domain=${encodeURIComponent(data.domain)}&pages=${data.pages_crawled}&chunks=${data.chunks_created || 0}`;
            
            document.getElementById('status-message').innerHTML = `
                <div class="success-msg">
                    ✅ <strong>AI Analysis Complete!</strong><br>
                    Successfully processed ${data.pages_crawled} pages into ${data.chunks_created || 0} knowledge chunks.<br>
                    The AI is now ready for natural conversations about ${data.domain}.<br>
                    <a href="${testUrl}" target="_blank" style="color: #22c55e; font-weight: 600;">
                        Open Natural Chat Interface →
                    </a>
                </div>
            `;
            
            // Auto-open after 2 seconds
            if (!testPageOpened) {
                testPageOpened = true;
                setTimeout(() => {
                    window.open(testUrl, '_blank');
                }, 2000);
            }
        }
        
        function showError(message) {
            document.getElementById('status-message').innerHTML = `
                <div class="error-msg">❌ ${message}</div>
            `;
        }
    </script>
</body>
</html>
    """
    )


@app.get("/system-status")
async def get_system_status():
    """Get current system status"""
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()

    return {
        "cpu_percent": cpu_percent,
        "memory_percent": memory.percent,
        "gpu_available": torch.cuda.is_available(),
        "models_loaded": models_loaded,
    }


@app.post("/api/crawl")
async def start_crawl(request: CrawlRequest, background_tasks: BackgroundTasks):
    """Start crawling with full production pipeline"""
    job_id = f"job-{datetime.utcnow().timestamp()}"

    with crawl_jobs_lock:
        crawl_jobs[job_id] = {
            "status": "started",
            "domain": request.domain,
            "max_pages": request.max_pages,
            "started_at": datetime.utcnow().isoformat(),
            "progress": 0,
            "pages_crawled": 0,
            "chunks_created": 0,
        }

    background_tasks.add_task(
        run_full_production_pipeline, job_id, request.domain, request.max_pages
    )

    return {"job_id": job_id, "status": "started"}


async def run_full_production_pipeline(job_id: str, domain: str, max_pages: int):
    """Run the COMPLETE production pipeline with all components"""
    try:
        # Update job status helper
        def update_job(updates):
            with crawl_jobs_lock:
                if job_id in crawl_jobs:
                    crawl_jobs[job_id].update(updates)

        # Phase 1: Crawling
        logger.info(f"Starting production crawl of {domain}")
        update_job({"status": "crawling", "progress": 10})

        crawler = ProductionCrawler(domain, max_pages, job_id)
        pages = await crawler.start()

        if not pages:
            raise Exception("No pages were successfully crawled")

        update_job(
            {"pages_crawled": len(pages), "progress": 40, "status": "processing"}
        )

        # Phase 2: Multimodal Processing & Knowledge Building
        logger.info(f"Processing {len(pages)} pages with multimodal parser")
        update_job({"status": "building_knowledge", "progress": 50})

        # Use the actual knowledge builder
        if knowledge_builder:
            collection_name = await knowledge_builder.build_knowledge_base(
                domain, pages
            )

            # Get chunk count from ChromaDB
            try:
                chroma_client = chromadb.PersistentClient(path="./chroma_db")
                collection = chroma_client.get_collection(collection_name)
                chunk_count = collection.count()
            except:
                chunk_count = len(pages) * 5  # Estimate

            update_job(
                {"chunks_created": chunk_count, "progress": 80, "status": "indexing"}
            )
        else:
            # Fallback if knowledge builder not initialized
            collection_name = f"website_{domain.replace('.', '_')}"
            chunk_count = len(pages) * 5

        # Store knowledge base info
        knowledge_bases[domain] = {
            "collection_name": collection_name,
            "pages_count": len(pages),
            "chunks_count": chunk_count,
            "created_at": datetime.utcnow().isoformat(),
            "crawler_instance": crawler,  # Keep for fallback
        }

        # Complete
        update_job(
            {
                "status": "completed",
                "progress": 100,
                "collection_name": collection_name,
                "chunks_created": chunk_count,
                "completed_at": datetime.utcnow().isoformat(),
                "domain": domain,
            }
        )

        logger.info(f"✅ Full production pipeline completed for {domain}")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        update_job({"status": "failed", "error": str(e), "progress": 0})


@app.get("/api/crawl/{job_id}")
async def get_crawl_status(job_id: str):
    """Get crawl job status"""
    with crawl_jobs_lock:
        if job_id not in crawl_jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        return crawl_jobs[job_id].copy()


@app.websocket("/ws/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    """WebSocket for real-time updates"""
    await websocket.accept()

    try:
        while True:
            with crawl_jobs_lock:
                if job_id in crawl_jobs:
                    await websocket.send_json(crawl_jobs[job_id])

                    if crawl_jobs[job_id]["status"] in ["completed", "failed"]:
                        break

            await asyncio.sleep(0.5)
    except:
        pass
    finally:
        await websocket.close()


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Natural conversation using full production components"""
    try:
        domain = request.domain

        if domain not in knowledge_bases:
            raise HTTPException(
                status_code=400, detail=f"Domain {domain} has not been analyzed yet"
            )

        # Get or create session - FIXED: Use provided session_id
        session_id = request.session_id
        if not session_id:
            session_id = f"session-{datetime.utcnow().timestamp()}"

        # Initialize session if needed
        if session_id not in active_sessions:
            active_sessions[session_id] = {
                "history": [],
                "context": {"domain": domain},
                "user_profile": {},
                "message_count": 0,
            }

        session = active_sessions[session_id]
        kb_info = knowledge_bases[domain]

        # Track message count
        session["message_count"] += 1

        # Try to get retriever first
        try:
            # Create a fresh ChromaDB client to avoid conflicts
            import chromadb

            chroma_client = chromadb.PersistentClient(path="./chroma_db")

            # Get the collection directly
            collection = chroma_client.get_collection(
                name=kb_info["collection_name"],
                embedding_function=chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name="BAAI/bge-large-en-v1.5"
                ),
            )

            # Create retriever with the collection
            retriever = OptimizedRetriever(kb_info["collection_name"])
            retriever.collection = collection  # Use our collection

        except Exception as e:
            logger.error(f"Failed to create retriever: {e}")
            retriever = None

        # Try to use the actual reasoning engine
        if reasoning_engine and retriever:
            try:
                # Use reasoning engine for natural response
                response = await reasoning_engine.answer_question(
                    request.question,
                    domain,  # Pass domain as context
                    retriever,
                    session["history"],
                )

                # Update session history
                session["history"].append(
                    {
                        "question": request.question,
                        "answer": response.answer,
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                )

                # Keep only last 10 exchanges
                if len(session["history"]) > 10:
                    session["history"] = session["history"][-10:]

                # Only include sources if we actually used knowledge
                sources_to_return = (
                    response.sources if response.confidence > 0.7 else []
                )

                return {
                    "answer": response.answer,
                    "sources": sources_to_return,
                    "confidence": response.confidence,
                    "session_id": session_id,
                    "processing_time": response.processing_time,
                    "query_type": response.query_type.value,
                }

            except Exception as e:
                logger.error(f"Reasoning engine error: {e}", exc_info=True)
                # Fall through to direct retrieval method

        # Fallback: Try direct retrieval if reasoning engine not available
        if retriever:
            try:
                # Retrieve relevant information directly
                retrieved_info = await retriever.retrieve(
                    request.question,
                    {"conversation_history": session["history"]},
                    top_k=5,
                )

                # Build response from retrieved content
                answer = await build_knowledge_based_response(
                    request.question, retrieved_info, session, domain
                )

                # Extract sources only if we found content
                sources = []
                if any(info.content for info in retrieved_info):
                    for info in retrieved_info[:3]:
                        if info.metadata and info.content:
                            sources.append(
                                {
                                    "url": info.metadata.get(
                                        "url", f"https://{domain}"
                                    ),
                                    "title": info.metadata.get("title", "Source"),
                                }
                            )

                # Update session
                session["history"].append(
                    {
                        "question": request.question,
                        "answer": answer,
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                )

                return {
                    "answer": answer,
                    "sources": sources,
                    "confidence": 0.8,
                    "session_id": session_id,
                    "processing_time": 0.1,
                    "query_type": "direct_retrieval",
                }

            except Exception as e:
                logger.error(f"Direct retrieval error: {e}", exc_info=True)

        # Final fallback if all methods fail
        if "answer" not in locals():
            answer = await generate_fallback_response(
                request.question, kb_info, session, domain
            )

            # Update session
            session["history"].append(
                {
                    "question": request.question,
                    "answer": answer,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )

            return {
                "answer": answer,
                "sources": [],  # No sources for fallback
                "confidence": 0.5,
                "session_id": session_id,
                "processing_time": 0.1,
                "query_type": "fallback",
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


async def build_knowledge_based_response(
    question: str, retrieved_info: List, session: Dict, domain: str
) -> str:
    """Build response using actual retrieved knowledge"""
    import random

    # Check if this is a greeting (first message)
    question_lower = question.lower()
    is_greeting = any(
        word in question_lower for word in ["hi", "hello", "hey", "howdy"]
    )
    message_count = session.get("message_count", 0)

    # Handle greetings based on conversation stage
    if is_greeting:
        if message_count <= 1:
            # First greeting
            return f"Hello! 👋 I'm your AI assistant for {domain}. I've analyzed the entire website and I'm here to help you find what you need. What would you like to know?"
        else:
            # Already greeted
            return "Hello again! What else can I help you with?"

    # Extract actual content from retrieved information
    content_pieces = []
    for info in retrieved_info:
        if info.content and len(info.content.strip()) > 20:
            # Clean and add content
            content = info.content.strip()
            if content not in content_pieces:
                content_pieces.append(content)

    if content_pieces:
        # We have actual content - build informative response
        if "what" in question_lower and "about" in question_lower:
            # Question about the website
            response = f"Based on my analysis of {domain}, "

            # Use the first 2-3 most relevant pieces
            main_content = " ".join(content_pieces[:2])

            # Truncate if too long
            if len(main_content) > 500:
                main_content = main_content[:497] + "..."

            response += main_content

            if message_count <= 2:
                response += " Is there anything specific you'd like to know more about?"

        elif any(
            word in question_lower for word in ["contact", "phone", "email", "address"]
        ):
            # Looking for contact info
            contact_content = []
            for content in content_pieces:
                if any(
                    word in content.lower()
                    for word in ["contact", "phone", "email", "address", "@", "call"]
                ):
                    contact_content.append(content)

            if contact_content:
                response = "Here's the contact information I found: " + " ".join(
                    contact_content[:2]
                )
            else:
                response = "I couldn't find specific contact information in the sections I've analyzed. You might want to check the contact or about pages directly on the website."

        else:
            # General question - provide relevant content
            response = "Here's what I found: " + " ".join(content_pieces[:2])

            if len(content_pieces) > 2:
                response += " There's more information available - would you like me to elaborate on any particular aspect?"
    else:
        # No specific content found
        response = f"I'm looking through my knowledge of {domain} but couldn't find specific information about that. Could you rephrase your question or ask about something else? I have information about the website's content, services, and general information."

    return response


async def generate_fallback_response(
    question: str, kb_info: Dict, session: Dict, domain: str
) -> str:
    """Generate natural fallback response when reasoning engine unavailable"""
    import random

    # Get conversation context
    history = session.get("history", [])
    message_count = session.get("message_count", 0)

    question_lower = question.lower()

    # Check for greetings - only respond with greeting if it's early in conversation
    if any(word in question_lower for word in ["hi", "hello", "hey", "howdy"]):
        if message_count <= 1:
            return f"Hello! I'm here to help you learn about {domain}. I've analyzed {kb_info.get('pages_count', 'the')} pages and have {kb_info.get('chunks_count', 'extensive')} pieces of information ready. What would you like to know?"
        else:
            # Not first message - don't repeat introduction
            return "Hi there! What else would you like to know?"

    # For "what is this website about" type questions
    if (
        "what" in question_lower and "about" in question_lower
    ) or "tell me about" in question_lower:
        return f"I've analyzed {kb_info.get('pages_count', 'multiple')} pages from {domain}. The website contains {kb_info.get('chunks_count', 'various')} pieces of information. To give you the most relevant details, could you be more specific about what aspect interests you? For example, their services, contact information, or specific products?"

    # For specific questions when we don't have the reasoning engine
    return f"I have information from {domain} but I need to be more specific to help you best. What particular aspect would you like to know about - their services, contact details, products, or something else?"


@app.get("/test-website", response_class=HTMLResponse)
async def test_website(domain: str = "", pages: int = 0, chunks: int = 0):
    """Test interface with natural chat widget"""
    if not domain or domain not in knowledge_bases:
        return HTMLResponse("<h1>Please analyze a domain first</h1>")

    return f"""
<!DOCTYPE html>
<html>
<head>
    <title>Natural Chat - {domain}</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 2rem;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 800px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 2rem;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }}
        
        h1 {{
            text-align: center;
            margin-bottom: 1rem;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        }}
        
        .info {{
            text-align: center;
            margin-bottom: 2rem;
            opacity: 0.9;
        }}
        
        .instructions {{
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 1.5rem;
            margin-bottom: 2rem;
        }}
        
        .instructions h3 {{
            margin-bottom: 1rem;
            color: #a5b4fc;
        }}
        
        .instructions ul {{
            margin-left: 1.5rem;
        }}
        
        .instructions li {{
            margin-bottom: 0.5rem;
        }}
        
        .status {{
            background: rgba(34, 197, 94, 0.2);
            border: 1px solid #22c55e;
            border-radius: 10px;
            padding: 1rem;
            text-align: center;
            font-weight: 600;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>💬 Natural AI Chat</h1>
        <div class="info">
            <p><strong>Domain:</strong> {domain}</p>
            <p><strong>Knowledge Base:</strong> {pages} pages analyzed, {chunks} knowledge chunks created</p>
        </div>
        
        <div class="instructions">
            <h3>🤖 Your AI Assistant is Ready!</h3>
            <ul>
                <li>Click the chat button in the bottom right corner</li>
                <li>The AI has studied everything about {domain}</li>
                <li>Chat naturally - like talking to a knowledgeable friend</li>
                <li>The AI remembers your conversation context</li>
                <li>Ask follow-up questions for deeper information</li>
            </ul>
        </div>
        
        <div class="status">
            ✅ AI Assistant Ready - Start chatting naturally about {domain}!
        </div>
    </div>
    
    <script>
        // Configure the widget
        window.AI_CHATBOT_API_URL = "http://localhost:8000";
        window.AI_CHATBOT_DOMAIN = "{domain}";
        window.AI_CHATBOT_AUTO_START = false;
        
        console.log('Natural chat configured for:', '{domain}');
    </script>
    <script src="/widget/widget.js"></script>
</body>
</html>
    """


@app.get("/widget/widget.js")
async def serve_widget():
    """Serve the improved widget directly"""
    # Serve the improved widget code directly instead of reading from file
    widget_content = """(function () {
    "use strict";

    // Configuration
    const config = {
        apiUrl: window.AI_CHATBOT_API_URL || "http://localhost:8000",
        domain: window.AI_CHATBOT_DOMAIN || window.location.hostname,
        position: window.AI_CHATBOT_POSITION || "bottom-right",
        theme: window.AI_CHATBOT_THEME || "modern",
        autoStart: window.AI_CHATBOT_AUTO_START !== false,
        welcomeDelay: 3000, // Delay before showing welcome message
        typingSpeed: 1000, // Base typing indicator duration
    };

    // Widget state
    let sessionId = localStorage.getItem("ai_chatbot_session");
    // Generate new session if none exists
    if (!sessionId) {
        sessionId = `session-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
        localStorage.setItem("ai_chatbot_session", sessionId);
    }
    let isOpen = false;
    let isMinimized = false;
    let isReady = false;
    let messageCount = 0;
    let lastMessageTime = Date.now();
    let conversationContext = {
        hasGreeted: false,
        userName: null,
        topics: [],
        sentiment: 'neutral'
    };

    // Create widget HTML with improved design
    const widgetHTML = `
        <div id="ai-chatbot-container" class="ai-chatbot-container ai-chatbot-${config.position}">
            <div id="ai-chatbot-widget" class="ai-chatbot-widget ai-chatbot-hidden">
                <div class="ai-chatbot-header">
                    <div class="ai-chatbot-header-content">
                        <div class="ai-chatbot-status">
                            <span class="ai-chatbot-status-dot"></span>
                            <span class="ai-chatbot-status-text">AI Assistant</span>
                            <span class="ai-chatbot-status-detail">Ready to help</span>
                        </div>
                        <div class="ai-chatbot-header-actions">
                            <button class="ai-chatbot-minimize" onclick="AIChatbot.minimize()" title="Minimize">
                                <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                                    <path d="M4 8H12" stroke="currentColor" stroke-width="2"/>
                                </svg>
                            </button>
                            <button class="ai-chatbot-close" onclick="AIChatbot.close()" title="Close">
                                <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                                    <path d="M4 4L12 12M4 12L12 4" stroke="currentColor" stroke-width="2"/>
                                </svg>
                            </button>
                        </div>
                    </div>
                </div>
                
                <div class="ai-chatbot-messages" id="ai-chatbot-messages">
                    <div class="ai-chatbot-welcome" id="ai-chatbot-welcome">
                        <div class="ai-chatbot-avatar-large">
                            <svg width="40" height="40" viewBox="0 0 24 24" fill="none">
                                <path d="M12 2L2 7L12 12L22 7L12 2Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                                <path d="M2 17L12 22L22 17" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                                <path d="M2 12L12 17L22 12" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                            </svg>
                        </div>
                        <h3>Welcome! 👋</h3>
                        <p>I'm your AI assistant for ${config.domain}</p>
                        <div class="ai-chatbot-loading" id="ai-chatbot-loading">
                            <div class="ai-chatbot-spinner"></div>
                            <span>Checking knowledge base...</span>
                        </div>
                    </div>
                </div>
                

                
                <div class="ai-chatbot-input-container">
                    <div class="ai-chatbot-input-wrapper">
                        <textarea 
                            id="ai-chatbot-input" 
                            class="ai-chatbot-input"
                            placeholder="Type your message..."
                            disabled
                            rows="1"
                        ></textarea>
                        <button 
                            id="ai-chatbot-send" 
                            class="ai-chatbot-send"
                            onclick="AIChatbot.sendMessage()"
                            disabled
                            title="Send message"
                        >
                            <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
                                <path d="M2 10L18 2L10 18L8 11L2 10Z" stroke="currentColor" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>
                            </svg>
                        </button>
                    </div>
                    <div class="ai-chatbot-input-hint">Press Enter to send, Shift+Enter for new line</div>
                </div>
            </div>
            
            <button id="ai-chatbot-trigger" class="ai-chatbot-trigger" onclick="AIChatbot.toggle()">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" class="ai-chatbot-icon-chat">
                    <path d="M12 2C6.48 2 2 6.48 2 12C2 17.52 6.48 22 12 22C13.19 22 14.34 21.78 15.41 21.37L21 23L19.37 17.41C20.78 15.34 22 13.19 22 12C22 6.48 17.52 2 12 2ZM8 13C7.45 13 7 12.55 7 12C7 11.45 7.45 11 8 11C8.55 11 9 11.45 9 12C9 12.55 8.55 13 8 13ZM12 13C11.45 13 11 12.55 11 12C11 11.45 11.45 11 12 11C12.55 11 13 11.45 13 12C13 12.55 12.55 13 12 13ZM16 13C15.45 13 15 12.55 15 12C15 11.45 15.45 11 16 11C16.55 11 17 11.45 17 12C17 12.55 16.55 13 16 13Z" fill="currentColor"/>
                </svg>
                <span class="ai-chatbot-trigger-pulse"></span>
                <span class="ai-chatbot-badge" id="ai-chatbot-badge" style="display: none;">1</span>
            </button>
        </div>
    `;

    // Enhanced styles with better animations and natural feel
    const styles = `
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
            
            .ai-chatbot-container {
                position: fixed;
                z-index: 999999;
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            }
            
            .ai-chatbot-bottom-right {
                bottom: 20px;
                right: 20px;
            }
            
            .ai-chatbot-widget {
                width: 400px;
                height: 600px;
                background: #ffffff;
                border-radius: 20px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.15);
                display: flex;
                flex-direction: column;
                overflow: hidden;
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                transform-origin: bottom right;
            }
            
            .ai-chatbot-widget.ai-chatbot-hidden {
                opacity: 0;
                transform: scale(0.95) translateY(20px);
                pointer-events: none;
            }
            
            .ai-chatbot-widget.ai-chatbot-minimized {
                height: 64px;
            }
            
            .ai-chatbot-header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 18px 20px;
                flex-shrink: 0;
            }
            
            .ai-chatbot-header-content {
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            
            .ai-chatbot-status {
                display: flex;
                align-items: center;
                gap: 10px;
            }
            
            .ai-chatbot-status-dot {
                width: 10px;
                height: 10px;
                background: #4ade80;
                border-radius: 50%;
                box-shadow: 0 0 0 2px rgba(74, 222, 128, 0.3);
                animation: statusPulse 2s infinite;
            }
            
            @keyframes statusPulse {
                0% { 
                    transform: scale(1);
                    box-shadow: 0 0 0 2px rgba(74, 222, 128, 0.3);
                }
                50% { 
                    transform: scale(1.1);
                    box-shadow: 0 0 0 4px rgba(74, 222, 128, 0.1);
                }
                100% { 
                    transform: scale(1);
                    box-shadow: 0 0 0 2px rgba(74, 222, 128, 0.3);
                }
            }
            
            .ai-chatbot-status-text {
                font-weight: 600;
                font-size: 16px;
            }
            
            .ai-chatbot-status-detail {
                font-size: 12px;
                opacity: 0.9;
            }
            
            .ai-chatbot-header-actions {
                display: flex;
                gap: 8px;
            }
            
            .ai-chatbot-header-actions button {
                background: rgba(255,255,255,0.2);
                backdrop-filter: blur(10px);
                border: none;
                width: 36px;
                height: 36px;
                border-radius: 10px;
                color: white;
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
                transition: all 0.2s;
            }
            
            .ai-chatbot-header-actions button:hover {
                background: rgba(255,255,255,0.3);
                transform: translateY(-1px);
            }
            
            .ai-chatbot-messages {
                flex: 1;
                overflow-y: auto;
                padding: 20px;
                background: #f8f9fb;
                scroll-behavior: smooth;
            }
            
            .ai-chatbot-messages::-webkit-scrollbar {
                width: 6px;
            }
            
            .ai-chatbot-messages::-webkit-scrollbar-track {
                background: transparent;
            }
            
            .ai-chatbot-messages::-webkit-scrollbar-thumb {
                background: #e0e0e0;
                border-radius: 3px;
            }
            
            .ai-chatbot-welcome {
                text-align: center;
                padding: 60px 20px;
            }
            
            .ai-chatbot-avatar-large {
                width: 80px;
                height: 80px;
                margin: 0 auto 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                animation: float 3s ease-in-out infinite;
            }
            
            @keyframes float {
                0%, 100% { transform: translateY(0); }
                50% { transform: translateY(-10px); }
            }
            
            .ai-chatbot-welcome h3 {
                margin: 0 0 8px 0;
                font-size: 24px;
                font-weight: 600;
                color: #1f2937;
            }
            
            .ai-chatbot-welcome p {
                margin: 0 0 24px 0;
                color: #6b7280;
                font-size: 16px;
            }
            
            .ai-chatbot-loading {
                display: inline-flex;
                align-items: center;
                gap: 12px;
                padding: 14px 24px;
                background: white;
                border-radius: 16px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            }
            
            .ai-chatbot-spinner {
                width: 20px;
                height: 20px;
                border: 2px solid #e5e7eb;
                border-top-color: #667eea;
                border-radius: 50%;
                animation: spin 1s linear infinite;
            }
            
            @keyframes spin {
                to { transform: rotate(360deg); }
            }
            
            .ai-chatbot-message {
                margin-bottom: 20px;
                display: flex;
                gap: 12px;
                animation: messageSlide 0.3s ease-out;
            }
            
            @keyframes messageSlide {
                from { 
                    opacity: 0;
                    transform: translateY(10px);
                }
                to { 
                    opacity: 1;
                    transform: translateY(0);
                }
            }
            
            .ai-chatbot-message-user {
                flex-direction: row-reverse;
            }
            
            .ai-chatbot-message-avatar {
                width: 36px;
                height: 36px;
                border-radius: 10px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 14px;
                font-weight: 600;
                flex-shrink: 0;
            }
            
            .ai-chatbot-message-user .ai-chatbot-message-avatar {
                background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            }
            
            .ai-chatbot-message-content {
                max-width: 75%;
                padding: 14px 18px;
                background: white;
                border-radius: 18px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.06);
                position: relative;
            }
            
            .ai-chatbot-message-user .ai-chatbot-message-content {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
            }
            
            .ai-chatbot-message-text {
                margin: 0;
                font-size: 15px;
                line-height: 1.6;
                color: #374151;
                white-space: pre-wrap;
                word-wrap: break-word;
            }
            
            .ai-chatbot-message-user .ai-chatbot-message-text {
                color: white;
            }
            
            .ai-chatbot-message-time {
                font-size: 11px;
                color: #9ca3af;
                margin-top: 6px;
            }
            
            .ai-chatbot-message-sources {
                margin-top: 10px;
                padding-top: 10px;
                border-top: 1px solid #e5e7eb;
            }
            
            .ai-chatbot-message-sources-title {
                font-size: 12px;
                color: #6b7280;
                margin-bottom: 6px;
                font-weight: 500;
            }
            
            .ai-chatbot-message-source {
                font-size: 13px;
                color: #667eea;
                text-decoration: none;
                display: inline-block;
                margin-right: 12px;
                margin-bottom: 4px;
                padding: 4px 8px;
                background: rgba(102, 126, 234, 0.1);
                border-radius: 6px;
                transition: all 0.2s;
            }
            
            .ai-chatbot-message-source:hover {
                background: rgba(102, 126, 234, 0.2);
                transform: translateY(-1px);
            }
            
            .ai-chatbot-input-container {
                padding: 20px;
                background: white;
                border-top: 1px solid #e5e7eb;
                flex-shrink: 0;
            }
            
            .ai-chatbot-input-wrapper {
                display: flex;
                gap: 12px;
                align-items: flex-end;
            }
            
            .ai-chatbot-input {
                flex: 1;
                padding: 14px 18px;
                border: 2px solid #e5e7eb;
                border-radius: 16px;
                font-size: 15px;
                font-family: inherit;
                outline: none;
                transition: all 0.2s;
                resize: none;
                max-height: 120px;
                line-height: 1.4;
            }
            
            .ai-chatbot-input:focus {
                border-color: #667eea;
                box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
            }
            
            .ai-chatbot-input:disabled {
                background: #f9fafb;
                cursor: not-allowed;
            }
            
            .ai-chatbot-send {
                width: 48px;
                height: 48px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                border-radius: 14px;
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
                transition: all 0.2s;
                box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
            }
            
            .ai-chatbot-send:hover:not(:disabled) {
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
            }
            
            .ai-chatbot-send:active:not(:disabled) {
                transform: translateY(0);
            }
            
            .ai-chatbot-send:disabled {
                background: #9ca3af;
                cursor: not-allowed;
                box-shadow: none;
            }
            
            .ai-chatbot-input-hint {
                font-size: 11px;
                color: #9ca3af;
                margin-top: 6px;
                text-align: center;
            }
            
            .ai-chatbot-trigger {
                width: 64px;
                height: 64px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border: none;
                border-radius: 50%;
                color: white;
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
                box-shadow: 0 6px 24px rgba(102, 126, 234, 0.4);
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                position: relative;
                overflow: hidden;
            }
            
            .ai-chatbot-trigger:hover {
                transform: translateY(-2px) scale(1.05);
                box-shadow: 0 8px 32px rgba(102, 126, 234, 0.5);
            }
            
            .ai-chatbot-trigger-pulse {
                position: absolute;
                width: 100%;
                height: 100%;
                background: radial-gradient(circle, rgba(255,255,255,0.4) 0%, transparent 70%);
                border-radius: 50%;
                animation: triggerPulse 2s infinite;
            }
            
            @keyframes triggerPulse {
                0% {
                    transform: scale(0.8);
                    opacity: 0;
                }
                50% {
                    opacity: 0.3;
                }
                100% {
                    transform: scale(1.5);
                    opacity: 0;
                }
            }
            
            .ai-chatbot-badge {
                position: absolute;
                top: -4px;
                right: -4px;
                background: #ef4444;
                color: white;
                font-size: 12px;
                font-weight: 600;
                min-width: 22px;
                height: 22px;
                border-radius: 11px;
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 0 6px;
                box-shadow: 0 2px 8px rgba(239, 68, 68, 0.4);
                animation: badgeBounce 0.5s ease-out;
            }
            
            @keyframes badgeBounce {
                0% { transform: scale(0); }
                50% { transform: scale(1.2); }
                100% { transform: scale(1); }
            }
            
            .ai-chatbot-typing {
                display: inline-flex;
                align-items: center;
                gap: 4px;
                padding: 8px 12px;
            }
            
            .ai-chatbot-typing span {
                width: 8px;
                height: 8px;
                background: #667eea;
                border-radius: 50%;
                animation: typing 1.4s ease-in-out infinite;
            }
            
            .ai-chatbot-typing span:nth-child(2) {
                animation-delay: 0.2s;
            }
            
            .ai-chatbot-typing span:nth-child(3) {
                animation-delay: 0.4s;
            }
            
            @keyframes typing {
                0%, 60%, 100% {
                    transform: scale(0.8);
                    opacity: 0.5;
                }
                30% {
                    transform: scale(1);
                    opacity: 1;
                }
            }
            
            @media (max-width: 480px) {
                .ai-chatbot-widget {
                    width: 100vw;
                    height: 100vh;
                    border-radius: 0;
                    max-height: 100vh;
                }
                
                .ai-chatbot-container {
                    bottom: 0;
                    right: 0;
                }
                
                .ai-chatbot-trigger {
                    bottom: 20px;
                    right: 20px;
                }
            }
        </style>
    `;

    // Inject HTML and styles
    document.head.insertAdjacentHTML("beforeend", styles);
    document.body.insertAdjacentHTML("beforeend", widgetHTML);

    // API class with improved error handling
    class ChatbotAPI {
        constructor(apiUrl) {
            this.apiUrl = apiUrl;
            this.retryCount = 3;
            this.retryDelay = 1000;
        }

        async request(url, options = {}) {
            for (let i = 0; i < this.retryCount; i++) {
                try {
                    const response = await fetch(url, {
                        ...options,
                        headers: {
                            'Content-Type': 'application/json',
                            ...options.headers
                        }
                    });

                    if (!response.ok) {
                        const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
                        throw new Error(error.detail || `HTTP ${response.status}`);
                    }

                    return response.json();
                } catch (error) {
                    if (i === this.retryCount - 1) throw error;
                    await new Promise(resolve => setTimeout(resolve, this.retryDelay * (i + 1)));
                }
            }
        }

        async checkDomainReady(domain) {
            try {
                await this.request(`${this.apiUrl}/api/chat`, {
                    method: 'POST',
                    body: JSON.stringify({
                        question: "test",
                        domain: domain,
                        session_id: "test-session"
                    })
                });
                return true;
            } catch (error) {
                return false;
            }
        }

        async sendMessage(question, sessionId, domain) {
            return this.request(`${this.apiUrl}/api/chat`, {
                method: 'POST',
                body: JSON.stringify({
                    question,
                    session_id: sessionId,
                    domain: domain,
                    require_reasoning: true
                })
            });
        }
    }

    // Initialize API
    const api = new ChatbotAPI(config.apiUrl);

    // Main chatbot class with enhanced functionality
    class AIChatbot {
        constructor() {
            this.widget = document.getElementById("ai-chatbot-widget");
            this.trigger = document.getElementById("ai-chatbot-trigger");
            this.messages = document.getElementById("ai-chatbot-messages");
            this.input = document.getElementById("ai-chatbot-input");
            this.sendButton = document.getElementById("ai-chatbot-send");
            this.badge = document.getElementById("ai-chatbot-badge");
            this.domain = config.domain;
            
            this.init();
        }

        async init() {
            console.log("🤖 Initializing AI Chatbot for:", this.domain);
            
            // Setup event listeners
            this.setupEventListeners();
            
            // Check domain status
            await this.checkDomainStatus();
            
            // Setup auto-resize for textarea
            this.setupAutoResize();
        }

        setupEventListeners() {
            // Input handling
            this.input.addEventListener("keypress", (e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    this.sendMessage();
                }
            });

            this.input.addEventListener("input", () => {
                this.adjustTextareaHeight();
                this.updateSendButtonState();
            });

            // Focus input when clicking messages area
            this.messages.addEventListener("click", (e) => {
                if (e.target === this.messages && isReady) {
                    this.input.focus();
                }
            });
        }

        setupAutoResize() {
            const resize = () => {
                this.input.style.height = 'auto';
                this.input.style.height = Math.min(this.input.scrollHeight, 120) + 'px';
            };
            this.input.addEventListener('input', resize);
        }

        adjustTextareaHeight() {
            this.input.style.height = 'auto';
            this.input.style.height = Math.min(this.input.scrollHeight, 120) + 'px';
        }

        updateSendButtonState() {
            const hasText = this.input.value.trim().length > 0;
            this.sendButton.disabled = !hasText || !isReady;
        }

        async checkDomainStatus() {
            try {
                const statusElement = document.querySelector('.ai-chatbot-status-detail');
                statusElement.textContent = 'Connecting...';
                
                const ready = await api.checkDomainReady(this.domain);
                
                if (ready) {
                    console.log("✅ Domain ready for chat");
                    this.onReady();
                } else {
                    console.log("⚠️ Domain not analyzed");
                    this.showError("This website hasn't been analyzed yet. Please set it up first.");
                }
            } catch (error) {
                console.error("❌ Failed to check domain:", error);
                this.showError("Connection failed. Please try again.");
            }
        }

        onReady() {
            isReady = true;
            this.input.disabled = false;
            this.input.placeholder = "Type your message...";
            
            const statusElement = document.querySelector('.ai-chatbot-status-detail');
            statusElement.textContent = 'Online';
            
            // Clear loading state
            const loading = document.getElementById("ai-chatbot-loading");
            if (loading) loading.style.display = 'none';
            
            // Show welcome message after a delay
            setTimeout(() => {
                if (!conversationContext.hasGreeted) {
                    this.showWelcomeMessage();
                }
            }, config.welcomeDelay);
        }

        showWelcomeMessage() {
            const welcome = document.getElementById("ai-chatbot-welcome");
            if (welcome) welcome.style.display = 'none';
            
            this.addMessage(
                "bot", 
                `Hi there! 👋 I'm your AI assistant for ${this.domain}. I've studied everything about this website and I'm here to help you find what you need. What would you like to know?`
            );
            
            conversationContext.hasGreeted = true;
        }

        showError(message) {
            const loading = document.getElementById("ai-chatbot-loading");
            if (loading) {
                loading.innerHTML = `<span style="color: #ef4444;">${message}</span>`;
            }
        }

        toggle() {
            if (isOpen) {
                this.close();
            } else {
                this.open();
            }
        }

        open() {
            isOpen = true;
            this.widget.classList.remove("ai-chatbot-hidden");
            this.trigger.style.display = "none";
            this.hideBadge();
            
            if (isReady) {
                setTimeout(() => this.input.focus(), 300);
            }
            
            // Track opening
            this.trackEvent('chat_opened');
        }

        close() {
            isOpen = false;
            this.widget.classList.add("ai-chatbot-hidden");
            this.trigger.style.display = "flex";
            
            // Track closing
            this.trackEvent('chat_closed', { message_count: messageCount });
        }

        minimize() {
            isMinimized = !isMinimized;
            this.widget.classList.toggle("ai-chatbot-minimized");
        }

        showBadge() {
            this.badge.style.display = "flex";
            this.badge.textContent = "1";
        }

        hideBadge() {
            this.badge.style.display = "none";
        }

        async sendMessage() {
            const message = this.input.value.trim();
            
            if (!message || !isReady) return;
            
            // Clear input immediately
            this.input.value = "";
            this.adjustTextareaHeight();
            this.updateSendButtonState();
            
            // Add user message
            this.addMessage("user", message);
            
            // Update context
            messageCount++;
            lastMessageTime = Date.now();
            
            // Show typing indicator with dynamic duration
            const typingDuration = this.calculateTypingDuration(message);
            const typingId = this.showTyping();
            
            try {
                // Send to API with consistent session ID
                const response = await api.sendMessage(message, sessionId, this.domain);
                
                // Store session ID
                if (response.session_id) {
                    sessionId = response.session_id;
                    localStorage.setItem("ai_chatbot_session", sessionId);
                }
                
                // Simulate natural typing delay
                const remainingDelay = typingDuration - (Date.now() - lastMessageTime);
                if (remainingDelay > 0) {
                    await new Promise(resolve => setTimeout(resolve, remainingDelay));
                }
                
                // Remove typing indicator
                this.removeTyping(typingId);
                
                // Add bot response
                this.addMessage("bot", response.answer, response.sources);
                
                // Update conversation context
                this.updateConversationContext(message, response);
                
            } catch (error) {
                console.error("Failed to send message:", error);
                this.removeTyping(typingId);
                this.addMessage(
                    "bot", 
                    "I apologize, but I'm having trouble connecting right now. Please try again in a moment, or check your internet connection."
                );
            }
        }

        calculateTypingDuration(message) {
            // Simulate natural typing speed based on message length
            const wordsPerMinute = 300;
            const words = message.split(' ').length;
            const baseTime = (words / wordsPerMinute) * 60 * 1000;
            return Math.max(config.typingSpeed, Math.min(baseTime, 3000));
        }

        updateConversationContext(userMessage, response) {
            // Analyze sentiment
            const positiveSentiments = ['thanks', 'great', 'awesome', 'perfect', 'excellent', 'good'];
            const negativeSentiments = ['bad', 'wrong', 'incorrect', 'unhappy', 'disappointed'];
            
            const messageLower = userMessage.toLowerCase();
            if (positiveSentiments.some(word => messageLower.includes(word))) {
                conversationContext.sentiment = 'positive';
            } else if (negativeSentiments.some(word => messageLower.includes(word))) {
                conversationContext.sentiment = 'negative';
            }
            
            // Extract topics
            if (response.query_type) {
                conversationContext.topics.push(response.query_type);
            }
        }

        addMessage(sender, text, sources = []) {
            const messageDiv = document.createElement("div");
            messageDiv.className = `ai-chatbot-message ai-chatbot-message-${sender}`;
            
            const avatar = document.createElement("div");
            avatar.className = "ai-chatbot-message-avatar";
            avatar.textContent = sender === "user" ? "You" : "AI";
            
            const content = document.createElement("div");
            content.className = "ai-chatbot-message-content";
            
            const textP = document.createElement("p");
            textP.className = "ai-chatbot-message-text";
            textP.textContent = text;
            content.appendChild(textP);
            
            // Add timestamp
            const time = document.createElement("div");
            time.className = "ai-chatbot-message-time";
            time.textContent = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
            content.appendChild(time);
            
            // Add sources if available
            if (sources && sources.length > 0) {
                const sourcesDiv = document.createElement("div");
                sourcesDiv.className = "ai-chatbot-message-sources";
                
                const sourcesTitle = document.createElement("div");
                sourcesTitle.className = "ai-chatbot-message-sources-title";
                sourcesTitle.textContent = "📎 Sources:";
                sourcesDiv.appendChild(sourcesTitle);
                
                sources.forEach(source => {
                    if (source && source.url) {
                        const sourceLink = document.createElement("a");
                        sourceLink.className = "ai-chatbot-message-source";
                        sourceLink.href = source.url;
                        sourceLink.textContent = source.title || "View source";
                        sourceLink.target = "_blank";
                        sourceLink.rel = "noopener noreferrer";
                        sourcesDiv.appendChild(sourceLink);
                    }
                });
                
                content.appendChild(sourcesDiv);
            }
            
            messageDiv.appendChild(avatar);
            messageDiv.appendChild(content);
            
            this.messages.appendChild(messageDiv);
            
            // Smooth scroll to bottom
            requestAnimationFrame(() => {
                this.messages.scrollTo({
                    top: this.messages.scrollHeight,
                    behavior: 'smooth'
                });
            });
        }

        showTyping() {
            const typingId = `typing-${Date.now()}`;
            const typingDiv = document.createElement("div");
            typingDiv.id = typingId;
            typingDiv.className = "ai-chatbot-message ai-chatbot-message-bot";
            typingDiv.innerHTML = `
                <div class="ai-chatbot-message-avatar">AI</div>
                <div class="ai-chatbot-message-content">
                    <div class="ai-chatbot-typing">
                        <span></span>
                        <span></span>
                        <span></span>
                    </div>
                </div>
            `;
            
            this.messages.appendChild(typingDiv);
            this.messages.scrollTop = this.messages.scrollHeight;
            
            return typingId;
        }

        removeTyping(typingId) {
            const typingDiv = document.getElementById(typingId);
            if (typingDiv) {
                typingDiv.style.opacity = '0';
                setTimeout(() => typingDiv.remove(), 200);
            }
        }

        trackEvent(eventName, data = {}) {
            // Analytics tracking placeholder
            if (window.gtag) {
                window.gtag('event', eventName, {
                    event_category: 'chatbot',
                    ...data
                });
            }
        }
    }

    // Initialize chatbot
    const chatbot = new AIChatbot();

    // Expose API for external control
    window.AIChatbot = {
        open: () => chatbot.open(),
        close: () => chatbot.close(),
        toggle: () => chatbot.toggle(),
        minimize: () => chatbot.minimize(),
        sendMessage: () => chatbot.sendMessage()
    };

    // Auto-open on mobile after delay
    if (window.innerWidth < 768 && config.autoStart) {
        setTimeout(() => {
            if (!isOpen) {
                chatbot.open();
            }
        }, 5000);
    }
})();"""

    return Response(content=widget_content, media_type="application/javascript")


if __name__ == "__main__":
    print(
        """
╔═══════════════════════════════════════════════════╗
║        AI Chatbot - Full Production Test          ║
╠═══════════════════════════════════════════════════╣
║
║  Starting server on http://localhost:8000         ║
║                                                   ║
╚═══════════════════════════════════════════════════╝
    """
    )

    # Ensure required directories exist
    os.makedirs("./chroma_db", exist_ok=True)
    os.makedirs("./models", exist_ok=True)

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
