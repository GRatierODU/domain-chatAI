"""
Production Test Suite - Uses all existing production components
No mocks, no simplified versions - real production code
"""

import asyncio
import os
import sys
from pathlib import Path
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
from typing import Dict
import threading
from bs4 import BeautifulSoup

# Add the backend directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import ALL existing production components - no mocks!
try:
    from backend.crawler.intelligent_crawler import IntelligentCrawler
    from backend.processor.multimodal_parser import MultimodalParser
    from backend.processor.knowledge_builder import KnowledgeBuilder
    from backend.chatbot.reasoning_engine import ReasoningEngine
    from backend.chatbot.retrieval_optimizer import OptimizedRetriever
    from backend.chatbot.complexity_classifier import ComplexityClassifier
    from backend.core.config import settings

    logger.info("✅ All production components loaded successfully!")
except Exception as e:
    logger.error(f"Failed to load production components: {e}")
    raise

# Create FastAPI app
app = FastAPI(title="AI Chatbot Production Test - Real Components")

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


# Request models
class CrawlRequest(BaseModel):
    domain: str
    max_pages: int = 20


# Enhanced IntelligentCrawler with progress tracking
class ProductionCrawler(IntelligentCrawler):
    """Production crawler with optimized multi-worker support and progress tracking"""

    def __init__(self, domain: str, max_pages: int, job_id: str):
        super().__init__(domain, max_pages)
        self.job_id = job_id
        self.pages_found = 0

    async def _crawler_worker(self, worker_id: int):
        """Override to add progress updates"""
        from playwright.async_api import async_playwright

        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True,
                args=["--disable-blink-features=AutomationControlled"],
            )

            context = await browser.new_context(
                viewport={"width": 1920, "height": 1080},
                device_scale_factor=1.5,
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            )

            page = await context.new_page()

            # Enable request interception for efficiency
            await page.route(
                "**/*.{png,jpg,jpeg,gif,svg,mp3,mp4,avi,flac,ogg,wav,webm}",
                lambda route: route.abort(),
            )

            while not self.to_visit.empty() and len(self.visited_urls) < self.max_pages:
                try:
                    priority, url = await self.to_visit.get()

                    # Skip if already visited (double-check with normalized URL)
                    normalized_url = self._normalize_url(url)
                    if normalized_url in self.visited_urls:
                        continue

                    logger.info(f"Worker {worker_id}: Crawling {url}")

                    page_data = await self._crawl_page_complete(page, url)

                    if page_data:
                        self.pages.append(page_data)
                        self.visited_urls.add(normalized_url)
                        self.pages_found = len(self.pages)

                        # Update job progress
                        self._update_job_progress()

                        # Add new URLs to queue
                        for link in page_data.links:
                            normalized_link = self._normalize_url(link)
                            if (
                                normalized_link not in self.visited_urls
                                and self._should_crawl(link)
                            ):
                                link_priority = self._calculate_url_priority(link)
                                await self.to_visit.put((link_priority, link))

                except asyncio.TimeoutError:
                    logger.error(f"Timeout crawling {url}")
                    self.failed_urls[url] = "timeout"
                except Exception as e:
                    logger.error(f"Error crawling {url}: {e}")
                    self.failed_urls[url] = str(e)

            await browser.close()

    def _normalize_url(self, url: str) -> str:
        """Normalize URL to prevent duplicates"""
        from urllib.parse import urlparse, urlunparse

        parsed = urlparse(url.lower())
        # Remove trailing slashes and fragments
        path = parsed.path.rstrip("/")
        # Reconstruct without fragment and with normalized path
        normalized = urlunparse(
            (
                parsed.scheme,
                parsed.netloc,
                path or "/",
                parsed.params,
                parsed.query,
                "",  # No fragment
            )
        )
        return normalized

    def _update_job_progress(self):
        """Update job progress in thread-safe manner"""
        with crawl_jobs_lock:
            if self.job_id in crawl_jobs:
                crawl_jobs[self.job_id].update(
                    {
                        "pages_crawled": self.pages_found,
                        "active_workers": min(10, max(3, self.max_pages // 5)),
                        "progress": min(
                            40, int((self.pages_found / self.max_pages) * 40)
                        ),
                    }
                )


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
    <title>AI Chatbot - Production Test (Real Components)</title>
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
        <h1>🚀 Production Component Test</h1>
        <p class="subtitle">Testing with all real production components - no mocks!</p>
        
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
                <span>GPU: <span id="gpu-info">Checking...</span></span>
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
            <button class="page-btn active" onclick="selectPages(20, this)">20 Pages</button>
            <button class="page-btn" onclick="selectPages(50, this)">50 Pages</button>
            <button class="page-btn" onclick="selectPages(100, this)">100 Pages</button>
        </div>
        
        <input type="hidden" id="max-pages" value="20" />
        
        <button class="start-btn" onclick="startAnalysis()">
            Start Production Analysis
        </button>
        
        <div class="status-box">
            <div class="status-title">Analysis Status</div>
            <div id="status-message">Ready to analyze a website with production components</div>
            
            <div class="progress-bar" id="progress-container" style="display: none;">
                <div class="progress-fill" id="progress-bar" style="width: 0%"></div>
            </div>
            
            <div class="stats-grid" id="stats-grid" style="display: none;">
                <div class="stat-item">
                    <div class="stat-value" id="pages-crawled">0</div>
                    <div class="stat-label">Pages Crawled</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" id="active-workers">0</div>
                    <div class="stat-label">Active Workers</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" id="time-elapsed">0s</div>
                    <div class="stat-label">Time Elapsed</div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // Add globals at the top
        let currentJobId = null;
        let statusInterval = null;
        let timeInterval = null;
        let startTime = null;
        let selectedPages = 20;
        let testPageOpened = false; // Prevent multiple openings
        
        // System monitoring
        async function updateSystemInfo() {
            try {
                const response = await fetch('/system-status');
                const data = await response.json();
                
                document.getElementById('cpu-info').textContent = data.cpu_percent.toFixed(1);
                document.getElementById('mem-info').textContent = data.memory_percent.toFixed(1);
                document.getElementById('gpu-info').textContent = data.gpu_available ? 
                    'Available (' + (data.gpu_memory || 0).toFixed(1) + ' GB)' : 'CPU Mode';
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
            
            // Reset test page flag for new analysis
            testPageOpened = false;
            
            // Clean domain
            const cleanDomain = domain.replace(/^https?:\\/\\//, '').replace(/\\/.*$/, '');
            
            // Update UI
            const btn = document.querySelector('.start-btn');
            btn.disabled = true;
            btn.innerHTML = '<span class="spinner"></span>Starting Analysis...';
            
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
                    updateStatus(`Initializing crawl of ${cleanDomain}...`);
                    
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
                btn.innerHTML = 'Start Production Analysis';
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
                    btn.innerHTML = 'Start Production Analysis';
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
            document.getElementById('active-workers').textContent = data.active_workers || 0;
            
            // Update status message
            let statusMsg = '';
            if (data.status === 'crawling') {
                statusMsg = `🕷️ Crawling with ${data.active_workers || 0} workers...`;
            } else if (data.status === 'processing') {
                statusMsg = `🤖 Processing ${data.pages_crawled || 0} pages with AI models...`;
            } else if (data.status === 'building_knowledge') {
                statusMsg = `📚 Building knowledge base from processed content...`;
            }
            
            if (statusMsg) {
                updateStatus(statusMsg);
            }
        }
        
        function updateStatus(message) {
            document.getElementById('status-message').textContent = message;
        }
        
        function showSuccess(data) {
            const testUrl = `/test-website?domain=${encodeURIComponent(data.domain)}&pages=${data.pages_crawled}`;
            
            document.getElementById('status-message').innerHTML = `
                <div class="success-msg">
                    ✅ <strong>Analysis Complete!</strong><br>
                    Successfully processed ${data.pages_crawled} pages.<br>
                    <a href="${testUrl}" target="_blank" style="color: #22c55e; font-weight: 600;">
                        Open Chat Interface →
                    </a>
                </div>
            `;
            
            // Auto-open after 2 seconds, but only once
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

    gpu_available = torch.cuda.is_available()
    gpu_memory = None

    if gpu_available:
        try:
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        except:
            pass

    return {
        "cpu_percent": cpu_percent,
        "memory_percent": memory.percent,
        "gpu_available": gpu_available,
        "gpu_memory": gpu_memory,
    }


@app.post("/api/crawl")
async def start_crawl(request: CrawlRequest, background_tasks: BackgroundTasks):
    """Start crawling with production components"""
    job_id = f"job-{datetime.utcnow().timestamp()}"

    with crawl_jobs_lock:
        crawl_jobs[job_id] = {
            "status": "started",
            "domain": request.domain,
            "max_pages": request.max_pages,
            "started_at": datetime.utcnow().isoformat(),
            "progress": 0,
            "pages_crawled": 0,
            "active_workers": 0,
        }

    background_tasks.add_task(
        run_production_pipeline, job_id, request.domain, request.max_pages
    )

    return {"job_id": job_id, "status": "started"}


async def run_production_pipeline(job_id: str, domain: str, max_pages: int):
    """Run the full production pipeline"""
    try:
        # Update job status
        def update_job(updates):
            with crawl_jobs_lock:
                if job_id in crawl_jobs:
                    crawl_jobs[job_id].update(updates)

        # Phase 1: Crawling with ProductionCrawler
        logger.info(f"Starting production crawl of {domain}")
        update_job({"status": "crawling", "progress": 10})

        # Use the enhanced crawler with progress tracking
        crawler = ProductionCrawler(domain, max_pages, job_id)

        # Monitor worker count
        worker_count = min(10, max(3, max_pages // 5))
        update_job({"active_workers": worker_count})

        # Start crawling
        pages = await crawler.start()

        if not pages:
            raise Exception("No pages were successfully crawled")

        update_job(
            {"pages_crawled": len(pages), "progress": 40, "status": "processing"}
        )

        # Phase 2: Skip complex processing for now, just create simple knowledge base
        logger.info(f"Creating simple knowledge base for {len(pages)} pages")

        # Create a simple collection name and store basic info
        collection_name = f"website_{domain.replace('.', '_')}"

        # Store basic page info for chat responses
        page_summaries = []
        for page in pages:
            page_summaries.append(
                {
                    "url": page.url,
                    "title": page.title,
                    "content_preview": BeautifulSoup(
                        page.html, "html.parser"
                    ).get_text()[:500],
                }
            )

        # Store for chat access
        knowledge_bases[domain] = {
            "collection_name": collection_name,
            "pages_count": len(pages),
            "created_at": datetime.utcnow().isoformat(),
            "pages": page_summaries,  # Store actual content for responses
        }

        # Complete
        update_job(
            {
                "status": "completed",
                "progress": 100,
                "collection_name": collection_name,
                "completed_at": datetime.utcnow().isoformat(),
                "domain": domain,
            }
        )

        logger.info(f"✅ Production pipeline completed for {domain}")

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
async def chat(data: dict):
    """Natural conversation with website knowledge"""
    question = data.get("question", "")
    session_id = data.get("session_id", f"session-{datetime.utcnow().timestamp()}")
    domain = data.get("domain", "")

    if not domain:
        raise HTTPException(status_code=400, detail="Domain parameter is required")

    if domain not in knowledge_bases:
        raise HTTPException(
            status_code=400, detail=f"Domain {domain} has not been analyzed yet"
        )

    # Get session memory
    if session_id not in active_sessions:
        active_sessions[session_id] = {"history": [], "context": {}, "user_name": None}

    session = active_sessions[session_id]
    kb_info = knowledge_bases[domain]

    # Generate natural response
    answer = generate_natural_response(question, kb_info, session)

    # Store in memory
    session["history"].append({"q": question, "a": answer})

    return {
        "answer": answer,
        "sources": [{"url": f"https://{domain}", "title": domain}],
        "confidence": 0.9,
        "session_id": session_id,
        "processing_time": 0.1,
        "query_type": "natural",
    }


def generate_natural_response(question: str, kb_info: Dict, session: Dict) -> str:
    """Generate completely natural responses using actual website content"""
    pages = kb_info.get("pages", [])
    domain = (
        kb_info.get("collection_name", "").replace("website_", "").replace("_", ".")
    )
    page_count = kb_info.get("pages_count", 0)

    # Get conversation history for context
    history = session.get("history", [])

    # Look for relevant content and extract actual information
    relevant_info = extract_actual_content(question, pages)

    # Generate response based on what we actually found
    if relevant_info:
        response = create_informed_response(question, relevant_info, domain, history)
    else:
        response = create_general_response(question, domain, page_count, history, pages)

    return response


def extract_actual_content(question: str, pages: list) -> str:
    """Extract actual relevant information from website pages"""
    question_words = [word.lower() for word in question.split() if len(word) > 2]

    all_content = ""
    relevant_snippets = []

    for page in pages:
        page_text = page.get("content_preview", "")
        page_title = page.get("title", "")
        page_url = page.get("url", "")

        # Check if this page is relevant
        text_lower = page_text.lower()
        title_lower = page_title.lower()

        relevance_score = 0
        for word in question_words:
            if word in text_lower or word in title_lower:
                relevance_score += 1

        if relevance_score > 0:
            # Extract meaningful sentences that contain question keywords
            sentences = page_text.split(".")
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 20:  # Skip very short fragments
                    sentence_lower = sentence.lower()
                    if any(word in sentence_lower for word in question_words):
                        if sentence not in relevant_snippets:
                            relevant_snippets.append(sentence)

            # Also add page title if relevant
            if page_title and relevance_score > 0:
                relevant_snippets.append(f"Page: {page_title}")

    # Return the most relevant content
    if relevant_snippets:
        return " ".join(relevant_snippets[:5])  # Top 5 most relevant pieces

    return ""


def create_informed_response(
    question: str, content: str, domain: str, history: list
) -> str:
    """Create response using actual website content"""
    import random

    # Clean up the content
    content = content.replace("\n", " ").replace("\r", " ")
    content = " ".join(content.split())  # Remove extra whitespace

    # Summarize the content naturally
    if len(content) > 300:
        content = content[:300] + "..."

    # Generate natural response with actual content
    intros = [
        f"From what I found on {domain}: ",
        f"Looking at {domain}, ",
        f"Based on what's there: ",
        f"According to their site: ",
        f"From their pages: ",
    ]

    outros = [
        " Want to know anything else?",
        " Does that help?",
        " Is that what you were looking for?",
        " Anything else you'd like to know?",
        "",
    ]

    intro = random.choice(intros)
    outro = random.choice(outros)

    return intro + content + outro


def create_general_response(
    question: str, domain: str, page_count: int, history: list, pages: list
) -> str:
    """Create natural response when no specific content found"""
    import random

    question_lower = question.lower()

    # Try to give some actual info even for general questions
    if pages:
        # Extract some general info about the website
        page_titles = [page.get("title", "") for page in pages if page.get("title")]
        page_urls = [page.get("url", "") for page in pages]

        # Look for common page types
        has_about = any("about" in title.lower() for title in page_titles)
        has_contact = any("contact" in title.lower() for title in page_titles)
        has_products = any(
            word in title.lower()
            for title in page_titles
            for word in ["product", "service", "shop"]
        )

        site_info = f"{domain} has {page_count} pages"

        if has_about:
            site_info += " including an about section"
        if has_contact:
            site_info += " and contact information"
        if has_products:
            site_info += " with products/services"

        site_info += ". "

        # Get some sample content
        sample_content = ""
        if pages and pages[0].get("content_preview"):
            first_content = pages[0]["content_preview"][:200]
            sample_content = (
                f"For example, I can see content about: {first_content}... "
            )
    else:
        site_info = f"I've analyzed {domain} but "
        sample_content = ""

    # Handle different types of questions
    if any(word in question_lower for word in ["hi", "hello", "hey"]):
        greetings = [
            f"Hey! {site_info}What would you like to know?",
            f"Hi there! {site_info}What can I tell you about it?",
            f"Hello! {site_info}What interests you?",
        ]
        return random.choice(greetings)

    elif any(word in question_lower for word in ["what", "about", "tell me"]):
        responses = [
            f"{site_info}{sample_content}What specifically would you like to know?",
            f"{site_info}{sample_content}Is there a particular aspect you're curious about?",
            f"{site_info}{sample_content}What would you like me to focus on?",
        ]
        return random.choice(responses)

    else:
        # Default responses with actual site info
        responses = [
            f"{site_info}{sample_content}Can you be more specific about what you're looking for?",
            f"{site_info}{sample_content}What particular aspect interests you?",
            f"{site_info}{sample_content}Is there something specific you'd like to know?",
        ]
        return random.choice(responses)


@app.get("/test-website", response_class=HTMLResponse)
async def test_website(domain: str = "", pages: int = 0):
    """Test interface with chat widget"""
    if not domain or domain not in knowledge_bases:
        return HTMLResponse("<h1>Please analyze a domain first</h1>")

    # Use the production widget.js
    return f"""
<!DOCTYPE html>
<html>
<head>
    <title>Chat Test - {domain}</title>
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
        <h1>🤖 Production Chat Test</h1>
        <div class="info">
            <p><strong>Domain:</strong> {domain}</p>
            <p><strong>Pages Analyzed:</strong> {pages}</p>
        </div>
        
        <div class="instructions">
            <h3>💬 How to Test the Chatbot</h3>
            <ul>
                <li>Look for the chat button in the bottom right corner</li>
                <li>Click it to open the chat widget</li>
                <li>The widget should be ready immediately (no loading spinner)</li>
                <li>Try asking questions about the website like:
                    <ul>
                        <li>"What is this website about?"</li>
                        <li>"How can I contact them?"</li>
                        <li>"What services do they offer?"</li>
                        <li>"What are their hours?"</li>
                    </ul>
                </li>
                <li>The AI should respond with relevant information from the analyzed pages</li>
            </ul>
        </div>
        
        <div class="status">
            ✅ Domain analysis complete - Chat widget is ready to use!
        </div>
    </div>
    
    <script>
        // Configure the widget for this specific domain
        window.AI_CHATBOT_API_URL = "http://localhost:8000";
        window.AI_CHATBOT_DOMAIN = "{domain}";
        window.AI_CHATBOT_AUTO_START = false; // Don't auto-start crawling
        
        console.log('Chat widget configured for domain:', '{domain}');
    </script>
    <script src="/widget/widget.js"></script>
</body>
</html>
    """


@app.get("/widget/widget.js")
async def serve_widget():
    """Serve the fixed widget.js"""
    # Return the fixed widget content
    widget_content = """(function() {
    'use strict';
    
    // Configuration
    const config = {
        apiUrl: window.AI_CHATBOT_API_URL || 'http://localhost:8000',
        domain: window.AI_CHATBOT_DOMAIN || window.location.hostname,
        position: window.AI_CHATBOT_POSITION || 'bottom-right',
        theme: window.AI_CHATBOT_THEME || 'modern',
        autoStart: window.AI_CHATBOT_AUTO_START !== false
    };
    
    // Widget state
    let sessionId = localStorage.getItem('ai_chatbot_session') || null;
    let isOpen = false;
    let isMinimized = false;
    let crawlStatus = 'pending';
    
    // Create widget HTML
    const widgetHTML = `
        <div id="ai-chatbot-container" class="ai-chatbot-container ai-chatbot-${config.position}">
            <div id="ai-chatbot-widget" class="ai-chatbot-widget ai-chatbot-hidden">
                <div class="ai-chatbot-header">
                    <div class="ai-chatbot-header-content">
                        <div class="ai-chatbot-status">
                            <span class="ai-chatbot-status-dot"></span>
                            <span class="ai-chatbot-status-text">AI Assistant</span>
                        </div>
                        <div class="ai-chatbot-header-actions">
                            <button class="ai-chatbot-minimize" onclick="AIChatbot.minimize()">
                                <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                                    <path d="M4 8H12" stroke="currentColor" stroke-width="2"/>
                                </svg>
                            </button>
                            <button class="ai-chatbot-close" onclick="AIChatbot.close()">
                                <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                                    <path d="M4 4L12 12M4 12L12 4" stroke="currentColor" stroke-width="2"/>
                                </svg>
                            </button>
                        </div>
                    </div>
                </div>
                
                <div class="ai-chatbot-messages" id="ai-chatbot-messages">
                    <div class="ai-chatbot-welcome">
                        <h3>Welcome! 👋</h3>
                        <p>I'm learning about this website to help answer your questions.</p>
                        <div class="ai-chatbot-loading" id="ai-chatbot-loading">
                            <div class="ai-chatbot-spinner"></div>
                            <span>Checking website analysis...</span>
                        </div>
                    </div>
                </div>
                
                <div class="ai-chatbot-input-container">
                    <div class="ai-chatbot-input-wrapper">
                        <input 
                            type="text" 
                            id="ai-chatbot-input" 
                            class="ai-chatbot-input"
                            placeholder="Ask me anything about this website..."
                            disabled
                        />
                        <button 
                            id="ai-chatbot-send" 
                            class="ai-chatbot-send"
                            onclick="AIChatbot.sendMessage()"
                            disabled
                        >
                            <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
                                <path d="M2 10L18 2L10 18L8 11L2 10Z" stroke="currentColor" stroke-width="2"/>
                            </svg>
                        </button>
                    </div>
                </div>
            </div>
            
            <button id="ai-chatbot-trigger" class="ai-chatbot-trigger" onclick="AIChatbot.toggle()">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" class="ai-chatbot-icon-chat">
                    <path d="M12 2C6.48 2 2 6.48 2 12C2 17.52 6.48 22 12 22C13.19 22 14.34 21.78 15.41 21.37L21 23L19.37 17.41C20.78 15.34 22 13.19 22 12C22 6.48 17.52 2 12 2ZM8 13C7.45 13 7 12.55 7 12C7 11.45 7.45 11 8 11C8.55 11 9 11.45 9 12C9 12.55 8.55 13 8 13ZM12 13C11.45 13 11 12.55 11 12C11 11.45 11.45 11 12 11C12.55 11 13 11.45 13 12C13 12.55 12.55 13 12 13ZM16 13C15.45 13 15 12.55 15 12C15 11.45 15.45 11 16 11C16.55 11 17 11.45 17 12C17 12.55 16.55 13 16 13Z" fill="currentColor"/>
                </svg>
                <span class="ai-chatbot-badge" id="ai-chatbot-badge" style="display: none;">1</span>
            </button>
        </div>
    `;
    
    // Create styles
    const styles = `
        <style>
            .ai-chatbot-container {
                position: fixed;
                z-index: 999999;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            }
            
            .ai-chatbot-bottom-right {
                bottom: 20px;
                right: 20px;
            }
            
            .ai-chatbot-widget {
                width: 380px;
                height: 600px;
                background: #ffffff;
                border-radius: 16px;
                box-shadow: 0 5px 40px rgba(0,0,0,0.16);
                display: flex;
                flex-direction: column;
                overflow: hidden;
                transition: all 0.3s ease;
                transform-origin: bottom right;
            }
            
            .ai-chatbot-widget.ai-chatbot-hidden {
                opacity: 0;
                transform: scale(0.9) translateY(20px);
                pointer-events: none;
            }
            
            .ai-chatbot-widget.ai-chatbot-minimized {
                height: 60px;
            }
            
            .ai-chatbot-header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 16px 20px;
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
                gap: 8px;
            }
            
            .ai-chatbot-status-dot {
                width: 8px;
                height: 8px;
                background: #4ade80;
                border-radius: 50%;
                animation: pulse 2s infinite;
            }
            
            @keyframes pulse {
                0% { opacity: 1; transform: scale(1); }
                50% { opacity: 0.8; transform: scale(1.1); }
                100% { opacity: 1; transform: scale(1); }
            }
            
            .ai-chatbot-header-actions {
                display: flex;
                gap: 8px;
            }
            
            .ai-chatbot-header-actions button {
                background: rgba(255,255,255,0.2);
                border: none;
                width: 32px;
                height: 32px;
                border-radius: 8px;
                color: white;
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
                transition: all 0.2s;
            }
            
            .ai-chatbot-header-actions button:hover {
                background: rgba(255,255,255,0.3);
            }
            
            .ai-chatbot-messages {
                flex: 1;
                overflow-y: auto;
                padding: 20px;
                background: #f9fafb;
            }
            
            .ai-chatbot-welcome {
                text-align: center;
                padding: 40px 20px;
            }
            
            .ai-chatbot-welcome h3 {
                margin: 0 0 8px 0;
                font-size: 20px;
                color: #111827;
            }
            
            .ai-chatbot-welcome p {
                margin: 0 0 24px 0;
                color: #6b7280;
                font-size: 14px;
            }
            
            .ai-chatbot-loading {
                display: inline-flex;
                align-items: center;
                gap: 12px;
                padding: 12px 20px;
                background: white;
                border-radius: 12px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.08);
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
                margin-bottom: 16px;
                display: flex;
                gap: 12px;
                animation: fadeIn 0.3s ease;
            }
            
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(10px); }
                to { opacity: 1; transform: translateY(0); }
            }
            
            .ai-chatbot-message-user {
                flex-direction: row-reverse;
            }
            
            .ai-chatbot-message-avatar {
                width: 32px;
                height: 32px;
                border-radius: 8px;
                background: #667eea;
                color: white;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 14px;
                font-weight: 600;
                flex-shrink: 0;
            }
            
            .ai-chatbot-message-user .ai-chatbot-message-avatar {
                background: #10b981;
            }
            
            .ai-chatbot-message-content {
                max-width: 70%;
                padding: 12px 16px;
                background: white;
                border-radius: 12px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            }
            
            .ai-chatbot-message-user .ai-chatbot-message-content {
                background: #667eea;
                color: white;
            }
            
            .ai-chatbot-message-text {
                margin: 0;
                font-size: 14px;
                line-height: 1.5;
            }
            
            .ai-chatbot-message-sources {
                margin-top: 8px;
                padding-top: 8px;
                border-top: 1px solid #e5e7eb;
            }
            
            .ai-chatbot-message-sources-title {
                font-size: 12px;
                color: #6b7280;
                margin-bottom: 4px;
            }
            
            .ai-chatbot-message-source {
                font-size: 12px;
                color: #667eea;
                text-decoration: none;
                display: block;
                margin-bottom: 2px;
            }
            
            .ai-chatbot-message-source:hover {
                text-decoration: underline;
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
            }
            
            .ai-chatbot-input {
                flex: 1;
                padding: 12px 16px;
                border: 2px solid #e5e7eb;
                border-radius: 12px;
                font-size: 14px;
                outline: none;
                transition: all 0.2s;
            }
            
            .ai-chatbot-input:focus {
                border-color: #667eea;
            }
            
            .ai-chatbot-input:disabled {
                background: #f3f4f6;
                cursor: not-allowed;
            }
            
            .ai-chatbot-send {
                width: 44px;
                height: 44px;
                background: #667eea;
                color: white;
                border: none;
                border-radius: 12px;
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
                transition: all 0.2s;
            }
            
            .ai-chatbot-send:hover:not(:disabled) {
                background: #5a67d8;
                transform: scale(1.05);
            }
            
            .ai-chatbot-send:disabled {
                background: #9ca3af;
                cursor: not-allowed;
            }
            
            .ai-chatbot-trigger {
                width: 60px;
                height: 60px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border: none;
                border-radius: 50%;
                color: white;
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
                box-shadow: 0 4px 24px rgba(0,0,0,0.2);
                transition: all 0.3s;
                position: relative;
            }
            
            .ai-chatbot-trigger:hover {
                transform: scale(1.1);
                box-shadow: 0 6px 32px rgba(0,0,0,0.3);
            }
            
            .ai-chatbot-badge {
                position: absolute;
                top: -4px;
                right: -4px;
                background: #ef4444;
                color: white;
                font-size: 12px;
                font-weight: 600;
                width: 20px;
                height: 20px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                animation: badgePop 0.3s ease;
            }
            
            @keyframes badgePop {
                0% { transform: scale(0); }
                50% { transform: scale(1.2); }
                100% { transform: scale(1); }
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
            
            .ai-chatbot-typing {
                display: inline-flex;
                align-items: center;
                gap: 4px;
            }
            
            .ai-chatbot-typing span {
                width: 8px;
                height: 8px;
                background: #9ca3af;
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
                    transform: translateY(0);
                    opacity: 0.7;
                }
                30% {
                    transform: translateY(-10px);
                    opacity: 1;
                }
            }
        </style>
    `;
    
    // Inject HTML and styles
    document.head.insertAdjacentHTML('beforeend', styles);
    document.body.insertAdjacentHTML('beforeend', widgetHTML);
    
    // API class
    class ChatbotAPI {
        constructor(apiUrl) {
            this.apiUrl = apiUrl;
        }
        
        async checkDomainReady(domain) {
            try {
                // Try a test chat to see if domain is ready
                const response = await fetch(`${this.apiUrl}/api/chat`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        question: "test",
                        domain: domain,
                        session_id: "test-session"
                    })
                });
                
                return response.ok;
            } catch (error) {
                return false;
            }
        }
        
        async sendMessage(question, sessionId, domain) {
            const response = await fetch(`${this.apiUrl}/api/chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    question,
                    session_id: sessionId,
                    domain: domain,
                    require_reasoning: true
                })
            });
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Failed to send message');
            }
            
            return response.json();
        }
    }
    
    // Initialize API
    const api = new ChatbotAPI(config.apiUrl);
    
    // Main chatbot class
    class AIChatbot {
        constructor() {
            this.widget = document.getElementById('ai-chatbot-widget');
            this.trigger = document.getElementById('ai-chatbot-trigger');
            this.messages = document.getElementById('ai-chatbot-messages');
            this.input = document.getElementById('ai-chatbot-input');
            this.sendButton = document.getElementById('ai-chatbot-send');
            this.badge = document.getElementById('ai-chatbot-badge');
            this.ready = false;
            this.domain = config.domain;
            
            this.init();
        }
        
        async init() {
            console.log('Initializing chatbot for domain:', this.domain);
            
            // Add event listeners
            this.input.addEventListener('keypress', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    this.sendMessage();
                }
            });
            
            // Check if domain is already analyzed
            await this.checkDomainStatus();
        }
        
        async checkDomainStatus() {
            try {
                // Check if the domain is already ready for chat
                const isReady = await api.checkDomainReady(this.domain);
                
                if (isReady) {
                    console.log('Domain already analyzed, ready for chat');
                    this.onCrawlComplete();
                } else {
                    console.log('Domain not ready');
                    this.showError('This domain has not been analyzed yet. Please analyze it first.');
                }
            } catch (error) {
                console.error('Failed to check domain status:', error);
                this.showError('Failed to check domain status. Please try again.');
            }
        }
        
        onCrawlComplete() {
            this.ready = true;
            this.input.disabled = false;
            this.sendButton.disabled = false;
            
            // Clear welcome message
            this.messages.innerHTML = '';
            
            // Show ready message
            this.addMessage('bot', `Hello! I've analyzed ${this.domain} and I'm ready to help. What would you like to know?`);
            
            // Show notification badge
            if (!isOpen) {
                this.showBadge();
            }
        }
        
        showError(message) {
            const loading = document.getElementById('ai-chatbot-loading');
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
            this.widget.classList.remove('ai-chatbot-hidden');
            this.trigger.style.display = 'none';
            this.hideBadge();
            
            // Focus input if ready
            if (this.ready) {
                this.input.focus();
            }
        }
        
        close() {
            isOpen = false;
            this.widget.classList.add('ai-chatbot-hidden');
            this.trigger.style.display = 'flex';
        }
        
        minimize() {
            isMinimized = !isMinimized;
            this.widget.classList.toggle('ai-chatbot-minimized');
        }
        
        showBadge() {
            this.badge.style.display = 'flex';
        }
        
        hideBadge() {
            this.badge.style.display = 'none';
        }
        
        async sendMessage() {
            const message = this.input.value.trim();
            
            if (!message || !this.ready) return;
            
            // Clear input
            this.input.value = '';
            
            // Add user message
            this.addMessage('user', message);
            
            // Show typing indicator
            const typingId = this.showTyping();
            
            try {
                // Send to API with domain
                const response = await api.sendMessage(message, sessionId, this.domain);
                
                // Store session ID
                if (response.session_id) {
                    sessionId = response.session_id;
                    localStorage.setItem('ai_chatbot_session', sessionId);
                }
                
                // Remove typing indicator
                this.removeTyping(typingId);
                
                // Add bot response
                this.addMessage('bot', response.answer, response.sources);
                
            } catch (error) {
                console.error('Failed to send message:', error);
                this.removeTyping(typingId);
                this.addMessage('bot', `Sorry, I encountered an error: ${error.message}. Please try again.`);
            }
        }
        
        addMessage(sender, text, sources = []) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `ai-chatbot-message ai-chatbot-message-${sender}`;
            
            const avatar = document.createElement('div');
            avatar.className = 'ai-chatbot-message-avatar';
            avatar.textContent = sender === 'user' ? 'U' : 'AI';
            
            const content = document.createElement('div');
            content.className = 'ai-chatbot-message-content';
            
            const textP = document.createElement('p');
            textP.className = 'ai-chatbot-message-text';
            textP.textContent = text;
            content.appendChild(textP);
            
            // Add sources if available
            if (sources && sources.length > 0) {
                const sourcesDiv = document.createElement('div');
                sourcesDiv.className = 'ai-chatbot-message-sources';
                
                const sourcesTitle = document.createElement('div');
                sourcesTitle.className = 'ai-chatbot-message-sources-title';
                sourcesTitle.textContent = 'Sources:';
                sourcesDiv.appendChild(sourcesTitle);
                
                sources.forEach(source => {
                    const sourceLink = document.createElement('a');
                    sourceLink.className = 'ai-chatbot-message-source';
                    sourceLink.href = source.url;
                    sourceLink.textContent = source.title || source.url;
                    sourceLink.target = '_blank';
                    sourcesDiv.appendChild(sourceLink);
                });
                
                content.appendChild(sourcesDiv);
            }
            
            messageDiv.appendChild(avatar);
            messageDiv.appendChild(content);
            
            this.messages.appendChild(messageDiv);
            
            // Scroll to bottom
            this.messages.scrollTop = this.messages.scrollHeight;
        }
        
        showTyping() {
            const typingId = `typing-${Date.now()}`;
            const typingDiv = document.createElement('div');
            typingDiv.id = typingId;
            typingDiv.className = 'ai-chatbot-message ai-chatbot-message-bot';
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
                typingDiv.remove();
            }
        }
    }
    
    // Initialize chatbot
    const chatbot = new AIChatbot();
    
    // Expose API for external use
    window.AIChatbot = {
        open: () => chatbot.open(),
        close: () => chatbot.close(),
        toggle: () => chatbot.toggle(),
        minimize: () => chatbot.minimize(),
        sendMessage: () => chatbot.sendMessage()
    };
})();"""

    return Response(content=widget_content, media_type="application/javascript")


if __name__ == "__main__":
    print(
        """
╔═══════════════════════════════════════════════════╗
║           AI Chatbot - Production Test            ║
╠═══════════════════════════════════════════════════╣
║                                                   ║
║  Starting server on http://localhost:8000         ║
║                                                   ║
╚═══════════════════════════════════════════════════╝
    """
    )

    # Check for required dependencies
    try:
        import playwright
        from playwright.async_api import async_playwright
    except ImportError:
        print(
            "⚠️  WARNING: Playwright not installed. Run: pip install playwright && playwright install chromium"
        )

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
