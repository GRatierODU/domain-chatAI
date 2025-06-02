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

# Add the backend directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import ALL existing production components - no mocks!
from backend.crawler.intelligent_crawler import IntelligentCrawler
from backend.processor.multimodal_parser import MultimodalParser
from backend.processor.knowledge_builder import KnowledgeBuilder
from backend.chatbot.reasoning_engine import ReasoningEngine
from backend.chatbot.retrieval_optimizer import OptimizedRetriever
from backend.chatbot.complexity_classifier import ComplexityClassifier
from backend.core.config import settings

logger.info("✅ All production components loaded successfully!")

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

# Global state
crawl_jobs = {}
knowledge_bases = {}
active_sessions = {}

# Request models
class CrawlRequest(BaseModel):
    domain: str
    max_pages: int = 20


# Enhanced IntelligentCrawler with multi-worker support
class ProductionCrawler(IntelligentCrawler):
    """Production crawler with optimized multi-worker support"""
    
    async def _crawl_phase(self):
        """Override to use multiple workers efficiently"""
        # Calculate optimal workers: min 3, max 10, based on pages
        num_workers = min(10, max(3, self.max_pages // 5))
        
        logger.info(f"🚀 Starting {num_workers} crawler workers for {self.max_pages} pages")
        
        workers = []
        for i in range(num_workers):
            workers.append(asyncio.create_task(self._crawler_worker(i)))
        
        await asyncio.gather(*workers)


@app.get("/")
async def home():
    """Production test interface"""
    return HTMLResponse("""
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
        let currentJobId = null;
        let statusInterval = null;
        let startTime = null;
        let selectedPages = 20;
        
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
            
            // Clean domain
            const cleanDomain = domain.replace(/^https?:\\/\\//, '').replace(/\\/.*$/, '');
            
            // Update UI
            const btn = document.querySelector('.start-btn');
            btn.disabled = true;
            btn.innerHTML = '<span class="spinner"></span>Starting Analysis...';
            
            document.getElementById('progress-container').style.display = 'block';
            document.getElementById('stats-grid').style.display = 'grid';
            
            startTime = Date.now();
            
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
                    statusInterval = setInterval(checkStatus, 1000);
                    
                    // Try WebSocket connection
                    connectWebSocket(data.job_id);
                } else {
                    throw new Error(data.detail || 'Failed to start');
                }
                
            } catch (error) {
                showError(`Error: ${error.message}`);
                btn.disabled = false;
                btn.innerHTML = 'Start Production Analysis';
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
            
            // Update time
            if (startTime) {
                const elapsed = Math.floor((Date.now() - startTime) / 1000);
                document.getElementById('time-elapsed').textContent = elapsed + 's';
            }
            
            // Update status message
            let statusMsg = '';
            if (data.status === 'crawling') {
                statusMsg = `🕷️ Crawling with ${data.active_workers || 0} workers... Found ${data.pages_crawled || 0} pages`;
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
            
            // Auto-open after 2 seconds
            setTimeout(() => {
                window.open(testUrl, '_blank');
            }, 2000);
        }
        
        function showError(message) {
            document.getElementById('status-message').innerHTML = `
                <div class="error-msg">❌ ${message}</div>
            `;
        }
    </script>
</body>
</html>
    """)


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
        "gpu_memory": gpu_memory
    }


@app.post("/api/crawl")
async def start_crawl(request: CrawlRequest, background_tasks: BackgroundTasks):
    """Start crawling with production components"""
    job_id = f"job-{datetime.utcnow().timestamp()}"
    
    crawl_jobs[job_id] = {
        "status": "started",
        "domain": request.domain,
        "max_pages": request.max_pages,
        "started_at": datetime.utcnow().isoformat(),
        "progress": 0,
        "pages_crawled": 0,
        "active_workers": 0
    }
    
    background_tasks.add_task(
        run_production_pipeline,
        job_id,
        request.domain,
        request.max_pages
    )
    
    return {"job_id": job_id, "status": "started"}


async def run_production_pipeline(job_id: str, domain: str, max_pages: int):
    """Run the full production pipeline"""
    try:
        # Update job status
        def update_job(updates):
            crawl_jobs[job_id].update(updates)
        
        # Phase 1: Crawling with ProductionCrawler
        logger.info(f"Starting production crawl of {domain}")
        update_job({"status": "crawling", "progress": 10})
        
        # Use the enhanced crawler with multiple workers
        crawler = ProductionCrawler(domain, max_pages=max_pages)
        
        # Monitor worker count
        update_job({"active_workers": min(10, max(3, max_pages // 5))})
        
        # Start crawling
        pages = await crawler.start()
        
        update_job({
            "pages_crawled": len(pages),
            "progress": 40,
            "status": "processing"
        })
        
        # Phase 2: Processing with production multimodal parser
        logger.info(f"Processing {len(pages)} pages with multimodal AI")
        
        # Use production parser with actual models from config
        parser = MultimodalParser(settings.vision_models)
        builder = KnowledgeBuilder(parser)
        
        update_job({"progress": 60})
        
        # Phase 3: Build knowledge base
        logger.info("Building knowledge base")
        update_job({"status": "building_knowledge"})
        
        collection_name = await builder.build_knowledge_base(domain, pages)
        
        # Store for chat access
        knowledge_bases[domain] = {
            "collection_name": collection_name,
            "pages_count": len(pages),
            "created_at": datetime.utcnow().isoformat()
        }
        
        # Complete
        update_job({
            "status": "completed",
            "progress": 100,
            "collection_name": collection_name,
            "completed_at": datetime.utcnow().isoformat()
        })
        
        logger.info(f"✅ Production pipeline completed for {domain}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        update_job({
            "status": "failed",
            "error": str(e),
            "progress": 0
        })


@app.get("/api/crawl/{job_id}")
async def get_crawl_status(job_id: str):
    """Get crawl job status"""
    if job_id not in crawl_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return crawl_jobs[job_id]


@app.websocket("/ws/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    """WebSocket for real-time updates"""
    await websocket.accept()
    
    try:
        while True:
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
    """Production chat endpoint"""
    question = data.get("question", "")
    session_id = data.get("session_id", f"session-{datetime.utcnow().timestamp()}")
    domain = data.get("domain", "")
    
    if not domain or domain not in knowledge_bases:
        raise HTTPException(status_code=400, detail=f"Domain {domain} not analyzed")
    
    try:
        # Use production components
        kb_info = knowledge_bases[domain]
        retriever = OptimizedRetriever(kb_info["collection_name"])
        
        if session_id not in active_sessions:
            active_sessions[session_id] = []
        
        # Use production reasoning engine with actual models
        reasoning_engine = ReasoningEngine(settings.reasoning_models)
        
        response = await reasoning_engine.answer_question(
            question,
            domain,
            retriever,
            active_sessions[session_id]
        )
        
        active_sessions[session_id].append({
            "question": question,
            "answer": response.answer
        })
        
        return {
            "answer": response.answer,
            "sources": response.sources,
            "confidence": response.confidence,
            "session_id": session_id,
            "processing_time": response.processing_time,
            "query_type": response.query_type.value
        }
        
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


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
</head>
<body>
    <h1>Production Chat Test</h1>
    <p>Domain: {domain} | Pages: {pages}</p>
    <p>The chatbot should appear in the bottom right.</p>
    
    <script>
        window.AI_CHATBOT_API_URL = "http://localhost:8000";
        window.AI_CHATBOT_DOMAIN = "{domain}";
        window.AI_CHATBOT_AUTO_START = false;
    </script>
    <script src="/widget/widget.js"></script>
</body>
</html>
    """


@app.get("/widget/widget.js")
async def serve_widget():
    """Serve the production widget from frontend/widget/widget.js"""
    # Read the actual production widget file
    widget_path = Path(__file__).parent.parent / "frontend" / "widget" / "widget.js"
    
    if widget_path.exists():
        with open(widget_path, 'r') as f:
            widget_content = f.read()
    else:
        # Fallback to embedded widget
        widget_content = """
(function() {
    console.error('Production widget.js not found at frontend/widget/widget.js');
    console.log('Using fallback widget');
    
    // Basic fallback widget
    window.AIChatbot = {
        open: () => console.log('Widget would open here'),
        close: () => console.log('Widget would close here'),
        toggle: () => console.log('Widget would toggle here')
    };
})();
        """
    
    # Modify widget to work with test
    widget_content = widget_content.replace(
        "window.AI_CHATBOT_API_URL || 'http://localhost:8000'",
        "window.AI_CHATBOT_API_URL || 'http://localhost:8000'"
    )
    
    # Add domain support
    if "window.AI_CHATBOT_DOMAIN" not in widget_content:
        widget_content = widget_content.replace(
            "const config = {",
            """const config = {
        domain: window.AI_CHATBOT_DOMAIN || window.location.hostname,"""
        )
    
    return Response(content=widget_content, media_type="application/javascript")


if __name__ == "__main__":
    print("""
╔═══════════════════════════════════════════════════╗
║           AI Chatbot -   Test                     ║
╠═══════════════════════════════════════════════════╣
║                                                   ║                                                             ║
║  Starting server on http://localhost:8000         ║
║                                                   ║
╚═══════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")