import asyncio
import os
import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn
import logging
from datetime import datetime
import json

# Add the backend directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our actual components
try:
    from backend.crawler.intelligent_crawler import IntelligentCrawler
    from backend.processor.multimodal_parser import MultimodalParser
    from backend.processor.knowledge_builder import KnowledgeBuilder
    from backend.chatbot.reasoning_engine import ReasoningEngine
    from backend.chatbot.retrieval_optimizer import OptimizedRetriever
    from backend.core.config import Settings

    COMPONENTS_AVAILABLE = True
    logger.info("✅ All components loaded successfully!")
except ImportError as e:
    logger.error(f"❌ Failed to import components: {e}")
    raise RuntimeError(
        "Components must be available for testing. Please check your imports."
    )

app = FastAPI(title="AI Chatbot Domain Tester")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store state
crawl_jobs = {}
knowledge_bases = {}
active_sessions = {}


# Request models
class CrawlRequest(BaseModel):
    domain: str
    max_pages: int = 50


@app.get("/")
async def home():
    return HTMLResponse(
        """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Chatbot - Test Any Website</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f5f5f5;
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .container {
            background: white;
            padding: 3rem;
            border-radius: 15px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
            max-width: 600px;
            width: 100%;
        }
        
        h1 {
            color: #333;
            margin-bottom: 0.5rem;
            text-align: center;
        }
        
        .subtitle {
            color: #666;
            text-align: center;
            margin-bottom: 2rem;
        }
        
        .input-group {
            margin-bottom: 2rem;
        }
        
        label {
            display: block;
            margin-bottom: 0.5rem;
            color: #555;
            font-weight: 500;
        }
        
        input {
            width: 100%;
            padding: 12px 16px;
            border: 2px solid #e1e1e1;
            border-radius: 8px;
            font-size: 16px;
            transition: border-color 0.3s;
        }
        
        input:focus {
            outline: none;
            border-color: #667eea;
        }
        
        .button-group {
            display: flex;
            gap: 1rem;
            margin-bottom: 2rem;
        }
        
        button {
            flex: 1;
            padding: 12px 24px;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
        }
        
        .btn-primary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        
        .btn-primary:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
        }
        
        .btn-secondary {
            background: #f3f4f6;
            color: #374151;
        }
        
        .btn-secondary:hover {
            background: #e5e7eb;
        }
        
        .status {
            background: #f9fafb;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 2rem;
            min-height: 100px;
        }
        
        .status-title {
            font-weight: 600;
            color: #374151;
            margin-bottom: 0.5rem;
        }
        
        .status-message {
            color: #6b7280;
            font-size: 14px;
            line-height: 1.5;
        }
        
        .progress {
            margin-top: 1rem;
            background: #e5e7eb;
            height: 8px;
            border-radius: 4px;
            overflow: hidden;
        }
        
        .progress-bar {
            height: 100%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            transition: width 0.3s;
        }
        
        .examples {
            margin-top: 2rem;
            padding-top: 2rem;
            border-top: 1px solid #e5e7eb;
        }
        
        .examples-title {
            font-size: 14px;
            color: #6b7280;
            margin-bottom: 0.5rem;
        }
        
        .example-domains {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
        }
        
        .example-domain {
            background: #f3f4f6;
            padding: 6px 12px;
            border-radius: 6px;
            font-size: 14px;
            color: #4b5563;
            cursor: pointer;
            transition: all 0.2s;
        }
        
        .example-domain:hover {
            background: #667eea;
            color: white;
        }
        
        .error {
            background: #fee2e2;
            border: 1px solid #fecaca;
            color: #dc2626;
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1rem;
        }
        
        .success {
            background: #d1fae5;
            border: 1px solid #a7f3d0;
            color: #065f46;
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1rem;
        }
        
        .spinner {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid #f3f4f6;
            border-top-color: #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-right: 0.5rem;
            vertical-align: middle;
        }
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        
        .test-link {
            color: #667eea;
            text-decoration: none;
            font-weight: 600;
        }
        
        .test-link:hover {
            text-decoration: underline;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 AI Chatbot Tester</h1>
        <p class="subtitle">Enter any website domain to test the AI chatbot</p>
        
        <div class="input-group">
            <label for="domain">Website Domain</label>
            <input 
                type="text" 
                id="domain" 
                placeholder="example.com" 
                value=""
            />
        </div>
        
        <div class="input-group">
            <label for="max-pages">Max Pages to Crawl</label>
            <input 
                type="number" 
                id="max-pages" 
                placeholder="50" 
                value="20"
                min="5"
                max="100"
            />
        </div>
        
        <div class="button-group">
            <button class="btn-primary" id="start-btn" onclick="startCrawl()">
                Start Analysis
            </button>
        </div>
        
        <div class="status" id="status">
            <div class="status-title">Status</div>
            <div class="status-message" id="status-message">
                Ready to analyze a website. Enter a domain above and click "Start Analysis".
            </div>
            <div class="progress" id="progress" style="display: none;">
                <div class="progress-bar" id="progress-bar" style="width: 0%"></div>
            </div>
        </div>
        
        <div class="examples">
            <div class="examples-title">Try these examples:</div>
            <div class="example-domains">
                <div class="example-domain" onclick="setDomain('docs.python.org')">docs.python.org</div>
                <div class="example-domain" onclick="setDomain('reactjs.org')">reactjs.org</div>
                <div class="example-domain" onclick="setDomain('tailwindcss.com')">tailwindcss.com</div>
                <div class="example-domain" onclick="setDomain('fastapi.tiangolo.com')">fastapi.tiangolo.com</div>
            </div>
        </div>
    </div>
    
    <script>
        let currentJobId = null;
        let checkInterval = null;
        
        function setDomain(domain) {
            document.getElementById('domain').value = domain;
        }
        
        async function startCrawl() {
            const domain = document.getElementById('domain').value.trim();
            const maxPages = parseInt(document.getElementById('max-pages').value) || 20;
            
            if (!domain) {
                showError('Please enter a domain');
                return;
            }
            
            // Clean domain (remove protocol if present)
            const cleanDomain = domain.replace(/^https?:\\/\\//, '').replace(/\\/.*$/, '');
            
            const startBtn = document.getElementById('start-btn');
            startBtn.disabled = true;
            startBtn.innerHTML = '<span class="spinner"></span>Starting...';
            
            showStatus('Starting analysis...', 0);
            
            try {
                const response = await fetch('/api/crawl', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ 
                        domain: cleanDomain,
                        max_pages: maxPages 
                    })
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    currentJobId = data.job_id;
                    showStatus(`Crawling ${cleanDomain}...`, 10);
                    
                    // Start checking status
                    checkCrawlStatus();
                    checkInterval = setInterval(checkCrawlStatus, 2000);
                } else {
                    throw new Error(data.detail || 'Failed to start crawl');
                }
                
            } catch (error) {
                showError(`Error: ${error.message}`);
                startBtn.disabled = false;
                startBtn.innerHTML = 'Start Analysis';
            }
        }
        
        async function checkCrawlStatus() {
            if (!currentJobId) return;
            
            try {
                const response = await fetch(`/api/crawl/${currentJobId}`);
                const data = await response.json();
                
                if (data.status === 'crawling') {
                    const progress = data.progress || 20;
                    showStatus(`Crawling... (${data.pages_crawled || 0} pages found)`, progress);
                } else if (data.status === 'processing') {
                    const progress = data.progress || 50;
                    showStatus(`Processing content with AI... (${data.pages_crawled || 0} pages)`, progress);
                } else if (data.status === 'completed') {
                    clearInterval(checkInterval);
                    const testUrl = `/test-website?domain=${encodeURIComponent(data.domain)}&pages=${data.pages_crawled}`;
                    showSuccess(`Analysis complete! ${data.pages_crawled || 0} pages processed. <a href="${testUrl}" target="_blank" class="test-link">Open Test Page</a>`);
                    
                    // Open test page automatically
                    setTimeout(() => {
                        window.open(testUrl, '_blank');
                    }, 1000);
                    
                    // Reset button
                    const startBtn = document.getElementById('start-btn');
                    startBtn.disabled = false;
                    startBtn.innerHTML = 'Start Analysis';
                } else if (data.status === 'failed') {
                    clearInterval(checkInterval);
                    showError(`Analysis failed: ${data.error || 'Unknown error'}`);
                    
                    // Reset button
                    const startBtn = document.getElementById('start-btn');
                    startBtn.disabled = false;
                    startBtn.innerHTML = 'Start Analysis';
                }
            } catch (error) {
                console.error('Status check error:', error);
            }
        }
        
        function showStatus(message, progress) {
            const statusEl = document.getElementById('status');
            const messageEl = document.getElementById('status-message');
            const progressEl = document.getElementById('progress');
            const progressBar = document.getElementById('progress-bar');
            
            statusEl.className = 'status';
            messageEl.innerHTML = message;
            
            if (progress !== undefined) {
                progressEl.style.display = 'block';
                progressBar.style.width = progress + '%';
            } else {
                progressEl.style.display = 'none';
            }
        }
        
        function showError(message) {
            const statusEl = document.getElementById('status');
            statusEl.className = 'status error';
            document.getElementById('status-message').innerHTML = message;
            document.getElementById('progress').style.display = 'none';
        }
        
        function showSuccess(message) {
            const statusEl = document.getElementById('status');
            statusEl.className = 'status success';
            document.getElementById('status-message').innerHTML = message;
            document.getElementById('progress').style.display = 'none';
        }
    </script>
</body>
</html>
    """
    )


@app.post("/api/crawl")
async def start_crawl(request: CrawlRequest, background_tasks: BackgroundTasks):
    """Start crawling a real domain"""
    job_id = f"job-{datetime.utcnow().timestamp()}"

    # Store job info
    crawl_jobs[job_id] = {
        "status": "started",
        "domain": request.domain,
        "max_pages": request.max_pages,
        "started_at": datetime.utcnow().isoformat(),
        "progress": 0,
        "pages_crawled": 0,
    }

    # Start actual crawling in background
    background_tasks.add_task(crawl_website, job_id, request.domain, request.max_pages)

    return {
        "job_id": job_id,
        "status": "started",
        "message": f"Started crawling {request.domain}",
    }


async def crawl_website(job_id: str, domain: str, max_pages: int):
    """Actually crawl and process a website"""
    try:
        # Update status
        crawl_jobs[job_id]["status"] = "crawling"
        crawl_jobs[job_id]["progress"] = 10

        # Initialize crawler with correct max_pages
        logger.info(f"Starting crawl of {domain} with max {max_pages} pages")
        crawler = IntelligentCrawler(domain, max_pages=max_pages)

        # Crawl website
        pages = await crawler.start()

        # Update progress
        crawl_jobs[job_id]["pages_crawled"] = len(pages)
        crawl_jobs[job_id]["progress"] = 40
        crawl_jobs[job_id]["status"] = "processing"

        logger.info(f"Crawled {len(pages)} pages from {domain}")

        # Process with multimodal parser
        logger.info(f"Processing {len(pages)} pages")
        parser = MultimodalParser({})  # Use default config
        builder = KnowledgeBuilder(parser)

        # Build knowledge base
        collection_name = await builder.build_knowledge_base(domain, pages)

        # Store knowledge base reference
        knowledge_bases[domain] = {
            "collection_name": collection_name,
            "pages_count": len(pages),
            "created_at": datetime.utcnow().isoformat(),
        }

        # Update job status
        crawl_jobs[job_id]["status"] = "completed"
        crawl_jobs[job_id]["progress"] = 100
        crawl_jobs[job_id]["collection_name"] = collection_name
        crawl_jobs[job_id]["completed_at"] = datetime.utcnow().isoformat()

        logger.info(f"Crawl completed for {domain} - {len(pages)} pages processed")

    except Exception as e:
        logger.error(f"Crawl failed for {domain}: {e}")
        crawl_jobs[job_id]["status"] = "failed"
        crawl_jobs[job_id]["error"] = str(e)
        crawl_jobs[job_id]["progress"] = 0


@app.get("/api/crawl/{job_id}")
async def get_crawl_status(job_id: str):
    """Get crawl job status"""
    if job_id not in crawl_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    return crawl_jobs[job_id]


@app.post("/api/chat")
async def chat(data: dict):
    """Chat endpoint that works with any crawled domain"""
    question = data.get("question", "")
    session_id = data.get("session_id", f"session-{datetime.utcnow().timestamp()}")
    domain = data.get("domain", "")

    if not domain:
        raise HTTPException(status_code=400, detail="Domain is required")

    if domain not in knowledge_bases:
        raise HTTPException(
            status_code=400, detail=f"Domain {domain} has not been analyzed yet"
        )

    try:
        # Get knowledge base
        kb_info = knowledge_bases[domain]
        collection_name = kb_info["collection_name"]

        # Initialize retriever
        retriever = OptimizedRetriever(collection_name)

        # Get conversation history
        if session_id not in active_sessions:
            active_sessions[session_id] = []

        # Get reasoning engine response
        reasoning_engine = ReasoningEngine({})
        response = await reasoning_engine.answer_question(
            question, domain, retriever, active_sessions[session_id]
        )

        # Store in conversation history
        active_sessions[session_id].append(
            {"question": question, "answer": response.answer}
        )

        return {
            "answer": response.answer,
            "sources": [
                {"url": f"https://{domain}", "title": s.get("title", domain)}
                for s in response.sources[:3]
            ],
            "confidence": response.confidence,
            "session_id": session_id,
            "processing_time": response.processing_time,
        }

    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error processing question: {str(e)}"
        )


@app.get("/test-website", response_class=HTMLResponse)
async def test_website(domain: str = "", pages: int = 0):
    """Test page that works with any domain"""
    if not domain:
        return HTMLResponse(
            """
        <html>
        <body style="font-family: Arial; padding: 40px; text-align: center;">
            <h1>No Domain Specified</h1>
            <p>Please go back and analyze a website first.</p>
            <a href="/" style="color: #667eea;">← Back to Home</a>
        </body>
        </html>
        """
        )

    # Check if domain has been processed
    if domain not in knowledge_bases:
        return HTMLResponse(
            f"""
        <html>
        <body style="font-family: Arial; padding: 40px; text-align: center;">
            <h1>Domain Not Analyzed</h1>
            <p>The domain '{domain}' has not been analyzed yet.</p>
            <a href="/" style="color: #667eea;">← Back to Home</a>
        </body>
        </html>
        """
        )

    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Chatbot - {domain}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }}
        
        .header {{
            background: white;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 2rem;
            text-align: center;
        }}
        
        h1 {{
            margin: 0 0 0.5rem 0;
            color: #333;
        }}
        
        .domain {{
            color: #667eea;
            font-size: 1.5rem;
            font-weight: 600;
        }}
        
        .info {{
            color: #666;
            margin-top: 1rem;
        }}
        
        .stats {{
            display: inline-block;
            background: #f3f4f6;
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 14px;
            color: #4b5563;
            margin-top: 10px;
        }}
        
        .test-controls {{
            position: fixed;
            top: 20px;
            right: 20px;
            background: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            max-width: 300px;
        }}
        
        .test-controls h3 {{
            margin: 0 0 1rem 0;
            color: #333;
        }}
        
        .test-question {{
            display: block;
            width: 100%;
            padding: 0.75rem;
            margin: 0.5rem 0;
            background: #f3f4f6;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            text-align: left;
            transition: all 0.2s;
        }}
        
        .test-question:hover {{
            background: #667eea;
            color: white;
            transform: translateX(5px);
        }}
        
        .status {{
            background: #d1fae5;
            border: 1px solid #a7f3d0;
            color: #065f46;
            padding: 1rem;
            border-radius: 8px;
            margin: 2rem auto;
            max-width: 600px;
            text-align: center;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>AI Chatbot Test Interface</h1>
        <div class="domain">{domain}</div>
        <div class="stats">{pages} pages analyzed</div>
        <div class="info">
            Chat with the AI assistant about this website. The chatbot has analyzed the content and can answer questions.
        </div>
    </div>
    
    <div class="status">✅ Successfully analyzed {domain}. The chatbot is ready to answer questions!</div>
    
    <div class="test-controls">
        <h3>Sample Questions</h3>
        <button class="test-question" onclick="askQuestion('What is this website about?')">
            What is this website about?
        </button>
        <button class="test-question" onclick="askQuestion('What are the main sections or features?')">
            What are the main sections?
        </button>
        <button class="test-question" onclick="askQuestion('How can I contact them?')">
            How can I contact them?
        </button>
        <button class="test-question" onclick="askQuestion('What products or services do they offer?')">
            What products/services do they offer?
        </button>
        <button class="test-question" onclick="askQuestion('Tell me about their pricing')">
            Tell me about pricing
        </button>
        <button class="test-question" onclick="askQuestion('What are the key features?')">
            What are the key features?
        </button>
    </div>
    
    <!-- Load the chatbot widget -->
    <script>
        window.AI_CHATBOT_API_URL = "http://localhost:8000";
        window.AI_CHATBOT_DOMAIN = "{domain}";
        
        function askQuestion(question) {{
            if (window.AIChatbotWidget) {{
                window.AIChatbotWidget.open();
                setTimeout(() => {{
                    window.AIChatbotWidget.sendMessage(question);
                }}, 500);
            }}
        }}
    </script>
    <script src="http://localhost:8000/widget/widget.js"></script>
</body>
</html>
    """


@app.get("/widget/widget.js", response_class=HTMLResponse)
async def serve_widget():
    """Serve the chatbot widget"""
    return """
(function() {
    'use strict';
    
    // Check if widget already exists
    if (window.AIChatbotWidget) return;
    
    // Get configuration
    const API_URL = window.AI_CHATBOT_API_URL || 'http://localhost:8000';
    const DOMAIN = window.AI_CHATBOT_DOMAIN || window.location.hostname;
    
    // ... (styles remain the same) ...
    
    // Widget functionality
    const chatWindow = document.getElementById('chatWindow');
    const chatBubble = document.getElementById('chatBubble');
    const closeChat = document.getElementById('closeChat');
    const chatInput = document.getElementById('chatInput');
    const sendButton = document.getElementById('sendButton');
    const chatMessages = document.getElementById('chatMessages');
    
    let isOpen = false;
    let sessionId = localStorage.getItem('chat_session_id') || null;
    
    // Toggle chat window
    function toggleChat() {
        isOpen = !isOpen;
        if (isOpen) {
            chatWindow.classList.remove('hidden');
            chatBubble.style.transform = 'scale(0)';
            setTimeout(() => chatInput.focus(), 300);
            
            // Add welcome message if first time
            if (chatMessages.children.length === 0) {
                addMessage('Hello! I\\'m your AI assistant for ' + DOMAIN + '. I\\'ve analyzed this website and can answer any questions you have about it.');
            }
        } else {
            chatWindow.classList.add('hidden');
            chatBubble.style.transform = 'scale(1)';
        }
    }
    
    // Add message to chat
    function addMessage(text, isUser = false) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'ai-chat-message ' + (isUser ? 'user' : 'bot');
        
        messageDiv.innerHTML = '<div class="ai-chat-avatar">' + (isUser ? 'U' : 'AI') + '</div>' +
                              '<div class="ai-chat-bubble-message">' + text + '</div>';
        
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    // ... (rest of the widget code remains the same)
})();
    """
    
    
if __name__ == "__main__":
    print(
        """
╔════════════════════════════════════════════════════════════════╗
║              AI Chatbot - Real Website Tester                  ║
╚════════════════════════════════════════════════════════════════╝

Starting server...

To test with any website:
1. Open http://localhost:8000/ in your browser
2. Enter any domain (e.g., python.org, github.com, etc.)
3. Set the number of pages to crawl
4. Click "Start Analysis"
5. Wait for the crawl to complete
6. The test page will open automatically with the chatbot ready

The chatbot will answer questions based on the actual website content!

Press Ctrl+C to stop the server.
    """
    )

    uvicorn.run(app, host="0.0.0.0", port=8000)
