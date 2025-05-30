from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl
from typing import List, Optional, Dict
import asyncio
import uuid
from datetime import datetime
import redis
import json
from ..crawler.intelligent_crawler import IntelligentCrawler
from ..processor.multimodal_parser import MultimodalParser
from ..processor.knowledge_builder import KnowledgeBuilder
from ..chatbot.reasoning_engine import ReasoningEngine
from ..chatbot.retrieval_optimizer import OptimizedRetriever
from ..core.config import settings
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Customer Service Chatbot API",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Redis for job management
redis_client = redis.Redis.from_url(settings.redis_url, decode_responses=True)

# Initialize core components
multimodal_parser = MultimodalParser(settings.vision_models)
reasoning_engine = ReasoningEngine(settings.reasoning_models)
active_chatbots = {}  # domain -> chatbot instance

# Request/Response models
class CrawlRequest(BaseModel):
    domain: str
    max_pages: Optional[int] = 100
    priority_paths: Optional[List[str]] = []
    
class CrawlResponse(BaseModel):
    job_id: str
    status: str
    message: str
    estimated_time: int
    
class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = None
    require_reasoning: Optional[bool] = True
    
class ChatResponse(BaseModel):
    answer: str
    reasoning: Optional[str]
    sources: List[Dict]
    confidence: float
    session_id: str
    processing_time: float

# Health check
@app.get("/health")
async def health_check():
    """System health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "models_loaded": {
            "reasoning": len(reasoning_engine.models),
            "vision": hasattr(multimodal_parser, 'vl_model')
        }
    }

# Start crawling endpoint
@app.post("/api/crawl", response_model=CrawlResponse)
async def start_crawl(
    request: CrawlRequest,
    background_tasks: BackgroundTasks
):
    """Start crawling a website"""
    job_id = str(uuid.uuid4())
    
    # Validate domain
    if not request.domain:
        raise HTTPException(status_code=400, detail="Domain is required")
        
    # Check if already crawling
    existing_job = redis_client.get(f"crawl:domain:{request.domain}")
    if existing_job:
        return CrawlResponse(
            job_id=existing_job,
            status="already_crawling",
            message=f"Already crawling {request.domain}",
            estimated_time=0
        )
        
    # Start crawl in background
    background_tasks.add_task(
        crawl_and_process,
        job_id,
        request.domain,
        request.max_pages,
        request.priority_paths
    )
    
    # Store job info
    redis_client.setex(
        f"crawl:job:{job_id}",
        3600,  # 1 hour TTL
        json.dumps({
            "status": "started",
            "domain": request.domain,
            "started_at": datetime.utcnow().isoformat(),
            "progress": 0
        })
    )
    
    # Map domain to job
    redis_client.setex(f"crawl:domain:{request.domain}", 3600, job_id)
    
    # Estimate time based on pages
    estimated_time = min(300, request.max_pages * 2)  # 2 seconds per page, max 5 min
    
    return CrawlResponse(
        job_id=job_id,
        status="started",
        message=f"Started crawling {request.domain}",
        estimated_time=estimated_time
    )

# Crawl status endpoint
@app.get("/api/crawl/{job_id}")
async def get_crawl_status(job_id: str):
    """Get crawl job status"""
    job_data = redis_client.get(f"crawl:job:{job_id}")
    
    if not job_data:
        raise HTTPException(status_code=404, detail="Job not found")
        
    return json.loads(job_data)

# Chat endpoint
@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat with the AI assistant"""
    # Get or create session
    session_id = request.session_id or str(uuid.uuid4())
    
    # Get session data
    session_data = redis_client.get(f"session:{session_id}")
    if session_data:
        session = json.loads(session_data)
    else:
        session = {
            "id": session_id,
            "history": [],
            "created_at": datetime.utcnow().isoformat()
        }
        
    # Find chatbot for domain (from session or default)
    domain = session.get("domain")
    if not domain:
        # Try to find from recent crawls
        recent_crawl = redis_client.keys("crawl:domain:*")
        if recent_crawl:
            domain = recent_crawl[0].split(":")[-1]
            session["domain"] = domain
        else:
            raise HTTPException(
                status_code=400,
                detail="No domain found. Please crawl a website first."
            )
            
    # Get retriever for domain
    collection_name = f"website_{domain.replace('.', '_')}"
    
    try:
        retriever = OptimizedRetriever(collection_name)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Knowledge base not found for {domain}. Please crawl first."
        )
        
    # Get response from reasoning engine
    response = await reasoning_engine.answer_question(
        request.question,
        domain,
        retriever,
        session["history"]
    )
    
    # Update session history
    session["history"].append({
        "question": request.question,
        "answer": response.answer,
        "timestamp": datetime.utcnow().isoformat()
    })
    
    # Save session (5 hour TTL)
    redis_client.setex(
        f"session:{session_id}",
        18000,
        json.dumps(session)
    )
    
    return ChatResponse(
        answer=response.answer,
        reasoning=response.reasoning if request.require_reasoning else None,
        sources=response.sources,
        confidence=response.confidence,
        session_id=session_id,
        processing_time=response.processing_time
    )

# WebSocket for real-time updates
@app.websocket("/ws/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    """WebSocket for real-time crawl updates"""
    await websocket.accept()
    
    try:
        while True:
            # Get job status
            job_data = redis_client.get(f"crawl:job:{job_id}")
            
            if job_data:
                await websocket.send_json(json.loads(job_data))
                
                # Check if complete
                data = json.loads(job_data)
                if data["status"] in ["completed", "failed"]:
                    break
                    
            await asyncio.sleep(1)
            
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await websocket.close()

# Background task for crawling
async def crawl_and_process(
    job_id: str,
    domain: str,
    max_pages: int,
    priority_paths: List[str]
):
    """Background task to crawl and process website"""
    try:
        # Update status
        update_job_status(job_id, "crawling", {"progress": 10})
        
        # Initialize crawler
        crawler = IntelligentCrawler(domain, max_pages)
        
        # Add priority paths if provided
        if priority_paths:
            for path in priority_paths:
                await crawler.to_visit.put((0, f"https://{domain}{path}"))
                
        # Start crawling
        pages = await crawler.start()
        
        update_job_status(job_id, "processing", {
            "progress": 50,
            "pages_crawled": len(pages)
        })
        
        # Process with multimodal parser
        knowledge_builder = KnowledgeBuilder(multimodal_parser)
        collection_name = await knowledge_builder.build_knowledge_base(
            domain,
            pages
        )
        
        # Update completion
        update_job_status(job_id, "completed", {
            "progress": 100,
            "pages_crawled": len(pages),
            "collection_name": collection_name,
            "completed_at": datetime.utcnow().isoformat()
        })
        
        # Clear domain lock
        redis_client.delete(f"crawl:domain:{domain}")
        
    except Exception as e:
        logger.error(f"Crawl job {job_id} failed: {e}")
        update_job_status(job_id, "failed", {"error": str(e)})
        redis_client.delete(f"crawl:domain:{domain}")

def update_job_status(job_id: str, status: str, data: Dict):
    """Update job status in Redis"""
    current = redis_client.get(f"crawl:job:{job_id}")
    if current:
        job_data = json.loads(current)
    else:
        job_data = {}
        
    job_data.update(data)
    job_data["status"] = status
    job_data["updated_at"] = datetime.utcnow().isoformat()
    
    redis_client.setex(
        f"crawl:job:{job_id}",
        3600,
        json.dumps(job_data)
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port,
        log_level="info"
    )