from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
import time
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app with optimizations
app = FastAPI(
    title="AI Recommendation Service",
    description="AI-powered CV and job matching service",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Optimized CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8501", "http://localhost:8081"],  # Specific origins
    allow_credentials=True,
    allow_methods=["GET", "POST"],  # Specific methods
    allow_headers=["*"],
)

# Global variables
model_ready = False
startup_time = None

class SimilarityRequest(BaseModel):
    cv_text: str
    job_text: str

@app.get("/")
async def root():
    return {
        "message": "🤖 AI Recommendation Service is running!",
        "status": "healthy",
        "version": "1.0.0",
        "model_ready": model_ready,
        "uptime_seconds": time.time() - startup_time if startup_time else 0
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if model_ready else "loading",
        "service": "ai-recommendation-service",
        "model_ready": model_ready
    }

@app.post("/api/v1/similarity")
async def calculate_similarity(request: SimilarityRequest):
    """Calculate similarity with async processing"""
    
    if not model_ready:
        raise HTTPException(
            status_code=503, 
            detail="AI model is still loading. Please try again in a few seconds."
        )
    
    try:
        from app.models.ai_models import ai_model_manager
        
        start_time = time.time()
        
        # Use async processing for better performance
        similarity = await ai_model_manager.calculate_similarity_async(
            request.cv_text, 
            request.job_text
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        return {
            "similarity_score": round(similarity * 100, 2),
            "similarity_raw": similarity,
            "processing_time_ms": round(processing_time, 2),
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error calculating similarity: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.post("/api/v1/similarity/sync")
async def calculate_similarity_sync(request: SimilarityRequest):
    """Synchronous version for comparison"""
    
    if not model_ready:
        raise HTTPException(status_code=503, detail="AI model is still loading.")
    
    try:
        from app.models.ai_models import ai_model_manager
        
        start_time = time.time()
        similarity = ai_model_manager.calculate_similarity(request.cv_text, request.job_text)
        processing_time = (time.time() - start_time) * 1000
        
        return {
            "similarity_score": round(similarity * 100, 2),
            "similarity_raw": similarity,
            "processing_time_ms": round(processing_time, 2),
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error in sync similarity: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup_event():
    global model_ready, startup_time
    startup_time = time.time()
    
    logger.info("🚀 Starting Optimized AI Recommendation Service...")
    
    try:
        # Load model in background to avoid blocking startup
        def load_model():
            from app.models.ai_models import ai_model_manager
            model = ai_model_manager.get_model()
            
            # Warm up with test calculation
            test_similarity = ai_model_manager.calculate_similarity(
                "test python developer", 
                "python developer position"
            )
            return test_similarity
        
        # Run model loading in thread pool
        loop = asyncio.get_event_loop()
        test_similarity = await loop.run_in_executor(None, load_model)
        
        model_ready = True
        load_time = time.time() - startup_time
        
        logger.info(f"✅ AI model ready in {load_time:.2f}s! Test similarity: {test_similarity:.4f}")
        
    except Exception as e:
        logger.error(f"❌ Failed to load AI model: {e}")
        model_ready = False

@app.on_event("shutdown")
async def shutdown_event():
    global model_ready
    model_ready = False
    logger.info("🛑 Shutting down AI Recommendation Service")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        workers=1,
        loop="asyncio"
    )