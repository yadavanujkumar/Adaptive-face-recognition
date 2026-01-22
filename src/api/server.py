"""
FastAPI Backend for Face Recognition System

Provides REST API endpoints for:
- Face registration
- Face recognition
- Face search
- Database management

Integrates with vector database (Qdrant/Milvus) for efficient similarity search.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import cv2
import io
from datetime import datetime
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.pipeline import FaceRecognitionPipeline


# Pydantic models for request/response
class RegistrationResponse(BaseModel):
    success: bool
    message: str
    identity: Optional[str] = None
    liveness_score: Optional[float] = None


class RecognitionResult(BaseModel):
    bbox: List[float]
    is_live: bool
    liveness_score: float
    identity: str
    similarity: float


class RecognitionResponse(BaseModel):
    success: bool
    num_faces: int
    results: List[RecognitionResult]
    processing_time_ms: Optional[float] = None


class DatabaseStats(BaseModel):
    num_identities: int
    total_embeddings: int
    identities: List[str]


# Initialize FastAPI app
app = FastAPI(
    title="Adaptive Face Recognition API",
    description="Production-grade face recognition with liveness detection",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance
pipeline: Optional[FaceRecognitionPipeline] = None


@app.on_event("startup")
async def startup_event():
    """Initialize pipeline on startup"""
    global pipeline
    
    # Configuration (can be loaded from environment variables)
    detector_path = os.environ.get('DETECTOR_MODEL_PATH')
    liveness_path = os.environ.get('LIVENESS_MODEL_PATH')
    recognition_path = os.environ.get('RECOGNITION_MODEL_PATH')
    
    pipeline = FaceRecognitionPipeline(
        detector_path=detector_path,
        liveness_path=liveness_path,
        recognition_path=recognition_path,
        liveness_threshold=0.7,
        recognition_threshold=0.6,
        use_onnx=True
    )
    
    print("Face Recognition API initialized successfully!")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Adaptive Face Recognition API",
        "version": "1.0.0",
        "endpoints": {
            "register": "/api/register",
            "recognize": "/api/recognize",
            "search": "/api/search",
            "database": "/api/database",
            "health": "/api/health"
        }
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "pipeline_loaded": pipeline is not None,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/api/register", response_model=RegistrationResponse)
async def register_face(
    image: UploadFile = File(...),
    identity: str = Form(...),
    metadata: Optional[str] = Form(None)
):
    """
    Register a new face in the database
    
    Args:
        image: Face image file
        identity: Person's name/ID
        metadata: Optional JSON metadata
    
    Returns:
        Registration status and details
    """
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")
    
    try:
        # Read image
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # Parse metadata
        import json
        meta_dict = json.loads(metadata) if metadata else {}
        
        # Register face
        success = pipeline.register_face(frame, identity, meta_dict)
        
        if success:
            return RegistrationResponse(
                success=True,
                message=f"Successfully registered {identity}",
                identity=identity
            )
        else:
            return RegistrationResponse(
                success=False,
                message="Registration failed (no face detected or liveness check failed)"
            )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/recognize", response_model=RecognitionResponse)
async def recognize_face(image: UploadFile = File(...)):
    """
    Recognize faces in an image
    
    Args:
        image: Image file containing faces
    
    Returns:
        Recognition results for all detected faces
    """
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")
    
    try:
        # Read image
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # Process frame
        import time
        start_time = time.time()
        results = pipeline.process_frame(frame)
        processing_time = (time.time() - start_time) * 1000
        
        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append(RecognitionResult(
                bbox=result['bbox'],
                is_live=result['is_live'],
                liveness_score=result['liveness_score'],
                identity=result.get('identity', 'Unknown'),
                similarity=result.get('similarity', 0.0)
            ))
        
        return RecognitionResponse(
            success=True,
            num_faces=len(results),
            results=formatted_results,
            processing_time_ms=processing_time
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/database", response_model=DatabaseStats)
async def get_database_stats():
    """
    Get database statistics
    
    Returns:
        Number of registered identities and embeddings
    """
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")
    
    db = pipeline.face_database
    
    return DatabaseStats(
        num_identities=len(set(db['identities'])),
        total_embeddings=len(db['embeddings']),
        identities=list(set(db['identities']))
    )


@app.delete("/api/database/clear")
async def clear_database():
    """
    Clear all registered faces from database
    
    Returns:
        Success status
    """
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")
    
    pipeline.face_database = {
        'embeddings': [],
        'identities': [],
        'metadata': []
    }
    
    return {"success": True, "message": "Database cleared successfully"}


@app.delete("/api/database/identity/{identity}")
async def delete_identity(identity: str):
    """
    Delete all embeddings for a specific identity
    
    Args:
        identity: Identity name to delete
    
    Returns:
        Success status and number of embeddings deleted
    """
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")
    
    db = pipeline.face_database
    
    # Find indices to delete
    indices_to_delete = [i for i, id_name in enumerate(db['identities']) if id_name == identity]
    
    if len(indices_to_delete) == 0:
        raise HTTPException(status_code=404, detail=f"Identity '{identity}' not found")
    
    # Delete in reverse order to maintain indices
    for idx in sorted(indices_to_delete, reverse=True):
        del db['embeddings'][idx]
        del db['identities'][idx]
        del db['metadata'][idx]
    
    return {
        "success": True,
        "message": f"Deleted {len(indices_to_delete)} embedding(s) for '{identity}'"
    }


if __name__ == "__main__":
    import uvicorn
    
    print("=" * 80)
    print("Starting Face Recognition API Server")
    print("=" * 80)
    print("\nAPI Endpoints:")
    print("  - POST /api/register      - Register a new face")
    print("  - POST /api/recognize     - Recognize faces in image")
    print("  - GET  /api/database      - Get database statistics")
    print("  - DELETE /api/database/clear - Clear database")
    print("  - DELETE /api/database/identity/{name} - Delete identity")
    print("  - GET  /api/health        - Health check")
    print("\nStarting server on http://localhost:8000")
    print("API docs available at http://localhost:8000/docs")
    print("=" * 80 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
